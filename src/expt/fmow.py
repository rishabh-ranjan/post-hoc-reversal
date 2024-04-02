import argparse
import json
from pathlib import Path

import timm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import wandb
import wilds
from tqdm.auto import tqdm

from . import utils


def main(args):
    print(f"{args=}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    kv = utils.KeyValStore(args.root, device)
    kv.save(args, "args")

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(utils.IMAGENET_MEAN, utils.IMAGENET_STD),
        ]
    )

    dataset = wilds.get_dataset(
        dataset=args.dataset, root_dir="data/wilds", download=True
    )
    datasets = {
        "train": dataset.get_subset("train", transform=transform),
        "val": dataset.get_subset("id_val", transform=transform),
        "test": dataset.get_subset("id_test", transform=transform),
    }
    num_classes = dataset.n_classes

    print(f"{num_classes=}")
    print(f"{len(datasets['train'])=}")
    print(f"{len(datasets['val'])=}")
    print(f"{len(datasets['test'])=}")

    eval_loaders = {
        split: DataLoader(
            datasets[split],
            shuffle=False,
            batch_size=args.eval_batch_size,
            num_workers=args.eval_num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=True,
        )
        for split in ["val", "test"]
    }

    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.train_num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    net = timm.create_model(
        args.model, pretrained=args.pretrained, num_classes=num_classes
    )
    net = net.to(device, memory_format=torch.channels_last)

    opt = optim.Adam(net.parameters(), lr=args.lr)

    if args.lrs == "exponential":
        lrs = optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay)
    elif args.lrs == "cosine":
        lrs = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.wandb:
        run = wandb.init(
            project=args.wandb,
            name=f"{args.dataset}__{args.root[-3:]}",
            config=args,
        )
        print(f"{run.name=}")

    def pred(net, split):
        net.eval()
        with torch.no_grad():
            yhats = []
            ys = []
            for x, y, _ in eval_loaders[split]:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                ys.append(y)
                yhat = net(x)
                yhats.append(yhat)
            yhat = torch.cat(yhats)
            y = torch.cat(ys)
            return yhat, y

    evaluator = utils.Evaluator(
        args.root, pred, net, update_bn_loader=eval_loaders["val"]
    )

    step = 0
    for epoch in tqdm(range(args.num_epochs), desc="epochs", leave=False):
        net.train()
        for x, y, _ in tqdm(train_loader, desc="batches", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=args.amp):
                yhat = net(x)
                loss = F.cross_entropy(yhat, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            step += 1

            if args.wandb:
                wandb.log(
                    {
                        "epochs": step / len(train_loader),
                        "lr": opt.param_groups[0]["lr"],
                        "loss": loss,
                        "err/train_batch": (yhat.argmax(-1) != y).float().mean(),
                        "nll/train_batch": F.cross_entropy(yhat, y),
                    },
                    step=step,
                )

        lrs.step()

        evaluator(epoch + 1, net)
        if args.wandb:
            wandb.log({k: v[-1] for k, v in evaluator.stage.items()}, step)

    evaluator.finalize()

    if args.wandb:
        wandb.finish()

    Path(f"{args.root}/done").touch()


if __name__ == "__main__":
    import time

    root = f"runs/dev/{time.time_ns()}"
    Path(root).mkdir(parents=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=root)
    parser.add_argument("--wandb", type=str, default="")
    parser.add_argument("--dataset", type=str, default="fmow")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--eval_num_workers", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_num_workers", type=int, default=64)
    parser.add_argument("--model", type=str, default="densenet121")
    parser.add_argument(
        "--pretrained", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lrs", type=str, default="exponential")
    parser.add_argument("--lr_decay", type=float, default=0.96)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    main(args)
