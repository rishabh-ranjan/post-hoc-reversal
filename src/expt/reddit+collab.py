import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import random_split
import torch_geometric as pyg
import wandb
from tqdm.auto import tqdm

from . import utils

### fixed in pytorch 2.0.1

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

###


def main(args):
    print(f"{args=}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    kv = utils.KeyValStore(args.root, device)  ###
    kv.save(args, "args")

    if args.dataset == "COLLAB":
        transform = pyg.transforms.OneHotDegree(max_degree=499)
    else:
        transform = pyg.transforms.Constant()

    dataset = pyg.datasets.TUDataset(
        root="data/pyg", name=args.dataset, transform=transform
    )

    test_sz = int(0.1 * len(dataset))
    train_sz = len(dataset) - 2 * test_sz
    datasets = {}
    datasets["train"], datasets["val"], datasets["test"] = random_split(
        dataset,
        [train_sz, test_sz, test_sz],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"{dataset.num_classes=}")
    print(f"{len(datasets['train'])=}")
    print(f"{len(datasets['val'])=}")
    print(f"{len(datasets['test'])=}")

    eval_loaders = {
        split: pyg.loader.DataLoader(
            datasets[split],
            shuffle=False,
            batch_size=args.eval_batch_size,
            num_workers=args.eval_num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=True,
        )
        for split in ["val", "test"]
    }

    train_loader = pyg.loader.DataLoader(
        datasets["train"],
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.train_num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    if args.dataset == "COLLAB":
        gnn = pyg.nn.models.GIN(
            500,
            args.hidden_channels,
            args.num_gnn_layers,
            norm="graph_norm",
        )
        pool = pyg.nn.pool.global_mean_pool

    else:
        gnn = pyg.nn.models.GCN(
            1,
            args.hidden_channels,
            args.num_gnn_layers,
            norm="graph_norm",
        )
        pool = pyg.nn.pool.global_max_pool

    mlp = pyg.nn.models.MLP(
        in_channels=args.hidden_channels,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_mlp_layers,
        norm="instance_norm",
    )

    net = pyg.nn.Sequential(
        "x, edge_index, batch",
        [
            (gnn, "x, edge_index -> x"),
            (pool, "x, batch -> x"),
            (mlp, "x -> x"),
        ],
    )
    net = net.to(device)

    opt = optim.Adam(net.parameters(), lr=args.lr)
    lrs = optim.lr_scheduler.ExponentialLR(opt, args.lr_decay)

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
            for batch in eval_loaders[split]:
                batch = batch.to(device, non_blocking=True)
                ys.append(batch.y)
                yhat = net(batch.x, batch.edge_index, batch.batch)
                yhats.append(yhat)
            yhat = torch.cat(yhats)
            y = torch.cat(ys)
            return yhat, y

    evaluator = utils.Evaluator(args.root, pred, net, update_bn_loader=None)

    step = 0
    for epoch in tqdm(range(args.epochs), leave=False):
        net.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=args.amp):
                yhat = net(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(yhat, batch.y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            step += 1

            if args.wandb:
                wandb.log(
                    {
                        "epochs": step / len(train_loader),
                        "lr": opt.param_groups[0]["lr"],
                        "loss": loss,
                        "err/train_batch": (yhat.argmax(-1) != batch.y).float().mean(),
                        "nll/train_batch": F.cross_entropy(yhat, batch.y),
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

    root = f"runs/dev/{time.time_ns()}"  ###

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=root)
    parser.add_argument("--wandb", type=str, default="2023-07-29")
    parser.add_argument("--dataset", type=str, default="COLLAB")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_num_workers", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--train_num_workers", type=int, default=32)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_gnn_layers", type=int, default=5)
    parser.add_argument("--num_mlp_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--amp", default=False, action=argparse.BooleanOptionalAction)
    # use --amp with dataset in ["REDDIT-MULTI-5K", "REDDIT-MULTI-12K"]

    args = parser.parse_args()
    main(args)
