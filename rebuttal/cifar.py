import argparse
import io
from pathlib import Path
import requests

import timm
import torch
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as T
from tqdm.auto import tqdm
import wandb


class FastDataLoader:
    def __init__(
        self,
        tensors,
        batch_size,
        shuffle=False,
    ):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.len = self.tensors[0].size(0)
        self.num_batches, self.rem = divmod(self.len, batch_size)
        if self.rem > 0:
            self.num_batches += 1

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.len)
            self.tensors = [t[r] for t in self.tensors]
        self.cur_idx = 0
        return self

    def __next__(self):
        if self.cur_idx == self.len:
            raise StopIteration
        if self.cur_idx == 0 and self.rem > 0:
            end_idx = self.rem
        else:
            end_idx = self.cur_idx + self.batch_size
        batch = [t[self.cur_idx : end_idx] for t in self.tensors]
        self.cur_idx = end_idx
        return batch

    def __len__(self):
        return self.num_batches


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


def x_from_cifar(dataset):
    cls = CIFAR10 if dataset == "cifar10" else CIFAR100
    ds = cls(root="data/cifar", train=True, download=True, transform=T.ToTensor())
    x = torch.stack([x for x, _ in ds])
    return x


def all_y_from_cifar_n(dataset):
    name = "CIFAR-10" if dataset == "cifar10" else "CIFAR-100"
    url = f"https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/{name}_human.pt"
    print(f"downloading {name}-N labels from {url}")
    r = requests.get(url)
    pt = torch.load(io.BytesIO(r.content), map_location="cpu")

    if dataset == "cifar10":
        map_ = {
            "clean": "clean_label",
            "aggregate": "aggre_label",
            "random1": "random_label1",
            "random2": "random_label2",
            "random3": "random_label3",
            "worst": "worse_label",
        }
    elif dataset == "cifar100_coarse":
        map_ = {
            "clean": "clean_coarse_label",
            "noisy": "noisy_coarse_label",
        }
    elif dataset == "cifar100_fine":
        map_ = {
            "clean": "clean_label",
            "noisy": "noisy_label",
        }

    all_y = {}
    for k, v in map_.items():
        all_y[k] = torch.tensor(pt[v], dtype=torch.long)

    return all_y


def main(args):
    print(f"{args=}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    x_path = f"data/cifar_n/{args.dataset}_x.pt"
    if not Path(x_path).exists():
        x = x_from_cifar(args.dataset)
        Path(x_path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(x, x_path)
    x = torch.load(x_path, map_location=device)

    all_y_path = f"data/cifar_n/{args.dataset}_all_y.pt"
    if not Path(all_y_path).exists():
        all_y = all_y_from_cifar_n(args.dataset)
        Path(all_y_path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(all_y, all_y_path)
    all_y = torch.load(all_y_path, map_location=device)

    y = all_y[args.noise]

    X = {}
    X["train"], X["val"], X["test"] = x.split([40_000, 5_000, 5_000])
    Y = {}
    Y["train"], Y["val"], Y["test"] = y.split([40_000, 5_000, 5_000])

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100_coarse":
        num_classes = 20
    elif args.dataset == "cifar100_fine":
        num_classes = 100

    print(f"{num_classes=}")
    print(f"{len(X['train'])=}")
    print(f"{len(X['val'])=}")
    print(f"{len(X['test'])=}")
    print(f"max class freq: {y.bincount().max() / y.size(-1)}")
    print(f"min class freq: {y.bincount().min() / y.size(-1)}")

    mean = IMAGENET_MEAN if args.pretrained else CIFAR_MEAN
    std = IMAGENET_STD if args.pretrained else CIFAR_STD

    normalize = T.Normalize(mean, std)
    for split in ["train", "val", "test"]:
        X[split] = normalize(X[split])

    eval_loaders = {
        split: FastDataLoader([X[split]], args.eval_batch_size, shuffle=False)
        for split in ["val", "test"]
    }

    train_loader = FastDataLoader(
        [torch.arange(X["train"].size(0), device=device), X["train"], Y["train"]],
        args.train_batch_size,
        shuffle=True,
    )
    transform = T.Compose([T.RandomCrop(32, 4), T.RandomHorizontalFlip()])

    net = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=num_classes,
        width_factor=args.width,
    )
    net = net.to(device, memory_format=torch.channels_last)

    if args.optimizer == "sgd":
        opt = optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    if args.scheduler == "cosine":
        lrs = optim.lr_scheduler.CosineAnnealingLR(opt, args.num_epochs)

    scaler = torch.amp.GradScaler(enabled=args.amp)

    if args.wandb:
        run = wandb.init(
            project=args.wandb,
            name=f"{args.dataset}__{args.noise}__{args.root[-4:]}",
            config=args,
        )
        print(f"{run.name=}")

    def pred(net, split):
        net.eval()
        with torch.no_grad():
            yhats = []
            for (x,) in eval_loaders[split]:
                yhat = net(x)
                yhats.append(yhat)
            yhat = torch.cat(yhats)
            return yhat, Y[split]

    step = 0
    for epoch in tqdm(range(args.num_epochs), leave=False):
        net.train()
        for i, x, y in train_loader:
            x = transform(x)

            with torch.autocast("cuda", enabled=args.amp):
                yhat = net(x)

                if args.loss == "ce":
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
                        "loss": loss,
                        "lr": opt.param_groups[0]["lr"],
                        "err/train_batch": (yhat.argmax(-1) != y).float().mean(),
                        "nll/train_batch": F.cross_entropy(yhat, y),
                    },
                    step,
                )

        lrs.step()

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    import time

    root = f"runs/dev/{time.time_ns()}"

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=root)
    parser.add_argument("--wandb", type=str, default="2024-08-02")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--noise", type=str, default="worst")
    parser.add_argument("--eval-batch-size", type=int, default=1_000)
    parser.add_argument("--train-batch-size", type=int, default=500)
    parser.add_argument("--model", type=str, default="resnet18d")
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss", type=str, default="ce")

    args = parser.parse_args()
    main(args)
