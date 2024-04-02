import argparse
import json
from pathlib import Path

import folktables
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch_geometric as pyg
from tqdm.auto import tqdm
import wandb

from . import utils

FOLKTABLES = {
    "income": folktables.ACSIncome,
    "public_coverage": folktables.ACSPublicCoverage,
    "mobility": folktables.ACSMobility,
    "employment": folktables.ACSEmployment,
    "travel_time": folktables.ACSTravelTime,
}


def from_folktables(name, cat_enc=True):
    Path("data/folktables").mkdir(exist_ok=True, parents=True)

    data_source = folktables.ACSDataSource(
        survey_year="2018",
        horizon="1-Year",
        survey="person",
        root_dir="data/folktables",
    )
    acs_data = data_source.get_data(states=["CA"], download=True)
    defs = data_source.get_definitions(download=True)
    cats = folktables.generate_categories(
        features=FOLKTABLES[name].features, definition_df=defs
    )

    if cat_enc:
        x, y, _ = FOLKTABLES[name].df_to_pandas(acs_data, categories=cats, dummies=True)
    else:
        x, y, _ = FOLKTABLES[name].df_to_pandas(acs_data)

    x = x.astype("float")
    x = torch.as_tensor(x.to_numpy(), dtype=torch.float)
    y = torch.as_tensor(y.to_numpy()[:, 0], dtype=torch.long)

    # normalize
    x = (x - x.mean(0)) / x.std(0)

    # remove nan columns
    x = x[:, ~x.isnan().any(0)]

    # shuffle
    idx = torch.randperm(x.size(0), generator=torch.Generator().manual_seed(42))
    x = x[idx]
    y = y[idx]

    return x, y


def main(args):
    print(f"{args=}")  ###

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    kv = utils.KeyValStore(args.root, device)  ###
    kv.save(args, "args")

    dataset_path = f"data/folktables/{args.dataset}.pt"
    if not Path(dataset_path).exists():
        x, y = from_folktables(args.dataset)
        torch.save((x, y), dataset_path)

    x, y = torch.load(dataset_path, map_location=device)
    num_features = x.size(-1)
    num_classes = y.max().item() + 1

    test_sz = int(0.1 * x.size(0))
    split_sz = [x.size(0) - 2 * test_sz, test_sz, test_sz]
    X = {}
    X["train"], X["val"], X["test"] = x.split(split_sz)
    Y = {}
    Y["train"], Y["val"], Y["test"] = y.split(split_sz)

    print(f"{num_classes=}")
    print(f"{num_features=}")
    print(f"{len(X['train'])=}")
    print(f"{len(X['val'])=}")
    print(f"{len(X['test'])=}")
    print(f"max class freq: {y.bincount().max()/y.size(-1)}")  ###
    print(f"min class freq: {y.bincount().min()/y.size(-1)}")

    eval_loaders = {
        split: utils.FastDataLoader([X[split]], args.eval_batch_size, shuffle=False)
        for split in ["val", "test"]
    }

    train_loader = utils.FastDataLoader(
        [torch.arange(X["train"].size(0), device=device), X["train"], Y["train"]],
        args.train_batch_size,
        shuffle=True,
    )

    net = pyg.nn.MLP(
        in_channels=num_features,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        num_layers=args.num_layers,
    )
    net = net.to(device)

    if args.optimizer == "adamw":
        opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        opt = optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    if args.scheduler == "exponential":
        lrs = optim.lr_scheduler.ExponentialLR(opt, args.lr_decay)
    elif args.scheduler == "cosine":
        lrs = optim.lr_scheduler.CosineAnnealingLR(opt, args.num_epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.loss == "sop":
        sop = utils.SOPLoss(X["train"].size(0), num_classes).to(device)
        sop_opt = optim.SGD(
            [
                dict(params=[sop.u], lr=args.sop_lr_u),
                dict(params=[sop.v], lr=args.sop_lr_v),
            ]
        )

    elif args.loss == "elr":
        elr = utils.ELRLoss(X["train"].size(0), num_classes, args.elr_momentum).to(
            device
        )

    if args.wandb:
        run = wandb.init(
            project=args.wandb,
            name=f"{args.dataset}__{args.loss}__{args.root[-3:]}",  ###
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

    evaluator = utils.Evaluator(
        args.root, pred, net, update_bn_loader=eval_loaders["val"]  ###
    )

    step = 0
    for epoch in tqdm(range(args.num_epochs), leave=False):
        net.train()
        for i, x, y in train_loader:
            with torch.autocast("cuda", enabled=args.amp):
                yhat = net(x)

                if args.loss == "ce":
                    loss = F.cross_entropy(yhat, y)

                elif args.loss == "sop":
                    loss = sop(i, yhat, y)

                elif args.loss == "elr":
                    nll = F.cross_entropy(yhat, y)
                    reg = elr(i, yhat)
                    loss = nll + args.elr_weight * reg

            if args.loss == "sop":
                sop_opt.zero_grad(set_to_none=True)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)

            if args.loss == "sop":
                scaler.step(sop_opt)

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

                if args.loss == "sop":
                    wandb.log(
                        {
                            "sop/usq": torch.clamp(
                                (sop.u**2).max(-1).values, 0, 1
                            ).mean(),
                            "sop/vsq": torch.clamp(
                                (sop.v**2).max(-1).values, 0, 1
                            ).mean(),
                        },
                        step=step,
                    )

        lrs.step()

        evaluator(epoch + 1, net)
        if args.wandb:
            wandb.log({k: v[-1] for k, v in evaluator.stage.items()}, step)  ###

    evaluator.finalize()  ###

    if args.wandb:
        wandb.finish()

    Path(f"{args.root}/done").touch()  ###


if __name__ == "__main__":
    import time

    root = f"runs/dev/{time.time_ns()}"  ###
    Path(root).mkdir(parents=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=root)  ###
    parser.add_argument("--wandb", type=str, default="2023-08-23")
    parser.add_argument("--dataset", type=str, default="income")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--hidden_channels", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="exponential")
    parser.add_argument("--lr_decay", type=float, default=1.00)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    # run it for 100 epochs, if we want to plot less, we can decide later
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--sop_lr_u", type=float, default=1.0)
    parser.add_argument("--sop_lr_v", type=float, default=10.0)
    parser.add_argument("--elr_momentum", type=float, default=0.9)  # unknown best vals
    parser.add_argument("--elr_weight", type=float, default=1.0)  # unknown best vals

    args = parser.parse_args()
    main(args)
