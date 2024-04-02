import argparse
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm
import wandb

from . import utils


def main(args):
    print(f"{args=}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_kvs = []
    for run in args.runs:
        run_kvs.append(utils.KeyValStore(run, device))

    ens_kv = utils.KeyValStore(args.root, device)

    ens_kv.save(args.runs, "runs")
    ens_kv.save(run_kvs[0].load("args"), "args")

    for metric in ["err", "nll"]:
        key = f"{metric}/{args.split}/{args.base}"
        val = torch.stack([kv.load(key) for kv in run_kvs])
        ens_kv.save(val, key)

    stage = defaultdict(list)

    if args.wandb:
        wandb.init(
            project=args.wandb,
            name=f"{args.base}+ens__{args.root[-3:]}",
            config=args,
        )

    epochs = run_kvs[0].load("epochs")
    ens_kv.save(epochs, "epochs")
    for epoch_i, epoch in enumerate(tqdm(epochs, "epochs", leave=False)):
        key = f"yhat/{epoch}/{args.split}/{args.base}"
        yhat = torch.stack([kv.load(key) for kv in run_kvs])
        ens_kv.save(yhat, key)
        yhat_ens = yhat.mean(0)

        ens_kv.save(yhat_ens, f"{key}+ens")

        y = run_kvs[0].load(f"y/{args.split}", cache=True)
        if epoch_i == 0:
            ens_kv.save(y, f"y/{args.split}")

        for metric, val in utils.evaluate(yhat_ens, y).items():
            key = f"{metric}/{args.split}/{args.base}+ens"
            stage[key].append(val)

            if args.wandb:
                base_key = f"{metric}/{args.split}/{args.base}"
                base_val = ens_kv.load(base_key, cache=True).mean(0)[epoch_i]
                wandb.log(
                    {
                        "epochs": epoch,
                        base_key: base_val,
                        key: val,
                    },
                    step=epoch_i,
                )

    for key, vals in stage.items():
        ens_kv.save(torch.tensor(vals), key)


if __name__ == "__main__":
    import time

    root = f"enss/dev/{time.time_ns()}"

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=root)
    parser.add_argument("--wandb", type=str, default="2023-07-28")
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=[
            "runs/dev/1690523406022746920",
            "runs/dev/1690530959837558054",
        ],
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--base", type=str, default="base")

    args = parser.parse_args()
    main(args)
