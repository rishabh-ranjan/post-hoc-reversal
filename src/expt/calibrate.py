import argparse
from collections import defaultdict

import torch
import torchcal
from tqdm.auto import tqdm
import wandb

from . import utils


def main(args):
    print(f"{args=}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kv = utils.KeyValStore(args.root, device)

    num_classes = kv.load("y/val", cache=True).max().item() + 1
    cal = torchcal.calibrator(args.cal, num_classes, device=device)

    stage = defaultdict(list)

    if args.wandb:
        wandb.init(
            project=args.wandb,
            name=f"{args.base}+{args.suf}__{args.root[-3:]}",
            config=args,
        )

    epochs = kv.load("epochs")
    for epoch_i, epoch in enumerate(tqdm(epochs, "epochs", leave=False)):
        for split in ["val", "test"]:
            key = f"yhat/{epoch}/{split}/{args.base}"
            yhat = kv.load(key)
            y = kv.load(f"y/{split}", cache=True)

            if split == "val":
                cal.fit(yhat, y)
            yhat_cal = cal(yhat)

            kv.save(yhat_cal, f"{key}+{args.suf}")

            for metric, val in utils.evaluate(yhat_cal, y).items():
                key = f"{metric}/{split}/{args.base}+{args.suf}"
                stage[key].append(val)

                if args.wandb:
                    base_key = f"{metric}/{split}/{args.base}"
                    base_val = kv.load(base_key, cache=True)[epoch_i]
                    wandb.log(
                        {
                            "epochs": epoch,
                            base_key: base_val,
                            key: val,
                        },
                        step=epoch_i,
                    )

    for key, val in stage.items():
        kv.save(torch.tensor(val), key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="runs/dev/1690523406022746920")
    parser.add_argument("--wandb", type=str, default="2023-07-28")
    parser.add_argument("--base", type=str, default="base")
    parser.add_argument("--cal", type=str, default="temp_scaler")
    parser.add_argument("--suf", type=str, default="ts")

    args = parser.parse_args()
    main(args)
