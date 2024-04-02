import argparse

import torch
import torchcal

from . import utils


def main(args):
    print(f"{args=}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kv = utils.KeyValStore(args.root, device)

    # post-select

    for obj in ["err", "nll"]:
        for base in args.bases:
            idx = kv.load(f"{obj}/val/{base}").argmin(-1)
            epoch = kv.load("epochs")[idx]
            for split in ["val", "test"]:
                kv.save(epoch, f"sel=post/obj={obj}/epochs/{split}/{base}")

            for metric in ["err", "nll"]:
                for split in ["val", "test"]:
                    raw = kv.load(f"{metric}/{split}/{base}")
                    if idx.ndim == 0:
                        val = raw[idx]
                    elif idx.ndim == 1:
                        val = raw[torch.arange(idx.size(-1)), idx]
                    key = f"sel=post/obj={obj}/{metric}/{split}/{base}"
                    kv.save(val, key)

                    # base metrics
                    raw = kv.load(f"{metric}/{split}/base")
                    if idx.ndim == 0:
                        val = raw[:, idx]
                    elif idx.ndim == 1:
                        val = raw[torch.arange(idx.size(-1)), idx]
                    key = f"sel=post/obj={obj}/base_{metric}/{split}/{base}"
                    kv.save(val, key)

    # pre-select

    for obj in ["err", "nll"]:
        idx = kv.load(f"{obj}/val/base").argmin(-1)
        epoch = kv.load("epochs")[idx]
        for split in ["val", "test"]:
            kv.save(epoch, f"sel=pre/obj={obj}/epochs/{split}/{base}")

        # base metrics
        for metric in ["err", "nll"]:
            for split in ["val", "test"]:
                raw = kv.load(f"{metric}/{split}/base")
                val = raw[torch.arange(idx.size(-1)), idx]

                for base in args.bases:
                    key = f"sel=pre/obj={obj}/base_{metric}/{split}/{base}"
                    kv.save(val, key)

        for base in args.bases:
            if "ens" in base:
                continue

            for metric in ["err", "nll"]:
                for split in ["val", "test"]:
                    raw = kv.load(f"{metric}/{split}/{base}")
                    val = raw[torch.arange(idx.size(-1)), idx]
                    key = f"sel=pre/obj={obj}/{metric}/{split}/{base}"
                    kv.save(val, key)

        for base in args.bases:
            if not base.endswith("+ens"):
                continue

            pref = base[:-4]
            for split in ["val", "test"]:
                yhats = []
                for i in range(idx.size(-1)):
                    yhat = kv.load(f"yhat/{epoch[i]}/{split}/{pref}")[i]
                    yhats.append(yhat)
                yhat = torch.stack(yhats).mean(0)
                kv.save(yhat, f"sel=pre/obj={obj}/yhat/{split}/{pref}+ens")

                metrics = utils.evaluate(yhat, kv.load(f"y/{split}"))
                for metric, val in metrics.items():
                    key = f"sel=pre/obj={obj}/{metric}/{split}/{pref}+ens"
                    kv.save(val, key)

        for base in args.bases:
            if not base.endswith("+ens+ts"):
                continue

            pref = base[:-3]
            cal = torchcal.calibrator("temp_scaler", device=device)
            for split in ["val", "test"]:
                yhat = kv.load(f"sel=pre/obj={obj}/yhat/{split}/{pref}")
                y = kv.load(f"y/{split}")
                if split == "val":
                    cal.fit(yhat, y)
                yhat_cal = cal(yhat)

                metrics = utils.evaluate(yhat_cal, y)
                for metric, val in metrics.items():
                    key = f"sel=pre/obj={obj}/{metric}/{split}/{pref}+ts"
                    kv.save(val, key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="enss/dev/1690609886838165392")
    parser.add_argument(
        "--bases",
        type=str,
        nargs="*",
        default=[
            "base",
            "base+swa",
            "base+ts",
            "base+swa+ts",
            "base+ens",
            "base+swa+ens",
            "base+ts+ens",
            "base+swa+ts+ens",
            "base+ens+ts",
            "base+swa+ens+ts",
            "base+ts+ens+ts",
            "base+swa+ts+ens+ts",
        ],
    )

    args = parser.parse_args()
    main(args)
