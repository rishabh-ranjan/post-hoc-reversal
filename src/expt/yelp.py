import argparse
import json
from pathlib import Path

from datasets import load_dataset
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers.optimization import SchedulerType
from tqdm.auto import tqdm
import wandb

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

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    def tokenize(data):
        return tokenizer(data["text"], padding="max_length", truncation=True)

    dataset = (
        load_dataset(args.dataset)["train"]
        .shuffle(seed=42)
        .select(range(args.train_size + args.val_size + args.test_size))
    )
    num_classes = dataset.features["label"].num_classes

    tokenized_dataset = (
        dataset.map(tokenize, batched=True)
        .remove_columns(["text"])
        .rename_column("label", "labels")
    )
    tokenized_dataset.set_format("torch")

    datasets = {
        "train": tokenized_dataset.select(range(args.train_size)),
        "val": tokenized_dataset.select(
            range(args.train_size, args.train_size + args.val_size)
        ),
        "test": tokenized_dataset.select(
            range(
                args.train_size + args.val_size,
                args.train_size + args.val_size + args.test_size,
            )
        ),
    }

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

    net = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_classes
    ).to(device)

    opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lrs = optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1,
        end_factor=0 if args.lrs == "linear" else 1,
        total_iters=args.num_epochs * len(train_loader),
    )

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
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                ys.append(batch.pop("labels"))
                yhat = net(**batch).logits
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
        for batch in tqdm(train_loader, desc="batches", leave=False):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast("cuda", enabled=args.amp):
                outputs = net(**batch)
                yhat = outputs.logits
                y = batch["labels"]
                loss = F.cross_entropy(yhat, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            lrs.step()

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
    parser.add_argument("--wandb", type=str, default="")
    parser.add_argument("--dataset", type=str, default="yelp_review_full")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-num-workers", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--train-num-workers", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=25_000)
    parser.add_argument("--val-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument("--model", type=str, default="distilbert-base-cased")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lrs", type=str, default="linear")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    main(args)
