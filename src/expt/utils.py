from collections import defaultdict
import json
from pathlib import Path

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
import torchcal

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


def evaluate(yhat, y):
    return {
        "err": (yhat.argmax(-1) != y).float().mean(),
        "nll": F.cross_entropy(yhat, y),
    }


class KeyValStore:
    def __init__(
        self,
        root,
        device="cpu",
        default_cache=False,
    ):
        self.root = root
        self.device = device
        self.cache = {}
        self.default_cache = default_cache

    def save(self, val, key):
        path = f"{self.root}/{key}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(val, path)

    def load(self, key, cache=None):
        if cache is None:
            cache = self.default_cache

        if cache and key in self.cache:
            return self.cache[key]

        val = torch.load(f"{self.root}/{key}.pt", self.device)

        if cache:
            self.cache[key] = val

        return val


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


class Evaluator:
    def __init__(self, root, pred, net, update_bn_loader=None):
        self.root = root
        self.pred = pred
        self.update_bn_loader = update_bn_loader

        self.swa = AveragedModel(net)
        self.stage = defaultdict(list)
        self.kv = KeyValStore(root)

    def __call__(self, epochs, net):
        self.stage["epochs"].append(epochs)

        self.swa.update_parameters(net)

        if self.update_bn_loader is not None:
            optim.swa_utils.update_bn(
                self.update_bn_loader,
                self.swa,
                device=next(self.swa.parameters()).device,
            )

        yhats = {}
        ys = {}
        for split in ["val", "test"]:
            yhats[f"{split}/base"], ys[split] = self.pred(net, split)
            yhats[f"{split}/base+swa"], _ = self.pred(self.swa, split)

            if epochs == 1:
                self.kv.save(ys[split], f"y/{split}")

            for name in ["base", "base+swa"]:
                yhat = yhats[f"{split}/{name}"]
                self.kv.save(yhat, f"yhat/{epochs}/{split}/{name}")

                for metric, val in evaluate(yhat, ys[split]).items():
                    self.stage[f"{metric}/{split}/{name}"].append(val)

    def finalize(self):
        for key, val in self.stage.items():
            self.kv.save(torch.tensor(val), key)


class ELRLoss(nn.Module):
    def __init__(self, train_sz, num_classes, momentum=0.7):
        super().__init__()
        self.register_buffer("target", torch.zeros(train_sz, num_classes))
        self.momentum = momentum

    def forward(self, idx, yhat):
        prob = F.softmax(yhat, dim=-1)
        prob = torch.clamp(prob, min=torch.finfo().eps)
        prob = prob.detach()
        update = prob / prob.sum(-1, keepdim=True)
        self.target[idx] = (
            self.momentum * self.target[idx] + (1 - self.momentum) * update
        )
        elr = torch.log1p(-(self.target[idx] * prob).sum(-1)).mean()
        return elr


class SOPLoss(nn.Module):
    def __init__(self, train_sz, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.u = nn.Parameter(torch.empty(train_sz, 1))
        self.v = nn.Parameter(torch.empty(train_sz, num_classes))
        nn.init.normal_(self.u, mean=0.0, std=1e-8)
        nn.init.normal_(self.v, mean=0.0, std=1e-8)

    def forward(self, idx, yhat, y):
        if yhat.size() != y.size():
            y = F.one_hot(y, self.num_classes).float()

        usq = self.u[idx] ** 2 * y
        vsq = self.v[idx] ** 2 * (1 - y)

        usq = torch.clamp(usq, 0, 1)
        vsq = torch.clamp(vsq, 0, 1)

        prob = F.relu(F.softmax(yhat, dim=-1) + usq - vsq.detach())
        prob = torch.clamp(F.normalize(prob, p=1), min=torch.finfo().eps)
        ce = F.cross_entropy(torch.log(prob), y)

        hard_yhat = F.one_hot(torch.argmax(yhat, dim=-1), self.num_classes)
        mse = F.mse_loss(hard_yhat + usq - vsq, y)

        return ce + mse
