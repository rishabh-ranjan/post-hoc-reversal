import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb

wandb.require("core")


def viz(net, x, y, steps=100):
    plt.figure()
    x_min = x[:, 0].min()
    x_max = x[:, 0].max()
    y_min = x[:, 1].min()
    y_max = x[:, 1].max()
    xx = torch.linspace(x_min, x_max, steps)
    yy = torch.linspace(y_min, y_max, steps)
    xg, yg = torch.meshgrid(xx, yy)
    inp = torch.stack([xg.reshape(-1), yg.reshape(-1)], dim=-1).to("cuda")
    with torch.no_grad():
        pg = F.sigmoid(net(inp).detach().cpu().view(steps, steps))

    plt.pcolormesh(xg, yg, pg, cmap="coolwarm", alpha=0.6)

    x = x.cpu().numpy()
    plt.scatter(x[:, 0], x[:, 1], c=y.cpu().numpy(), cmap="coolwarm")
    plt.show()


def spiral(n_samples=2000, noise=0.5, random_state=42):
    np.random.seed(random_state)

    n = np.sqrt(np.random.rand(n_samples // 2)) * 720 * (2 * np.pi) / 360
    d1x = -np.cos(n) * (n + np.random.randn(n_samples // 2) * noise)
    d1y = np.sin(n) * (n + np.random.randn(n_samples // 2) * noise)
    # d1x = -np.cos(n) * n + np.random.randn(n_samples // 2) * noise
    # d1y = np.sin(n) * (n + np.random.randn(n_samples // 2) * noise
    X = np.vstack((np.column_stack((d1x, d1y)), np.column_stack((-d1x, -d1y))))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        fcs = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        self.fcs = nn.ModuleList(fcs)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        for fc in self.fcs:
            x = fc(x)
            x = F.relu(x)
        x = self.fc2(x)
        return x


def get_data():
    x, y = spiral(1000)
    flip_frac = 0.2
    flip_idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(42))[
        : int(flip_frac * len(y))
    ]
    y[flip_idx] = 1 - y[flip_idx]

    xt, yt = spiral(1000)
    return x, y, xt, yt


def main():
    device = "cuda"

    x, y, xt, yt = get_data()

    x = x.to(device)
    y = y.to(device)
    xt = xt.to(device)
    yt = yt.to(device)

    num_nets = 16
    nets = [MLP(512, 4).to(device) for _ in range(num_nets)]
    opts = [optim.Adam(net.parameters()) for net in nets]
    lrss = [
        optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=1000)
        for opt in opts
    ]

    sub_frac = 0.5
    xs = []
    ys = []
    for i in range(num_nets):
        sub_idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(i))[
            : int(sub_frac * len(y))
        ]
        xs.append(x[sub_idx])
        ys.append(y[sub_idx])

    wandb.init()

    # viz(nets[0], x, y)
    # wandb.log({"viz": wandb.Image(plt)})

    for epoch in tqdm(range(1000)):
        for opt in opts:
            opt.zero_grad()

        losses = []
        for my_x, my_y, net in zip(xs, ys, nets):
            logit = net(my_x)
            loss = F.binary_cross_entropy_with_logits(logit[:, 0], my_y.float())
            losses.append(loss)

        for loss, opt in zip(losses, opts):
            loss.backward()
            opt.step()

        # for lrs in lrss:
        #     lrs.step()

        if epoch % 10 == 9:
            with torch.no_grad():
                errs = []
                logits = []
                for net in nets:
                    logit = net(xt)
                    logits.append(logit)

                    pred = logit[:, 0] > 0
                    err = (pred != yt).float().mean()
                    errs.append(err)

                wandb.log({f"err_mean": sum(errs) / len(errs)}, step=epoch)

                logit = torch.stack(logits).mean(dim=0)
                pred = logit[:, 0] > 0
                err = (pred != yt).float().mean()
                wandb.log({"err_ens": err}, step=epoch)

    viz(net, x, y)
    wandb.log({"viz": wandb.Image(plt)})


if __name__ == "__main__":
    main()
