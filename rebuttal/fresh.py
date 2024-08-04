import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, TensorDataset


def pdf(x, num_rings=2):
    r = (x[:, 0] ** 2 + x[:, 1] ** 2).sqrt()
    return ((2 * math.pi * num_rings * r).cos() + 1) / 2


# def get_data(num_samples, seed=0):
#     rng = torch.Generator().manual_seed(seed)

#     x = 2 * torch.rand(num_samples, 2, generator=rng) - 1
#     y = torch.rand(num_samples, generator=rng) <= pdf(x)
#     y = y.to(torch.long)

#     return x, y


def get_data(n_samples=1000, noise=2.5, random_state=42):
    np.random.seed(random_state)

    n = np.sqrt(np.random.rand(n_samples // 2)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * (n + np.random.randn(n_samples // 2) * noise)
    d1y = np.sin(n) * (n + np.random.randn(n_samples // 2) * noise)
    X = np.vstack((np.column_stack((d1x, d1y)), np.column_stack((-d1x, -d1y))))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# def get_data(n_samples=1000, noise=0.05):
#     # Initialize the data array
#     X = np.zeros((n_samples * 4, 2))
#     y = np.zeros(n_samples * 4)

#     # Generate data for each circle
#     for i in range(4):
#         # Generate evenly spaced numbers over a specified interval
#         theta = np.linspace(0, 2 * np.pi, n_samples)

#         # Generate the radius of the circle
#         r = i + 1 + noise * np.random.normal(size=n_samples)

#         # Generate the x and y values
#         X[i * n_samples : (i + 1) * n_samples, 0] = r * np.cos(theta)
#         X[i * n_samples : (i + 1) * n_samples, 1] = r * np.sin(theta)

#         # Assign the class label
#         y[i * n_samples : (i + 1) * n_samples] = i % 2

#     # Convert to PyTorch tensors
#     X = torch.tensor(X, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.long)

#     return X, y


# def get_data(n_samples=1000, noise=0.4):
#     # Initialize the data array
#     X = np.zeros((n_samples * 4, 2))
#     y = np.zeros(n_samples * 4)

#     # Generate data for each circle
#     for i in range(4):
#         # Generate evenly spaced numbers over a specified interval
#         theta = np.linspace(0, 2 * np.pi, n_samples)

#         # Generate the radius of the circle
#         r = i + noise * np.random.normal(size=n_samples)

#         # Generate the x and y values
#         X[i * n_samples : (i + 1) * n_samples, 0] = r * np.cos(theta)
#         X[i * n_samples : (i + 1) * n_samples, 1] = r * np.sin(theta)

#         # Assign the class label
#         y[i * n_samples : (i + 1) * n_samples] = i % 2

#     # Convert to PyTorch tensors
#     X = torch.tensor(X, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.long)

#     return X, y


def viz_pdf(pdf, ax, x_min=-15, x_max=15, y_min=-15, y_max=15, steps=100):
    x = torch.linspace(x_min, x_max, steps)
    y = torch.linspace(y_min, y_max, steps)
    xg, yg = torch.meshgrid(x, y)
    inp = torch.stack([xg.reshape(-1), yg.reshape(-1)], dim=-1).to("cuda")
    with torch.no_grad():
        pg = pdf(inp).detach().cpu().view(steps, steps)

    ax.pcolormesh(xg, yg, pg, cmap="coolwarm", alpha=0.6)


def viz_data(x, y, ax):
    x = x.cpu().numpy()
    ax.scatter(x[:, 0], x[:, 1], c=y.cpu().numpy(), cmap="coolwarm")


class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out


def train(num_epochs=10_000, flip_frac=0.0, device="cuda"):
    x, y = get_data(500, noise=0.5)
    # randomly flip5some labels
    flip = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))[
        : int(flip_frac * len(y))
    ]
    y[flip] = 1 - y[flip]

    x = x.to(device)
    y = y.to(device)

    x_test, y_test = get_data(1000, noise=0.5)

    x_test = x_test.to(device)
    y_test = y_test.to(device)

    nets = [MLP(100).to(device) for _ in range(2)]
    opts = [optim.Adam(net.parameters(), lr=0.001) for net in nets]

    # Assuming X and y are your data and labels
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    test_errs = []
    ens_errs = []
    for epoch in tqdm(range(num_epochs)):
        for net, opt in zip(nets, opts):
            for batch_X, batch_y in dataloader:
                opt.zero_grad()
                out = net(batch_X)
                loss = F.binary_cross_entropy(out, batch_y.float().view(-1, 1))
                loss.backward()
                opt.step()

        with torch.no_grad():
            test_out = net(x_test)
            test_err = (test_out.round().view(-1) != y_test).float().mean()
            test_errs.append(test_err.item())

        test_outs = []
        for net in nets:
            with torch.no_grad():
                test_out = net(x_test)
            test_outs.append(test_out)
        test_out = torch.cat(test_outs, dim=1).mean(dim=1)
        ens_err = (test_out.round().view(-1) != y_test).float().mean()
        ens_errs.append(ens_err.item())

        if epoch % 1000 == 999 or epoch == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

            for net in nets:
                fig, ax = plt.subplots()
                viz_pdf(net, ax)
                viz_data(x, y, ax)
                fig.show()
                plt.show()

            fig, ax = plt.subplots()
            ax.plot(test_errs, label="Test Error")
            ax.plot(ens_errs, label="Ensemble Error")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Test Error")
            fig.show()
            plt.show()

            print("----------------")
