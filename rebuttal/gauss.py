import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def generate_data(n_samples=1000, n_features=2, means=None, std=1.0, seed=0):
    rng = torch.Generator().manual_seed(seed)

    if means is None:
        means = [[1, 1], [-1, -1]]

    half_samples = n_samples // 2
    X1 = torch.normal(
        mean=torch.tensor([means[0]] * half_samples, dtype=torch.float),
        std=std,
        generator=rng,
    )
    X2 = torch.normal(
        mean=torch.tensor([means[1]] * half_samples, dtype=torch.float),
        std=std,
        generator=rng,
    )
    X = torch.cat((X1, X2))

    y1 = torch.zeros(half_samples, dtype=torch.long)
    y2 = torch.ones(half_samples, dtype=torch.long)
    y = torch.cat((y1, y2))

    return X, y


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def visualize_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
    Z = torch.softmax(Z, dim=1)[:, 1]  # Get the probability of the second class
    Z = Z.reshape(xx.shape).detach()

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.plot([-2, 2], [2, -2], ls="--", color="black")

    plt.pcolormesh(
        xx, yy, Z, cmap="RdBu_r", alpha=0.6
    )  # Use color to represent the predicted probability
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", marker="o")
    plt.show()


def test_model(model, X, y):
    outputs = model(X)
    _, predicted = torch.max(outputs.data, 1)
    total = y.size(0)
    correct = (predicted == y).sum().item()
    return 100 - (correct / total) * 100


def train_model(
    model, X_train, y_train, X_test, y_test, epochs=1000, lr=0.01, warmup_epochs=10
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    swa_model = AveragedModel(model)

    train_errors = []
    test_errors = []
    swa_errors = []

    # Define a lambda function for the warmup phase
    def lr_lambda(epoch):
        return min(1.0, epoch / warmup_epochs)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda)

    # Define the cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        swa_model.update_parameters(model)

        train_errors.append(test_model(model, X_train, y_train))
        test_errors.append(test_model(model, X_test, y_test))
        swa_errors.append(test_model(swa_model, X_test, y_test))

        # Update the learning rate
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

    return train_errors, test_errors, swa_errors


def plot_errors(train_errors, test_errors, swa_errors):
    # plt.plot(train_errors, label="Train")
    plt.plot(test_errors, label="Test")
    plt.plot(swa_errors, label="SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Error (%)")
    plt.legend()
    plt.show()


def main():
    X_train, y_train = generate_data(n_samples=200)
    X_test, y_test = generate_data(n_samples=10000)

    model = MLP(input_size=2, hidden_size=100, num_classes=2)
    train_errors, test_errors, swa_errors = train_model(
        model, X_train, y_train, X_test, y_test
    )

    visualize_decision_boundary(model, X_train, y_train)
    plot_errors(train_errors, test_errors, swa_errors)


if __name__ == "__main__":
    main()
