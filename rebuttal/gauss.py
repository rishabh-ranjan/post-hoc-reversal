import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# from sklearn.datasets import make_circles

# from sklearn.datasets import make_moons


# from torch.distributions import MultivariateNormal


# def generate_data(n_samples, noise=0.2):
#     n = np.sqrt(np.random.rand(n_samples // 2)) * 780 * (2 * np.pi) / 360
#     d1x = -np.cos(n) * n + np.random.rand(n_samples // 2) * noise
#     d1y = np.sin(n) * n + np.random.rand(n_samples // 2) * noise
#     return (
#         np.vstack((np.column_stack((d1x, d1y)), np.column_stack((-d1x, -d1y)))),
#         np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2))),
#     )


def generate_data(n_samples=1000, noise=0.9, random_state=42):
    np.random.seed(random_state)

    n = np.sqrt(np.random.rand(n_samples // 2)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_samples // 2) * noise
    d1y = np.sin(n) * n + np.random.rand(n_samples // 2) * noise
    X = np.vstack((np.column_stack((d1x, d1y)), np.column_stack((-d1x, -d1y))))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# def generate_data(n_samples=1000, noise=2.5, seed=0):
#     generator = torch.Generator().manual_seed(seed)

#     half_samples = n_samples // 4
#     X1 = 1 + noise * (torch.rand(half_samples, 2, generator=generator) - 0.5)
#     X2 = -1 + noise * (torch.rand(half_samples, 2, generator=generator) - 0.5)
#     X3 = torch.tensor([-1.0, 1.0]) + noise * (
#         torch.rand(half_samples, 2, generator=generator) - 0.5
#     )
#     X4 = torch.tensor([1.0, -1.0]) + noise * (
#         torch.rand(half_samples, 2, generator=generator) - 0.5
#     )
#     X = torch.cat((X1, X2, X3, X4))

#     y1 = torch.ones(2 * half_samples, dtype=torch.long)
#     y2 = torch.zeros(2 * half_samples, dtype=torch.long)
#     y = torch.cat((y1, y2))

#     return X, y


# def generate_data(n_samples=1000, means=None, cov=None, seed=0):
#     torch.manual_seed(seed)

#     if means is None:
#         means = [torch.tensor([1.0, 1.0]), torch.tensor([-1.0, -1.0])]

#     if cov is None:
#         # cov = torch.eye(2)  # Identity matrix as default covariance
#         cov = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

#     half_samples = n_samples // 2
#     dist1 = MultivariateNormal(means[0], cov)
#     dist2 = MultivariateNormal(means[1], cov)
#     X1 = dist1.sample((half_samples,))
#     X2 = dist2.sample((half_samples,))
#     X = torch.cat((X1, X2))

#     y1 = torch.zeros(half_samples, dtype=torch.long)
#     y2 = torch.ones(half_samples, dtype=torch.long)
#     y = torch.cat((y1, y2))

#     return X, y


# def generate_data(n_samples=1000, flip_frac=0.0, seed=42):
#     generator = torch.Generator().manual_seed(seed)

#     half_samples = n_samples // 2

#     # Uniformly sample in the first and third quadrants
#     X1 = 1 + torch.rand(
#         half_samples, 2, generator=generator
#     )  # Uniformly sample between 1 and 2
#     X2 = -1 - torch.rand(
#         half_samples, 2, generator=generator
#     )  # Uniformly sample between -1 and -2
#     X = torch.cat((X1, X2))

#     # Labels before flipping
#     y = torch.cat((torch.ones(half_samples), torch.zeros(half_samples)))

#     # Flip the labels for a fraction of randomly selected examples
#     num_flip = int(n_samples * flip_frac)
#     flip_indices = torch.randperm(n_samples, generator=generator)[:num_flip]
#     y[flip_indices] = 1 - y[flip_indices]

#     y = y.long()

#     return X, y


# def generate_data(n_samples=1000, noise=0.4, factor=0.1, random_state=42):
#     X, y = make_circles(
#         n_samples=n_samples, noise=noise, factor=factor, random_state=random_state
#     )
#     return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# def flip_labels(y, flip_frac=0.1, seed=0):
#     generator = torch.Generator().manual_seed(seed)

#     num_flip = int(len(y) * flip_frac)
#     flip_indices = torch.randperm(len(y), generator=generator)[:num_flip]
#     y_flipped = y.clone()
#     y_flipped[flip_indices] = 1 - y[flip_indices]

#     return y_flipped


# def generate_data(n_samples=1000, noise=0.2, random_state=42):
#     X, y = make_circles(
#         n_samples=n_samples, factor=0.3, noise=0.2, random_state=random_state
#     )
#     return torch.tensor(X, dtype=torch.float32), flip_labels(
#         torch.tensor(y, dtype=torch.long), flip_frac=noise
#     )


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
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
    Z = torch.softmax(Z, dim=1)[:, 1]  # Get the probability of the second class
    Z = Z.reshape(xx.shape).detach()

    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    plt.plot([-2, 2], [2, -2], ls="--", color="black")

    plt.pcolormesh(
        xx, yy, Z, cmap="RdBu_r", alpha=0.6
    )  # Use color to represent the predicted probability
    # plt.contourf(
    #     xx, yy, Z, cmap="RdBu_r", alpha=0.6
    # )  # Use contour to represent the predicted probability
    # plt.contour(xx, yy, Z, levels=[0.5], colors="black")  # Decision boundary
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", marker="o")
    plt.show()


def test_model(model, X, y):
    outputs = model(X)
    _, predicted = torch.max(outputs.data, 1)
    total = y.size(0)
    correct = (predicted == y).sum().item()
    return 100 - (correct / total) * 100


def train_model(models, X_train, y_train, X_test, y_test, epochs=1000, lr=0.01):
    criterion = nn.CrossEntropyLoss()

    # Create 4 instances of the model and the optimizer
    optimizers = [torch.optim.Adam(model.parameters(), lr=lr) for model in models]

    train_errors = []
    test_errors = []
    ensemble_errors = []

    # Create a DataLoader for the training data
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=200, shuffle=True)

    for epoch in range(epochs):
        for model, optimizer in zip(models, optimizers):
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_errors.append(
            np.mean([test_model(model, X_train, y_train) for model in models])
        )
        test_errors.append(
            np.mean([test_model(model, X_test, y_test) for model in models])
        )

        # Calculate the ensemble error
        ensemble_output = sum(model(X_test) for model in models) / len(models)
        _, ensemble_predicted = torch.max(ensemble_output.data, 1)
        total = y_test.size(0)
        correct = (ensemble_predicted == y_test).sum().item()
        ensemble_error = 100 - (correct / total) * 100
        ensemble_errors.append(ensemble_error)

        if epoch % 100 == 0:
            for model in models:
                visualize_decision_boundary(model, X_train, y_train)
            plot_errors(train_errors, test_errors, ensemble_errors)
            print("----------------")

    return train_errors, test_errors, ensemble_errors


def plot_errors(train_errors, test_errors, ensemble_errors):
    # plt.plot(train_errors, label="Train")
    plt.plot(test_errors, label="Test")
    plt.plot(ensemble_errors, label="Ensemble")
    plt.xlabel("Epoch")
    plt.ylabel("Error (%)")
    plt.legend()
    plt.show()


def main():
    X_train, y_train = generate_data(n_samples=200)
    X_test, y_test = generate_data(n_samples=10000)

    models = [MLP(input_size=2, hidden_size=100, num_classes=2) for _ in range(4)]
    for model in models:
        visualize_decision_boundary(model, X_train, y_train)
    print("Training models...")

    train_errors, test_errors, ensemble_errors = train_model(
        models, X_train, y_train, X_test, y_test
    )

    for model in models:
        visualize_decision_boundary(model, X_train, y_train)
    plot_errors(train_errors, test_errors, ensemble_errors)


if __name__ == "__main__":
    main()
