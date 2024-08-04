import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_samples, mean1=-1, mean2=1, std_dev=1, seed=3):
    np.random.seed(seed)
    samples_class1 = np.random.normal(mean1, std_dev, n_samples)
    samples_class2 = np.random.normal(mean2, std_dev, n_samples)
    samples = np.concatenate((samples_class1, samples_class2), axis=0)
    labels = np.concatenate((np.zeros(n_samples), np.ones(n_samples)))
    return torch.from_numpy(samples).float(), torch.from_numpy(labels).long()


# MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        hidden_dim = 100
        self.layers = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.layers(x)


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models

    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.models], dim=1)
        return outputs.mean(dim=1)


# Training function
def train(model, criterion, optimizer, data, labels):
    p = torch.randperm(data.size(0))
    data = data[p]
    labels = labels[p]
    model.train()
    for data_batch, labels_batch in zip(data.split(1), labels.split(1)):
        optimizer.zero_grad()
        outputs = model(data_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
    return loss.item()


# Testing function
def test(model, criterion, data, labels):
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        loss = criterion(outputs, labels)
    return loss.item()


# Function to plot errors
def plot_errors(epochs, test_errors):
    plt.figure()  # Create a new figure
    for i in range(len(test_errors)):
        plt.plot(range(epochs), test_errors[i], label=f"Model {i}")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


# Function to plot predicted probabilities and training points
def plot_probabilities_and_points(model, train_data, train_labels):
    plt.figure(figsize=(8, 2))  # Create a new figure
    model.eval()
    with torch.no_grad():
        # Generate a range of input values
        x_values = torch.linspace(train_data.min(), train_data.max(), 500).view(-1, 1)
        outputs = model(x_values)
        probabilities = nn.functional.softmax(outputs, dim=1)[:, 1]
    # Plot the predicted probabilities as a curve
    plt.plot(x_values.numpy(), probabilities.numpy(), label="Predicted Probability")
    plt.scatter(
        train_data.numpy(), train_labels.numpy(), label="Training Points", alpha=0.3
    )
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.show()


# Updated main function
def main():
    # Generate synthetic data
    train_data, train_labels = generate_data(20)
    test_data, test_labels = generate_data(1000)

    # Initialize model, criterion and optimizer
    models = [MLP() for _ in range(2)]
    criterion = nn.CrossEntropyLoss()
    optimizers = [optim.Adam(model.parameters()) for model in models]

    ensemble = Ensemble(models)

    # Train and test the model
    epochs = 100
    test_errors = [[] for _ in range(len(models) + 1)]
    for epoch in range(epochs):
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            train(model, criterion, optimizer, train_data.view(-1, 1), train_labels)
            test_error = test(model, criterion, test_data.view(-1, 1), test_labels)
            test_errors[i].append(test_error)
        test_error = test(ensemble, criterion, test_data.view(-1, 1), test_labels)
        test_errors[-1].append(test_error)

        if epoch % 10 == 0:
            # Plot predicted probabilities and training points
            for model in models:
                plot_probabilities_and_points(model, train_data, train_labels)
            print("---")

    # Plot training and test errors
    plot_errors(epochs, test_errors)

    # Plot predicted probabilities and training points
    for model in models:
        plot_probabilities_and_points(model, train_data, train_labels)


if __name__ == "__main__":
    main()
