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
        hidden_dim = 64
        self.layers = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.layers(x)


# Training function
def train(model, criterion, optimizer, data, labels):
    model.train()
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
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
def plot_errors(epochs, train_errors, test_errors):
    plt.figure()  # Create a new figure
    plt.plot(range(epochs), train_errors, label="Train")
    plt.plot(range(epochs), test_errors, label="Test")
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
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train and test the model
    epochs = 1000
    train_errors = []
    test_errors = []
    for epoch in range(epochs):
        train_error = train(
            model, criterion, optimizer, train_data.view(-1, 1), train_labels
        )
        test_error = test(model, criterion, test_data.view(-1, 1), test_labels)
        train_errors.append(train_error)
        test_errors.append(test_error)

    # Plot training and test errors
    plot_errors(epochs, train_errors, test_errors)

    # Plot predicted probabilities and training points
    plot_probabilities_and_points(model, train_data, train_labels)


if __name__ == "__main__":
    main()
