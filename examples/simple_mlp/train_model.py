"""
Simple MLP Example - Train a simple neural network and save it
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron"""
    def __init__(self, input_size=20, hidden_size=64, num_classes=2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def train():
    print("Creating synthetic dataset...")
    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    print("Training model...")
    # Create model
    model = SimpleMLP(input_size=20, hidden_size=64, num_classes=2)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    batch_size = 32

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        # Print progress
        if (epoch + 1) % 10 == 0:
            accuracy = 100. * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = outputs.max(1)
        test_accuracy = 100. * predicted.eq(y_test).sum().item() / len(y_test)
        print(f'\nTest Accuracy: {test_accuracy:.2f}%')

    # Save the complete model
    print("\nSaving model...")
    torch.save(model, 'simple_mlp.pt')
    print("✓ Model saved to simple_mlp.pt")

    # Also save just the state dict (alternative format)
    torch.save(model.state_dict(), 'simple_mlp_state.pt')

    print("\nModel architecture:")
    print(model)

    print("\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    return model


if __name__ == '__main__':
    model = train()
