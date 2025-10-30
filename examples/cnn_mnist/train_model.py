"""
CNN for MNIST - Train a convolutional neural network for digit classification
This example shows the complete workflow including data preprocessing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json


class MNISTNet(nn.Module):
    """Convolutional Neural Network for MNIST digit classification"""
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 2 pooling layers: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(-1, 32 * 7 * 7)

        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def save_preprocessing_params(mean, std, filename='preprocessing_params.json'):
    """Save preprocessing parameters for deployment"""
    params = {
        'mean': mean.tolist() if isinstance(mean, np.ndarray) else mean,
        'std': std.tolist() if isinstance(std, np.ndarray) else std,
        'input_size': [28, 28],
        'num_channels': 1,
        'normalization': 'z-score'
    }
    with open(filename, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Saved preprocessing parameters to {filename}")


def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define preprocessing transforms
    # These will need to be replicated in C++/Rust
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Save preprocessing parameters for deployment
    save_preprocessing_params(mean=0.1307, std=0.3081)

    # Create model
    model = MNISTNet().to(device)
    print("\nModel architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')

        # Epoch summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

    # Save model
    print("\nSaving model...")
    model_cpu = model.cpu()
    torch.save(model_cpu, 'mnist_cnn.pt')
    print("✓ Model saved to mnist_cnn.pt")

    # Save a few test samples for validation
    test_samples = []
    test_labels = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            test_samples.append(data[:10].cpu().numpy())
            test_labels.append(target[:10].cpu().numpy())
            break

    np.save('test_samples.npy', test_samples[0])
    np.save('test_labels.npy', test_labels[0])
    print("✓ Saved test samples for validation")

    return model_cpu


if __name__ == '__main__':
    model = train()
