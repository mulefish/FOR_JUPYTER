# train_mnist_classifier.py
from datetime import datetime
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import os

# Logging function
def log(msg):
    right_now = datetime.now()
    print(f"{msg} {right_now.strftime('%Y-%m-%d %H:%M:%S')}")

# Simple Neural Network Definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (28x28 pixels)
        self.fc2 = nn.Linear(128, 64)       # Hidden layer
        self.fc3 = nn.Linear(64, 10)        # Output layer (10 classes for digits 0-9)

    def forward(self, x):
        x = x.view(-1, 28 * 28)            # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)                 # No activation here; loss function will handle it

# Data Preparation
def load_data(data_path, batch_size=64):
    """Loads MNIST data and returns data loaders."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Model Training
def train_model(model, train_loader, epochs=5, learning_rate=0.01):
    """Trains the model and prints the loss after each epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        log(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Save Model Weights
def save_model_weights(model, path="mnist_model_weights.pth"):
    torch.save(model.state_dict(), path)
    log(f"Model weights saved to {path}")

# Main function
def main():
    log("Starting MNIST Classifier Training")

    # Load data
    data_path = r'C:\Users\squar\FOR_JUPYTER\mnist\data'
    train_loader, _ = load_data(data_path)
    log("Data Loaded")

    # Initialize and train model
    model = SimpleNN()
    log("Model Initialized")
    train_model(model, train_loader)
    log("Model Training Complete")

    # Save model weights
    save_model_weights(model, "mnist_model_weights.pth")

if __name__ == "__main__":
    main()

