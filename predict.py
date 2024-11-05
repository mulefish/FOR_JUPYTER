# predict_mnist.py
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn

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

def load_model_weights(model, path="mnist_model_weights.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode

# Predict a single image from the test set
def predict_single_image(model, test_data, index=0):
    """Predicts a single image from the test set and shows the result."""
    image, label = test_data[index]
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Actual Label: {label}")
    plt.show()
    
    with torch.no_grad():
        image = image.view(-1, 28 * 28)  # Flatten the image
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    print(f"Predicted Label: {predicted.item()}")

def main():
    # Load test data
    data_path = r'C:\Users\squar\FOR_JUPYTER\mnist\data'
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    
    # Initialize and load model weights
    model = SimpleNN()
    load_model_weights(model, "mnist_model_weights.pth")
    print("Model weights loaded successfully.")

    # Predict and disply image: But! Haha! I thought '6' would point to 6 but somehow it is 4
    predict_single_image(model, test_data, index=6) 

if __name__ == "__main__":
    main()

