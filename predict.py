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
    model.load_state_dict(torch.load(path, weights_only=True))  # 'weights_only' to avoid warning
    model.eval()  # Set the model to evaluation mode

# Predict a single image and return predicted label
def predict_single_image(model, image):
    with torch.no_grad():
        image = image.view(-1, 28 * 28)  # Flatten the image
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def main():
    # Load test data
    data_path = r'C:\Users\squar\FOR_JUPYTER\mnist\data'
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    
    # Initialize and load model weights
    model = SimpleNN()
    load_model_weights(model, "mnist_model_weights.pth")
    print("Model weights loaded successfully.")

    # Loop through a range of indexes to predict
    for index in range(10):  # You can set any range you like
        image, label = test_data[index]
        predicted_label = predict_single_image(model, image)
        print(f"Index {index} - Actual Label: {label}, Predicted Label: {predicted_label}")

        # Display only the last image in the range
        if index == 9:  # Change '9' to the last index in your range
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f"Actual Label: {label}, Predicted Label: {predicted_label}")
            plt.show()

if __name__ == "__main__":
    main()
