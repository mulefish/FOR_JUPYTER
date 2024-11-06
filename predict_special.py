from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
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
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()  # Set the model to evaluation mode

# Predict a single image and return predicted label
def predict_single_image(model, image):
    with torch.no_grad():
        image = image.view(-1, 28 * 28)  # Flatten the image
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def main():
    # Initialize and load model weights
    model = SimpleNN()
    load_model_weights(model, "mnist_model_weights.pth")
    print("Model weights loaded successfully.")

    # Load and preprocess the custom "6" image
    image_path = "outside_number.png"
    image = Image.open(image_path).convert("L")  # Convert to grayscale

    # Enhance contrast to better match MNIST images
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Increase contrast

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize as per MNIST's training normalization
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict the label for the custom image
    predicted_label = predict_single_image(model, image_tensor)
    print(f"Predicted Label for outside_number.png: {predicted_label}")

    # Display the image and prediction result
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted Label: {predicted_label}")
    plt.show()

if __name__ == "__main__":
    main()
