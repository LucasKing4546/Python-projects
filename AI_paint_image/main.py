# prompt: use the above network to identify a number uploaded by the user

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from google.colab import files
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# Function to preprocess the uploaded image
def preprocess_image(image_path):
    # Open the image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    # Resize the image to 28x28
    image = image.resize((28, 28))
    # Convert image to numpy array and normalize
    image = np.array(image).astype(np.float32) / 255.0
    # Invert the colors (if needed, assuming the MNIST model expects white digits on black background)
    image = 1.0 - image
    # Convert to tensor and add batch and channel dimensions
    tensor = torch.Tensor(image).unsqueeze(0).unsqueeze(0)
    # Normalize the tensor (mean and std from MNIST dataset)
    tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)
    return tensor

# Upload the image file
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Preprocess the uploaded image
image_tensor = preprocess_image(image_path)


# Make a prediction
with torch.no_grad():
    output = model(image_tensor)
    predicted_digit = output.argmax(dim=1, keepdim=True).item()

print(f"Predicted digit: {predicted_digit}, from {output}")