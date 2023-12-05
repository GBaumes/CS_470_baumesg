# Assignment 05 - Train and evaluate PyTorch neural network models to predict classes from the CIFAR10 dataset
# Written by: Gregory Baumes
# 11/29/2023

import torch
import torch.optim as optim
from torchvision.transforms import v2
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, approach_name, class_cnt):
        super(CustomCNN, self).__init__()

        if approach_name == "CNN0":
            # CNN0
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Calculate the input size for the fully connected layers dynamically
            # based on the output size of the last convolutional layer
            self.fc_input_size = 128 * 8 * 8 

        elif approach_name == "CNN1":
            # CNN1
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Calculate the input size for the fully connected layers dynamically
            # based on the output size of the last convolutional layer
            self.fc_input_size = 64 * 8 * 8

        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, class_cnt)

    # Forward pass through the network
    def forward(self, x):
        
        # First convolutional layer, activation, and pooling
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second convolutional layer, activation, and pooling
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten the output for the fully connected layers
        x = self.flatten(x)
        
        # First fully connected layer and activation
        x = self.fc1(x)
        x = self.relu3(x)
        
        # Second fully connected layer (output layer)
        x = self.fc2(x)

        return x

def get_approach_names():
    # Return list of names for different approaches
    return ['CNN0', 'CNN1']

def get_approach_description(approach_name):
    # Returns the description depending on the approach name given
    match approach_name:
        case "CNN0":
            description = "An approach that focuses on using small filters in its convolutional layers."
        case "CNN1":
            description = "An approach with a deeper architecture with more layers."
        
    return description

def get_data_transform(approach_name, training):
    # Common transformations for both training and non-training data
    common_transforms = [v2.ToTensor(), v2.ConvertImageDtype(dtype=torch.float32)]

    # Data augmentation for training data
    if training:
        if approach_name == "CNN0":
            # Add CNN0-specific data augmentation
            common_transforms += [v2.RandomHorizontalFlip()]
        elif approach_name == "CNN1":
            # Add CNN1-specific data augmentation
            common_transforms += [v2.RandomRotation(10)]

    # Combine all transformations
    data_transform = v2.Compose(common_transforms)

    return data_transform
    
def get_batch_size(approach_name):
    # Set a default batch size
    batch_size = 32

    # Adjust batch size based on approach_name
    if approach_name == "CNN0":
        batch_size = 64
    elif approach_name == "CNN1":
        batch_size = 128

    return batch_size

def create_model(approach_name, class_cnt):
    # Generate model and return
    model = CustomCNN(approach_name, class_cnt)
    return model

def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Choose an optimizer (you are free to choose any optimizer)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move the model to the specified device
    model = model.to(device)

    # Set the model in training mode
    model.train()

    # number of training epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Iterate over the training dataset
        for inputs, labels in train_dataloader:
            # Move data to the specified device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        # Print out loss for each epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}")

    return model




