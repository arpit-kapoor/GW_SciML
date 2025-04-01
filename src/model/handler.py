import torch

import torch.nn as nn
import torch.optim as optim

class ModelHandler:
    def __init__(self, model, device, criterion=None, optimizer=None, lr=0.001):
        """
        Initializes the Handler class.

        Args:
            model (torch.nn.Module): The neural network model to train.
            device (torch.device): The device to run the training on (e.g., 'cpu' or 'cuda').
            criterion (torch.nn.Module, optional): Loss function. Defaults to nn.MSELoss().
            optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to Adam with lr=0.001.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion if criterion else nn.MSELoss()
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=lr)

    def train(self, train_loader, num_epochs):
        """
        Trains the model on the provided training data loader.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train the model.
        """
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    def predict(self, test_loader):
        """
        Generates predictions on the provided test data loader.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for test data.

        Returns:
            list: List of predictions for the test data.
        """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        return predictions
    

    def evaluate(self, test_loader):
        """
        Evaluates the model on the provided test data loader.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for test data.

        Returns:
            float: Average loss on the test data.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(test_loader)
    
    def save_model(self, path):
        """
        Saves the model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), path)