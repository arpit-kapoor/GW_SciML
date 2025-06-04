import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

class ModelHandler:
    def __init__(self, model, device, criterion=None, optimizer=None, scheduler=None, lr=0.001, scheduler_interval=5):
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
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval

    def train(self, train_loader, num_epochs=100, val_loader=None):
        """
        Trains the model on the provided training data loader.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train the model.
        """
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                if len(loss.shape) >= 1:
                    loss = loss.sum()

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            if (epoch + 1) % self.scheduler_interval and self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(running_loss)
                else:
                    self.scheduler.step()
            
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.4f}")
            else:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.4f}")

        if val_loader is not None:
            train_loss = running_loss / len(train_loader)
            return train_loss, val_loss
        else:
            train_loss = running_loss / len(train_loader)
            return train_loss


    def predict(self, test_loader, output_transform=None):
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
            for inputs, targets in tqdm(test_loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                # _, predicted = torch.max(outputs, 1)
                if output_transform is not None:
                    outputs = output_transform.inverse_transform(outputs)
                predictions.extend(outputs.cpu().numpy())
        predictions = np.array(predictions)
        return predictions

    def get_targets(self, data_loader, output_transform=None):
        """
        Gets the targets from the data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for test data.

        Returns:
            np.ndarray: Array of targets.
        """
        targets = []
        for inputs, labels in data_loader:
            if output_transform is not None:
                labels = output_transform.inverse_transform(labels)
            targets.extend(labels.cpu().numpy())
        targets = np.array(targets)
        return targets
    

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
                total_loss += loss

        if len(total_loss.shape) >= 1:
            total_loss = total_loss.sum()

        return total_loss.item() / len(test_loader)
    
    def save_model(self, path):
        """
        Saves the model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), path)