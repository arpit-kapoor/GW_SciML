import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU):
        """
        Generalized Multilayer Perceptron (MLP) class.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (list of int): List containing the number of neurons in each hidden layer.
            output_size (int): Number of output features.
            activation (torch.nn.Module): Activation function to use between layers (default: ReLU).
        """
        super(MLP, self).__init__()
        layers = []

        # Input layer to first hidden layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation())
            prev_size = hidden_size

        # Last hidden layer to output layer
        layers.append(nn.Linear(prev_size, output_size))

        # Combine all layers into a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)