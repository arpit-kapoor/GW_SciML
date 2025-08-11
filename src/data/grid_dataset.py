import os
import numpy as np
from torch.utils.data import Dataset


class GWGridDataset(Dataset):
    """
    PyTorch Dataset for training the GW model using gridded data.
    
    This dataset loads gridded time series data, creates input/output sequences for 
    temporal prediction, and applies optional transforms to the data.
    
    Attributes:
        FILLVAL (int): Default fill value for missing data.
    """

    FILLVAL = -999

    def __init__(
        self, 
        data_path, 
        dataset='train', 
        input_transform=None, 
        output_transform=None, 
        in_window_size=10, 
        out_window_size=10, 
        val_ratio=0.3, 
        fillval=None
    ):
        """
        Initialize the GWGridDataset.
        
        Args:
            data_path (str): Path to the dataset directory containing .npy files.
            dataset (str): Dataset type, either 'train' or 'val'.
            input_transform (callable, optional): Transform function for input sequences.
            output_transform (callable, optional): Transform function for output sequences.
            in_window_size (int): Number of time steps in each input sequence.
            out_window_size (int): Number of time steps in each output sequence.
            val_ratio (float): Ratio of data to use for validation.
            fillval (int, optional): Custom fill value for missing data.
        """
        # Initialize parameters
        self.data_path = data_path
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.dataset = dataset
        self.in_window_size = in_window_size
        self.out_window_size = out_window_size
        
        if fillval is not None:
            self.FILLVAL = fillval

        # Load data
        self._data, self.x_grid, self.y_grid, self.z_grid = self._load_data()

        # Select data based on the dataset type
        split_index = int((1 - val_ratio) * len(self._data))
        if self.dataset == 'train':
            self._data = self._data[:split_index]
        elif self.dataset == 'val':
            self._data = self._data[split_index:]
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset}")

        # Create sequences
        self.X, self.y = self.create_sequences(
            self._data, 
            self.in_window_size, 
            self.out_window_size
        )

    def _load_data(self):
        """
        Load the dataset from the given path.
        
        Returns:
            tuple: (data, x_grid, y_grid, z_grid)
                - data (np.ndarray): The time series data array.
                - x_grid (np.ndarray): The x coordinate grid array.
                - y_grid (np.ndarray): The y coordinate grid array.
                - z_grid (np.ndarray): The z coordinate grid array.
        """
        # Data file paths
        datapath = os.path.join(self.data_path, 'data.npy')
        x_grid_path = os.path.join(self.data_path, 'x_grid.npy')
        y_grid_path = os.path.join(self.data_path, 'y_grid.npy')
        z_grid_path = os.path.join(self.data_path, 'z_grid.npy')

        # Load data
        data = np.load(datapath)
        x_grid = np.load(x_grid_path)
        y_grid = np.load(y_grid_path)
        z_grid = np.load(z_grid_path)

        # Replace NaN values with fill value
        data = np.where(np.isnan(data), self.FILLVAL, data)
        
        return data, x_grid, y_grid, z_grid

    def create_sequences(self, data, in_window_size, out_window_size):
        """
        Create input/output sequences from the time series data.
        
        Args:
            data (np.ndarray): The time series data.
            in_window_size (int): Number of time steps in each input sequence.
            out_window_size (int): Number of time steps in each output sequence.
            
        Returns:
            tuple: (X, y)
                - X (np.ndarray): Input sequences.
                - y (np.ndarray): Output sequences.
        """
        # Create sequences
        sequences = []
        for i in range(len(data) - in_window_size - out_window_size):
            sequences.append(data[i:i + in_window_size + out_window_size])
        
        # Convert to numpy array
        sequences = np.array(sequences)

        # Split into input and output
        X = sequences[:, :in_window_size, :]
        y = sequences[:, in_window_size:in_window_size + out_window_size, :]

        # Apply transforms
        if self.input_transform:
            X = self.input_transform(X)
        if self.output_transform:
            y = self.output_transform(y)

        return X, y

    def __len__(self):
        """
        Return the number of sequences in the dataset.
        
        Returns:
            int: The total number of sequences in the dataset.
        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Retrieve a single data sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: (input_sequence, output_sequence)
                - input_sequence: Input sequence at the given index.
                - output_sequence: Output sequence at the given index.
        """
        return self.X[idx], self.y[idx]
