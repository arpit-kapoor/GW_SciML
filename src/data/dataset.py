import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class Normalize:
    def __init__(self, mean, std):
        """
        Args:
            mean (array-like): Mean values for each feature.
            std (array-like): Standard deviation values for each feature.
        """
        self.mean = torch.tensor(mean, dtype=torch.float64)
        self.std = torch.tensor(std, dtype=torch.float64)

    def __call__(self, sample):
        """
        Args:
            sample (array-like): Input sample to normalize.
        Returns:
            array-like: Normalized sample.
        """
        sample = torch.tensor(sample, dtype=torch.float64)
        return (sample - self.mean) / self.std

    def inverse_transform(self, sample):
        sample = torch.tensor(sample, dtype=torch.float64)
        return sample*self.std + self.mean

class GWDataset(Dataset):

    def __init__(self, data_path, dataset='train', input_transform=None, output_transform=None, val_ratio=0.3):
        """
        Args:
            data_path (str): Path to the dataset file or directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = data_path
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.dataset = dataset
        self.val_ratio = val_ratio
        self.data, self.origin = self._load_data()

    def _load_data(self):
        """
        Load the dataset from the given path.
        Returns:
            list: A list of data samples.
        """
        datafiles = sorted(os.listdir(self.data_path))

        # Select data files based on the dataset type
        split_index = int((1-self.val_ratio)*len(datafiles))
        if self.dataset == 'train':
            self.datafiles = datafiles[:split_index]
        elif self.dataset == 'val':
            self.datafiles = datafiles[split_index:]
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset}")

        data = []

        # Read all data files
        for datafile in self.datafiles:
            df = pd.read_csv(os.path.join(self.data_path, datafile))
            data.append(df)

        # Concatenate all data files
        data = pd.concat(data, axis=0).reset_index(drop=True)

        # Normalize coordinates
        origin = data[['X', 'Y', 'Z']].min().values
        data.loc[:, ['X', 'Y', 'Z']] -= origin

        return data, origin
        

    def __len__(self):
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            dict: A dictionary containing the data sample.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the sample at the given index
        sample = self.data.iloc[idx][['X', 'Y', 'Z', 'time (d)', 
                                      'head', 'mass_concentration']].values
        inputs = sample[:-2]
        targets = sample[-2:]

        if self.input_transform:
            inputs = self.input_transform(inputs)
        
        if self.output_transform:
            targets = self.output_transform(targets)

        return inputs, targets



class GWGridDataset(Dataset):

    FILLVAL = -999

    def __init__(self, data_path, dataset='train', input_transform=None, output_transform=None, in_window_size=10, out_window_size=10, val_ratio=0.3, fillval=None):
        """
        Args:
            data_path (str): Path to the dataset file or directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
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
        split_index = int((1-val_ratio)*len(self._data))
        if self.dataset == 'train':
            self._data = self._data[:split_index]
        elif self.dataset == 'val':
            self._data = self._data[split_index:]
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset}")


        # Create sequences
        self.X, self.y = self.create_sequences(self._data, 
                                               self.in_window_size, 
                                               self.out_window_size)

    def _load_data(self):
        """
        Load the dataset from the given path.
        Returns:
            data (np.ndarray): The data array.
            x_grid (np.ndarray): The x grid array.
            y_grid (np.ndarray): The y grid array.
            z_grid (np.ndarray): The z grid array.
        """
        # Data files
        datapath = os.path.join(self.data_path, 'data.npy')
        x_grid_path = os.path.join(self.data_path, 'x_grid.npy')
        y_grid_path = os.path.join(self.data_path, 'y_grid.npy')
        z_grid_path = os.path.join(self.data_path, 'z_grid.npy')

        # Load data
        data = np.load(datapath)
        x_grid = np.load(x_grid_path)
        y_grid = np.load(y_grid_path)
        z_grid = np.load(z_grid_path)

        # Fillna with -1
        data = np.where(np.isnan(data), self.FILLVAL, data)
        
        return data, x_grid, y_grid, z_grid

    
    def create_sequences(self, data, in_window_size, out_window_size):
        """
        Create a sequence of data from the given data.
        """

        # Create sequences
        sequences = []
        for i in range(len(data) - in_window_size - out_window_size):
            sequences.append(data[i:i+in_window_size+out_window_size])
        
        # Convert to numpy array
        sequences = np.array(sequences)

        # Split into input and output
        X = sequences[:, :in_window_size, :]
        y = sequences[:, in_window_size:in_window_size+out_window_size, :]

        # Normalize
        if self.input_transform:
            X = self.input_transform(X)
        if self.output_transform:
            y = self.output_transform(y)

        return X, y

    def __len__(self):
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing the data sample.
        """
        return self.X[idx], self.y[idx]

