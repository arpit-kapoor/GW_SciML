import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class GWDataset(Dataset):
    """
    PyTorch Dataset for training the GW model using tabular data.
    
    This dataset loads CSV files containing groundwater data, applies coordinate
    normalization, splits into training and validation sets, and provides
    input/output pairs for model training.
    """

    def __init__(
        self, 
        data_path, 
        dataset='train', 
        input_transform=None, 
        output_transform=None, 
        val_ratio=0.3
    ):
        """
        Initialize the GWDataset.
        
        Args:
            data_path (str): Path to the dataset directory containing CSV files.
            dataset (str): Dataset type, either 'train' or 'val'.
            input_transform (callable, optional): Transform function for input data.
            output_transform (callable, optional): Transform function for output data.
            val_ratio (float): Ratio of data to use for validation.
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
            tuple: (data, origin)
                - data (pd.DataFrame): The loaded and processed data.
                - origin (np.ndarray): The origin coordinates for normalization.
        """
        datafiles = sorted(os.listdir(self.data_path))

        # Select data files based on the dataset type
        split_index = int((1 - self.val_ratio) * len(datafiles))
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
        Return the total number of samples in the dataset.
        
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single data sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: (inputs, targets)
                - inputs: Input features (X, Y, Z, time).
                - targets: Target values (head, mass_concentration).
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
