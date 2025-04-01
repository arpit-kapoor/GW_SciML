import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class Normalize:
    def __init__(self, mean, std):
        """
        Args:
            mean (array-like): Mean values for each feature.
            std (array-like): Standard deviation values for each feature.
        """
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        """
        Args:
            sample (array-like): Input sample to normalize.
        Returns:
            array-like: Normalized sample.
        """
        sample = torch.tensor(sample, dtype=torch.float32)
        return (sample - self.mean) / self.std

class GWDataset(Dataset):
    val_ratio = 0.3

    def __init__(self, data_path, dataset='train', input_transform=None, output_transform=None):
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