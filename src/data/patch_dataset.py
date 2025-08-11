import os
import numpy as np
from torch.utils.data import Dataset

class GWPatchDataset(Dataset):
    """
    PyTorch Dataset for training the GW model using patch data.

    This dataset loads patch data from a directory, applies optional coordinate and observation
    transforms, splits into training and validation sets, and generates input/output sequences
    for model training.
    """

    def __init__(
        self,
        data_path,
        dataset='train',
        coord_transform=None,
        obs_transform=None,
        val_ratio=0.3,
        input_window_size=10,
        output_window_size=10,
        target_cols_idx=None
    ):
        """
        Initialize the GWPatchDataset.

        Args:
            data_path (str): Path to the data directory containing patch subdirectories.
            dataset (str): Dataset type, either 'train' or 'val'.
            coord_transform (callable, optional): Transform function for coordinates.
            obs_transform (callable, optional): Transform function for observations.
            val_ratio (float): Ratio of data to use for validation.
            input_window_size (int): Number of time steps in each input sequence.
            output_window_size (int): Number of time steps in each output sequence.
        """
        self.data_path = data_path
        self.dataset = dataset
        self.coord_transform = coord_transform
        self.obs_transform = obs_transform
        self.val_ratio = val_ratio

        # Load and process patch data
        patch_data = self.load_patch_data(
            data_path,
            val_ratio=val_ratio,
            dataset=dataset,
            target_cols_idx=target_cols_idx
        )

        # Create input/output sequences from patch data
        patch_data = self.create_sequence_data(
            patch_data,
            input_window_size=input_window_size,
            output_window_size=output_window_size
        )

        # Store processed data
        self.coords, self.input_sequence, self.output_sequence = patch_data

    def create_sequence_data(
        self,
        patch_data,
        input_window_size=10,
        output_window_size=10
    ):
        """
        Create input/output sequence data from patch data.

        Args:
            patch_data (list): List of patch data dictionaries.
            input_window_size (int): Number of time steps in each input sequence.
            output_window_size (int): Number of time steps in each output sequence.

        Returns:
            tuple: (coords, input_sequence, output_sequence)
                - coords: List of coordinate dictionaries for each sequence.
                - input_sequence: List of input sequence dictionaries.
                - output_sequence: List of output sequence dictionaries.
        """
        coords = []
        input_sequence = []
        output_sequence = []

        for patch in patch_data:
            core_obs = patch['core_obs']
            ghost_obs = patch['ghost_obs']

            # Generate sequences for this patch
            for i in range(0, len(core_obs) - input_window_size, output_window_size):
                coords.append({
                    'patch_id': patch['patch_id'],
                    'core_coords': patch['core_coords'],
                    'ghost_coords': patch['ghost_coords']
                })
                input_sequence.append({
                    'core_in': core_obs[i:i + input_window_size],
                    'ghost_in': ghost_obs[i:i + input_window_size]
                })
                output_sequence.append({
                    'core_out': core_obs[i + input_window_size:i + input_window_size + output_window_size],
                    'ghost_out': ghost_obs[i + input_window_size:i + input_window_size + output_window_size]
                })

        return coords, input_sequence, output_sequence

    def load_patch_data(
        self,
        data_path,
        val_ratio=0.3,
        dataset='train',
        target_cols_idx=None
    ):
        """
        Load patch data from the data directory.

        Args:
            data_path (str): Path to the data directory.
            val_ratio (float): Validation ratio.
            dataset (str): Dataset type, either 'train' or 'val'.

        Returns:
            list: List of patch data dictionaries, each containing patch_id, core/ghost coords and obs.
        """
        patch_data = []

        for idx, patch_dir in enumerate(sorted(os.listdir(data_path))):
            # Skip hidden files/directories
            if patch_dir.startswith('.'):
                continue

            patch_dir_path = os.path.join(data_path, patch_dir)
            core_coords = np.load(os.path.join(patch_dir_path, 'core_coords.npy'))
            core_obs = np.load(os.path.join(patch_dir_path, 'core_obs.npy'))
            ghost_coords = np.load(os.path.join(patch_dir_path, 'ghost_coords.npy'))
            ghost_obs = np.load(os.path.join(patch_dir_path, 'ghost_obs.npy'))

            # Select target columns
            if target_cols_idx is not None:
                core_obs = core_obs[..., target_cols_idx]
                ghost_obs = ghost_obs[..., target_cols_idx]

            # Determine split index for train/val
            train_idx = int(len(core_obs) * (1 - val_ratio))

            # Extract patch_id from directory name
            patch_id = int(patch_dir.split('_')[-1])

            # Apply coordinate and observation transforms if provided
            if self.coord_transform is not None:
                core_coords = self.coord_transform(core_coords)
                ghost_coords = self.coord_transform(ghost_coords)
            if self.obs_transform is not None:
                core_obs = self.obs_transform(core_obs)
                ghost_obs = self.obs_transform(ghost_obs)

            if dataset == 'train':
                patch_data.append({
                    'patch_id': patch_id,
                    'core_coords': core_coords,
                    'core_obs': core_obs[:train_idx],
                    'ghost_coords': ghost_coords,
                    'ghost_obs': ghost_obs[:train_idx]
                })
            else:
                patch_data.append({
                    'patch_id': patch_id,
                    'core_coords': core_coords,
                    'core_obs': core_obs[train_idx:],
                    'ghost_coords': ghost_coords,
                    'ghost_obs': ghost_obs[train_idx:]
                })

        return patch_data

    def __len__(self):
        """
        Return the number of input sequences in the dataset.

        Returns:
            int: Number of input sequences.
        """
        return len(self.input_sequence)

    def __getitem__(self, idx):
        """
        Retrieve a single data sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing input sequence, output sequence, and coordinates for the sample.
        """
        data_dict = {}
        data_dict.update(self.input_sequence[idx])
        data_dict.update(self.output_sequence[idx])
        data_dict.update(self.coords[idx])
        return data_dict