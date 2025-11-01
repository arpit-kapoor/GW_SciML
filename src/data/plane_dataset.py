import os
import numpy as np
import torch
from torch.utils.data import Dataset


class GWPlaneDataset(Dataset):
    """
    PyTorch Dataset for training the GW model using 2D plane data.

    This dataset loads plane data organized by timestep sequences, where each sample
    is a temporal sequence from a specific plane. Samples from the same plane should
    be batched together using the PlaneBatchSampler.
    
    Data structure expected:
    - input_sequences: dict with keys per plane, each containing:
        - 'input_geom': (n_sequences, n_bc_nodes * alpha, 3) - S, Z, T coordinates for boundary conditions
        - 'input_data': (n_sequences, n_bc_nodes * alpha, 2) - head, mass_conc values at boundary
        - 'latent_geom': (n_sequences, alpha, n_nodes, 3) - S, Z, T coordinates for latent grid
        - 'latent_features': (n_sequences, alpha, n_nodes, 4) - X, Y, head, mass_conc on latent grid
    - output_sequences: dict with keys per plane, each containing:
        - 'latent_geom': (n_sequences, alpha, n_nodes, 3) - S, Z, T coordinates for output
        - 'latent_features': (n_sequences, alpha, n_nodes, 4) - X, Y, head, mass_conc for output
    """

    def __init__(
        self,
        input_sequences,
        output_sequences,
        coord_transform=None,
        obs_transform=None,
        fill_nan_value=-999.0
    ):
        """
        Initialize the GWPlaneDataset.

        Args:
            input_sequences (dict): Dictionary mapping plane_id to input sequence data.
            output_sequences (dict): Dictionary mapping plane_id to output sequence data.
            coord_transform (callable, optional): Transform function for coordinates.
            obs_transform (callable, optional): Transform function for observations.
            fill_nan_value (float): Value to replace NaN values with (default: -999.0).
        """
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences
        self.coord_transform = coord_transform
        self.obs_transform = obs_transform
        self.fill_nan_value = fill_nan_value

        # Build index mapping: (plane_id, sequence_idx) -> global_idx
        self.index_map = []
        self.plane_ids = sorted(input_sequences.keys())
        
        for plane_id in self.plane_ids:
            n_sequences = len(input_sequences[plane_id]['input_geom'])
            for seq_idx in range(n_sequences):
                self.index_map.append((plane_id, seq_idx))
        
        # Cache plane_ids for fast access by PlaneBatchSampler
        self._patch_ids_cache = None
        
        print(f"Initialized GWPlaneDataset with {len(self.index_map)} sequences across {len(self.plane_ids)} planes")

    def __len__(self):
        """
        Return the total number of sequences in the dataset.

        Returns:
            int: Total number of sequences across all planes.
        """
        return len(self.index_map)

    def __getitem__(self, idx):
        """
        Retrieve a single data sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing:
                - 'plane_id': ID of the plane this sequence belongs to
                - 'input_geom': Input boundary condition geometry (n_bc_nodes * alpha, 3)
                - 'input_data': Input boundary condition values (n_bc_nodes * alpha, 2)
                - 'latent_geom': Latent grid geometry for input (alpha, n_nodes, 3)
                - 'latent_features': Latent grid features for input (alpha, n_nodes, 4)
                - 'output_latent_geom': Output latent grid geometry (alpha, n_nodes, 3)
                - 'output_latent_features': Output latent grid features (alpha, n_nodes, 4)
        """
        plane_id, seq_idx = self.index_map[idx]
        
        # Get input data
        input_geom = self.input_sequences[plane_id]['input_geom'][seq_idx]
        input_data = self.input_sequences[plane_id]['input_data'][seq_idx]
        latent_geom = self.input_sequences[plane_id]['latent_geom'][seq_idx]
        latent_features = self.input_sequences[plane_id]['latent_features'][seq_idx]
        
        # Get output data
        output_latent_geom = self.output_sequences[plane_id]['latent_geom'][seq_idx]
        output_latent_features = self.output_sequences[plane_id]['latent_features'][seq_idx]
        
        # Convert to tensors and handle NaN values
        input_geom = torch.from_numpy(input_geom).float()
        input_data = torch.from_numpy(input_data).float()
        latent_geom = torch.from_numpy(latent_geom).float()
        latent_features = torch.from_numpy(latent_features).float()
        output_latent_geom = torch.from_numpy(output_latent_geom).float()
        output_latent_features = torch.from_numpy(output_latent_features).float()
        
        # Replace NaN values
        if self.fill_nan_value is not None:
            input_geom = torch.nan_to_num(input_geom, nan=self.fill_nan_value)
            input_data = torch.nan_to_num(input_data, nan=self.fill_nan_value)
            latent_geom = torch.nan_to_num(latent_geom, nan=self.fill_nan_value)
            latent_features = torch.nan_to_num(latent_features, nan=self.fill_nan_value)
            output_latent_geom = torch.nan_to_num(output_latent_geom, nan=self.fill_nan_value)
            output_latent_features = torch.nan_to_num(output_latent_features, nan=self.fill_nan_value)
        
        # Apply transforms if provided
        if self.coord_transform is not None:
            input_geom = self.coord_transform(input_geom)
            latent_geom = self.coord_transform(latent_geom)
            output_latent_geom = self.coord_transform(output_latent_geom)
        
        if self.obs_transform is not None:
            input_data = self.obs_transform(input_data)
            latent_features = self.obs_transform(latent_features)
            output_latent_features = self.obs_transform(output_latent_features)
        
        return {
            'plane_id': plane_id,
            'input_geom': input_geom,
            'input_data': input_data,
            'latent_geom': latent_geom,
            'latent_features': latent_features,
            'output_latent_geom': output_latent_geom,
            'output_latent_features': output_latent_features
        }
    
    def get_all_patch_ids(self):
        """
        Optimized method to get all plane_ids (used as patch_ids) at once for PlaneBatchSampler.
        
        This method caches the result to avoid repeated computation.
        
        Returns:
            np.ndarray: Array of plane_ids for all samples in the dataset.
        """
        if self._patch_ids_cache is None:
            print("Building plane_ids cache...")
            self._patch_ids_cache = np.array([plane_id for plane_id, _ in self.index_map], dtype=np.int32)
            print(f"Cached {len(self._patch_ids_cache)} plane_ids")
        
        return self._patch_ids_cache


class GWPlaneDatasetFromFiles(Dataset):
    """
    PyTorch Dataset for training the GW model using 2D plane data loaded from files.

    This version loads data from disk on-demand rather than keeping everything in memory.
    Useful for large datasets that don't fit in memory.
    
    Expected directory structure:
    data_dir/
        plane_00/
            input_geom.npy
            input_data.npy
            latent_geom.npy
            latent_features.npy
            output_latent_geom.npy
            output_latent_features.npy
        plane_01/
            ...
    """

    def __init__(
        self,
        data_dir,
        coord_transform=None,
        obs_transform=None,
        fill_nan_value=-999.0
    ):
        """
        Initialize the GWPlaneDatasetFromFiles.

        Args:
            data_dir (str): Path to the directory containing plane subdirectories.
            coord_transform (callable, optional): Transform function for coordinates.
            obs_transform (callable, optional): Transform function for observations.
            fill_nan_value (float): Value to replace NaN values with (default: -999.0).
        """
        self.data_dir = data_dir
        self.coord_transform = coord_transform
        self.obs_transform = obs_transform
        self.fill_nan_value = fill_nan_value

        # Build index mapping by scanning directory
        self.index_map = []
        self.plane_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('plane_')])
        
        for plane_dir in self.plane_dirs:
            plane_id = int(plane_dir.split('_')[-1])
            plane_path = os.path.join(data_dir, plane_dir)
            
            # Load one file to determine number of sequences
            input_geom_path = os.path.join(plane_path, 'input_geom.npy')
            if os.path.exists(input_geom_path):
                input_geom = np.load(input_geom_path)
                n_sequences = len(input_geom)
                
                for seq_idx in range(n_sequences):
                    self.index_map.append((plane_id, plane_dir, seq_idx))
        
        # Cache plane_ids for fast access by PlaneBatchSampler
        self._patch_ids_cache = None
        
        print(f"Initialized GWPlaneDatasetFromFiles with {len(self.index_map)} sequences across {len(self.plane_dirs)} planes")

    def __len__(self):
        """
        Return the total number of sequences in the dataset.

        Returns:
            int: Total number of sequences across all planes.
        """
        return len(self.index_map)

    def __getitem__(self, idx):
        """
        Retrieve a single data sample from the dataset by loading from disk.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing all input/output data for the sequence.
        """
        plane_id, plane_dir, seq_idx = self.index_map[idx]
        plane_path = os.path.join(self.data_dir, plane_dir)
        
        # Load data from disk
        input_geom = np.load(os.path.join(plane_path, 'input_geom.npy'))[seq_idx]
        input_data = np.load(os.path.join(plane_path, 'input_data.npy'))[seq_idx]
        latent_geom = np.load(os.path.join(plane_path, 'latent_geom.npy'))[seq_idx]
        latent_features = np.load(os.path.join(plane_path, 'latent_features.npy'))[seq_idx]
        output_latent_geom = np.load(os.path.join(plane_path, 'output_latent_geom.npy'))[seq_idx]
        output_latent_features = np.load(os.path.join(plane_path, 'output_latent_features.npy'))[seq_idx]
        
        # Convert to tensors and handle NaN values
        input_geom = torch.from_numpy(input_geom).float()
        input_data = torch.from_numpy(input_data).float()
        latent_geom = torch.from_numpy(latent_geom).float()
        latent_features = torch.from_numpy(latent_features).float()
        output_latent_geom = torch.from_numpy(output_latent_geom).float()
        output_latent_features = torch.from_numpy(output_latent_features).float()
        
        # Replace NaN values
        if self.fill_nan_value is not None:
            input_geom = torch.nan_to_num(input_geom, nan=self.fill_nan_value)
            input_data = torch.nan_to_num(input_data, nan=self.fill_nan_value)
            latent_geom = torch.nan_to_num(latent_geom, nan=self.fill_nan_value)
            latent_features = torch.nan_to_num(latent_features, nan=self.fill_nan_value)
            output_latent_geom = torch.nan_to_num(output_latent_geom, nan=self.fill_nan_value)
            output_latent_features = torch.nan_to_num(output_latent_features, nan=self.fill_nan_value)
        
        # Apply transforms if provided
        if self.coord_transform is not None:
            input_geom = self.coord_transform(input_geom)
            latent_geom = self.coord_transform(latent_geom)
            output_latent_geom = self.coord_transform(output_latent_geom)
        
        if self.obs_transform is not None:
            input_data = self.obs_transform(input_data)
            latent_features = self.obs_transform(latent_features)
            output_latent_features = self.obs_transform(output_latent_features)
        
        return {
            'plane_id': plane_id,
            'input_geom': input_geom,
            'input_data': input_data,
            'latent_geom': latent_geom,
            'latent_features': latent_features,
            'output_latent_geom': output_latent_geom,
            'output_latent_features': output_latent_features
        }
    
    def get_all_patch_ids(self):
        """
        Optimized method to get all plane_ids (used as patch_ids) at once for PlaneBatchSampler.
        
        This method caches the result to avoid repeated computation.
        
        Returns:
            np.ndarray: Array of plane_ids for all samples in the dataset.
        """
        if self._patch_ids_cache is None:
            print("Building plane_ids cache...")
            self._patch_ids_cache = np.array([plane_id for plane_id, _, _ in self.index_map], dtype=np.int32)
            print(f"Cached {len(self._patch_ids_cache)} plane_ids")
        
        return self._patch_ids_cache
