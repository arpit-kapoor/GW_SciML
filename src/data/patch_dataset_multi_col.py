import os
import numpy as np
from torch.utils.data import Dataset

class GWPatchDatasetMultiCol(Dataset):
    """
    PyTorch Dataset for training the GW model using patch data with multi-column support.

    This dataset loads patch data from a directory, applies optional coordinate and observation
    transforms, splits into training and validation sets, and generates input/output sequences
    for model training with multiple target columns concatenated along the last dimension.
    
    Key difference from GWPatchDataset:
    - Accepts `target_col_indices` (list) instead of `target_col_idx` (single int)
    - Concatenates multiple target columns along the last dimension
    - Sequences have shape [N_points, window_size * n_target_cols]
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
        target_col_indices=None
    ):
        """
        Initialize the GWPatchDatasetMultiCol.

        Args:
            data_path (str): Path to the data directory containing patch subdirectories.
            dataset (str): Dataset type, either 'train' or 'val'.
            coord_transform (callable, optional): Transform function for coordinates.
            obs_transform (callable, optional): Transform function for observations.
            val_ratio (float): Ratio of data to use for validation.
            input_window_size (int): Number of time steps in each input sequence.
            output_window_size (int): Number of time steps in each output sequence.
            target_col_indices (list of int, optional): Indices of target columns to extract and concatenate.
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
            target_col_indices=target_col_indices
        )
        
        # Compute weights based on temporal variances across all patches
        patch_data = self.compute_weights(patch_data)
        
        # Create input/output sequences from patch data
        patch_data = self.create_sequence_data(
            patch_data,
            input_window_size=input_window_size,
            output_window_size=output_window_size
        )

        # Store processed data
        self.coords, self.input_sequence, self.output_sequence = patch_data
        
        # Cache patch_ids for fast access (optimization for PatchBatchSampler)
        self._patch_ids_cache = None

    def create_sequence_data(
        self,
        patch_data,
        input_window_size=10,
        output_window_size=10
    ):
        """
        Create input/output sequence data from patch data with multi-column concatenation.

        Args:
            patch_data (list): List of patch data dictionaries.
            input_window_size (int): Number of time steps in each input sequence.
            output_window_size (int): Number of time steps in each output sequence.

        Returns:
            tuple: (coords, input_sequence, output_sequence)
                - coords: List of coordinate dictionaries for each sequence.
                - input_sequence: List of input sequence dictionaries with concatenated columns.
                - output_sequence: List of output sequence dictionaries with concatenated columns.
        """
        coords = []
        input_sequence = []
        output_sequence = []

        for patch in patch_data:
            core_obs = patch['core_obs']  # Shape: [time_steps, n_points, n_target_cols]
            ghost_obs = patch['ghost_obs']  # Shape: [time_steps, n_points, n_target_cols]

            # Generate sequences for this patch
            for i in range(0, len(core_obs) - (input_window_size + output_window_size) + 1):
                coords.append({
                    'patch_id': patch['patch_id'],
                    'core_coords': patch['core_coords'],
                    'ghost_coords': patch['ghost_coords'],
                    'weights': patch['weights']  # [n_points]
                })
                
                # Extract sequences: [window_size, n_points, n_target_cols]
                core_in_seq = core_obs[i:i + input_window_size]
                ghost_in_seq = ghost_obs[i:i + input_window_size]
                core_out_seq = core_obs[i + input_window_size:i + input_window_size + output_window_size]
                ghost_out_seq = ghost_obs[i + input_window_size:i + input_window_size + output_window_size]
                
                # Reshape and concatenate: [n_points, window_size * n_target_cols]
                # This flattens across time and variable dimensions
                core_in = self._concat_sequence(core_in_seq)
                ghost_in = self._concat_sequence(ghost_in_seq)
                core_out = self._concat_sequence(core_out_seq)
                ghost_out = self._concat_sequence(ghost_out_seq)
                
                input_sequence.append({
                    'core_in': core_in,
                    'ghost_in': ghost_in
                })
                output_sequence.append({
                    'core_out': core_out,
                    'ghost_out': ghost_out
                })

        return coords, input_sequence, output_sequence
    
    def _concat_sequence(self, seq):
        """
        Concatenate sequence across time and variable dimensions.
        
        Args:
            seq: Array of shape [window_size, n_points, n_target_cols]
            
        Returns:
            Array of shape [n_points, window_size * n_target_cols]
        """
        # Transpose to [n_points, window_size, n_target_cols]
        seq = seq.permute(1, 0, 2)
        
        # Reshape to [n_points, window_size * n_target_cols]
        n_points = seq.shape[0]
        seq_flat = seq.reshape(n_points, -1)
        
        return seq_flat

    def load_patch_data(
        self,
        data_path,
        val_ratio=0.3,
        dataset='train',
        target_col_indices=None
    ):
        """
        Load patch data from the data directory with multi-column support.

        Args:
            data_path (str): Path to the data directory.
            val_ratio (float): Validation ratio.
            dataset (str): Dataset type, either 'train' or 'val'.
            target_col_indices (list of int, optional): Indices of target columns to extract and concatenate.

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

            # Determine split index for train/val
            train_idx = int(len(core_obs) * (1 - val_ratio))

            # Extract patch_id from directory name
            patch_id = int(patch_dir.split('_')[-1])
            
            # Compute temporal variances for mass concentration (before any transforms)
            # Use training data only for variance computation
            core_obs_train = core_obs[:train_idx]  # [train_timesteps, n_points, n_cols]
            # Variance of mass concentration (index 0) across time for each node
            temporal_variances = np.var(core_obs_train[..., 0], axis=0)  # [n_points]

            # Apply coordinate and observation transforms if provided
            if self.coord_transform is not None:
                core_coords = self.coord_transform(core_coords)
                ghost_coords = self.coord_transform(ghost_coords)
            if self.obs_transform is not None:
                core_obs = self.obs_transform(core_obs)
                ghost_obs = self.obs_transform(ghost_obs)
            
            # Select and concatenate target columns
            if target_col_indices is not None:
                # Extract selected columns: [..., indices] -> [..., n_target_cols]
                core_obs = core_obs[..., target_col_indices]
                ghost_obs = ghost_obs[..., target_col_indices]
            
            # Ensure we have 3D arrays: [time_steps, n_points, n_target_cols]
            if core_obs.ndim == 2:
                # If only one column selected, add dimension
                core_obs = core_obs[..., np.newaxis]
                ghost_obs = ghost_obs[..., np.newaxis]

            if dataset == 'train':
                patch_data.append({
                    'patch_id': patch_id,
                    'core_coords': core_coords,
                    'core_obs': core_obs[:train_idx],
                    'ghost_coords': ghost_coords,
                    'ghost_obs': ghost_obs[:train_idx],
                    'temporal_variances': temporal_variances  # [n_points]
                })
            else:
                patch_data.append({
                    'patch_id': patch_id,
                    'core_coords': core_coords,
                    'core_obs': core_obs[train_idx:],
                    'ghost_coords': ghost_coords,
                    'ghost_obs': ghost_obs[train_idx:],
                    'temporal_variances': temporal_variances  # [n_points] - same for val
                })

        return patch_data
    
    def compute_weights(self, patch_data, alpha=0.25, beta=1.0, use_log_scaling=True):
        """
        Compute variance-aware weights for each node across all patches.
        
        Weights are computed based on temporal variances of mass concentration,
        with higher variance (more dynamic) regions receiving higher weights.
        
        Args:
            patch_data (list): List of patch dictionaries containing temporal_variances
            alpha (float): Base weight for low-variance nodes (0 < alpha < 1)
            beta (float): Exponent controlling emphasis on high-variance nodes (default: 1.0)
            use_log_scaling (bool): Use logarithmic scaling for more balanced weights
            
        Returns:
            list: Updated patch_data with 'weights' key added to each patch
        """
        # Collect all temporal variances across all patches
        all_variances = []
        for patch in patch_data:
            all_variances.append(patch['temporal_variances'])
        
        # Concatenate all variances to get dataset-level statistics
        all_variances = np.concatenate(all_variances, axis=0)  # [total_nodes_across_patches]
        
        # Compute mean variance across entire dataset
        mean_variance = np.mean(all_variances)
        max_variance = np.max(all_variances)
        clip_variance = np.percentile(all_variances, 99) 
        eps = 1e-6  # Avoid division by zero
        
        # Compute weights for each patch
        for patch in patch_data:
            variances = patch['temporal_variances']

            # Clip variances to reduce outlier impact
            var_clip = np.clip(variances, 0.0, clip_variance)

            if use_log_scaling:
                # Use log scaling for more balanced weight distribution
                var_log = np.log1p(var_clip / (mean_variance + eps))
                var_norm = var_log / (np.log1p(clip_variance / (mean_variance + eps)) + eps)
            else:
                # Original: linear scaling
                var_norm = var_clip / (max_variance + eps)

            # Compute weights using the specified formula
            weights = alpha + (1 - alpha) * (var_norm ** beta)

            # Normalize weights to have mean 1.0
            weights /= np.mean(weights)
            
            # Store weights in patch data
            patch['weights'] = weights.astype(np.float32)  # [n_points]
        
        # Compute and print weight statistics
        weight_stats = np.concatenate([p['weights'] for p in patch_data])
        print(f"Computed variance-aware weights for {len(patch_data)} patches")
        print(f"Dataset variance range: [{all_variances.min():.6f}, {all_variances.max():.6f}]")
        print(f"Dataset mean variance: {mean_variance:.6f}")
        print(f"Weight range: [{weight_stats.min():.4f}, {weight_stats.max():.4f}]")
        print(f"Weight std: {weight_stats.std():.4f}")
        
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
    
    def get_all_patch_ids(self):
        """
        Optimized method to get all patch_ids at once for PatchBatchSampler.
        
        This method caches the result to avoid repeated computation.
        
        Returns:
            np.ndarray: Array of patch_ids for all samples in the dataset.
        """
        if self._patch_ids_cache is None:
            print("Building patch_ids cache...")
            self._patch_ids_cache = np.array([coord['patch_id'] for coord in self.coords], dtype=np.int32)
            print(f"Cached {len(self._patch_ids_cache)} patch_ids")
        
        return self._patch_ids_cache

