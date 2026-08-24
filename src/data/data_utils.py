"""
Data loading and transformation utilities for neural operator models.

This module provides generic data preparation functions that can be reused
across different datasets and model architectures.
"""

import os
import numpy as np
import pandas as pd
import torch

from src.data.transform import Normalize


def calculate_coord_transform(raw_data_dir, coord_columns=['X', 'Y', 'Z'], representative_file='0000.csv'):
    """
    Calculate mean and std of coordinates and create coordinate transform.

    Uses a representative CSV file to derive normalization stats for
    spatial coordinates. This ensures coordinates are zero-centered with unit variance.
    
    Args:
        raw_data_dir (str): Directory containing raw CSV files
        coord_columns (list): List of coordinate column names to normalize
        representative_file (str): Name of representative file to use for statistics
        
    Returns:
        Normalize: Transform object for coordinate normalization
    """
    # Read representative data file to compute statistics
    file_path = os.path.join(raw_data_dir, representative_file)
    df = pd.read_csv(file_path)

    # Calculate mean and std of coordinates
    coord_mean = df[coord_columns].mean().values
    coord_std = df[coord_columns].std().values

    # Print normalization statistics for debugging
    print(f"Coordinate mean: {coord_mean}")
    print(f"Coordinate std: {coord_std}")

    # Create coordinate transform
    coord_transform = Normalize(mean=coord_mean, std=coord_std)

    # Clean up memory
    del df
    return coord_transform


def calculate_obs_transform(raw_data_dir, target_obs_cols):
    """
    Calculate pooled mean and std of output variables across all files 
    and create an observation transform.
    
    Args:
        raw_data_dir (str): Directory containing raw CSV files
        target_obs_cols (list): List of observation column names to normalize
        
    Returns:
        Normalize: Transform object for observation normalization
    """
    csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {raw_data_dir}")

    obs_mean_arr, obs_std_arr, size_arr = [], [], []

    print(f"Calculating observation transform for columns: {target_obs_cols}")
    
    for csv_file in sorted(csv_files):
        file_path = os.path.join(raw_data_dir, csv_file)
        df = pd.read_csv(file_path)
        
        # Skip empty files to avoid NaN values in our math
        if len(df) == 0:
            continue
            
        file_level_mean = df[target_obs_cols].mean().values
        # Use ddof=0 for population standard deviation to match the pooled formula
        file_level_std = df[target_obs_cols].std(ddof=0).values 
        
        obs_mean_arr.append(file_level_mean)
        obs_std_arr.append(file_level_std)
        size_arr.append(len(df))

    # Convert to numpy arrays
    obs_mean_arr = np.array(obs_mean_arr)
    obs_std_arr = np.array(obs_std_arr)
    size_arr = np.expand_dims(np.array(size_arr), axis=-1)

    # 1. Calculate pooled mean
    total_size = np.sum(size_arr, axis=0)
    obs_mean = np.sum(size_arr * obs_mean_arr, axis=0) / total_size

    # 2. Calculate pooled variance, then take the square root for pooled std
    pooled_variance = (np.sum(size_arr * np.square(obs_std_arr), axis=0) + 
                       np.sum(size_arr * np.square(obs_mean_arr - obs_mean), axis=0)) / total_size
    obs_std = np.sqrt(pooled_variance)

    print(f"Output mean: {obs_mean}")
    print(f"Output std: {obs_std}")

    # Define output transform
    obs_transform = Normalize(mean=obs_mean, std=obs_std)

    return obs_transform


def calculate_forcings_transform():
    """
    Create a normalization transform for forcings data.
    Uses pre-computed mean and std values for forcings normalization.
    Returns:
        Normalize: Transform object for forcings normalization
    """
    # Pre-computed mean and std for forcings
    forcings_mean = np.array([ 1.48153188e+03,  9.42562257e-03,  3.84628900e-05, -8.05859849e-04])
    forcings_std = np.array([7.04689144e+03, 8.24747365e-02, 7.88193443e-04, 5.29570033e-02])

    # Create forcings transform
    forcings_transform = Normalize(mean=forcings_mean, std=forcings_std)
    
    return forcings_transform


def create_patch_datasets(dataset_class, patch_data_dir, coord_transform, obs_transform, **kwargs):
    """
    Create train/val datasets with normalization and sequencing.

    This is a generic factory function that works with any dataset class
    that accepts the standard parameters (data_path, dataset, transforms, etc.).
    
    Args:
        dataset_class (class): Dataset class to instantiate (e.g., GWPatchDatasetMultiCol)
        patch_data_dir (str): Directory containing patch data files
        coord_transform (Normalize): Normalization transform for coordinates
        obs_transform (Normalize): Normalization transform for observations
        **kwargs: Additional arguments to pass to the dataset class, including:
            - input_window_size (int): Number of input timesteps
            - output_window_size (int): Number of output timesteps
            - target_col_indices (list): Indices of target columns
            - Any other dataset-specific parameters
        
    Returns:
         tuple: (train_dataset, validation_dataset)
    """
    train_stride = kwargs.get('train_stride', 1)
    val_only = kwargs.get('val_only', False)
    
    if not val_only:
        # Create training dataset
        train_ds = dataset_class(
            data_path=patch_data_dir,
            dataset='train', 
            coord_transform=coord_transform, 
            obs_transform=obs_transform,
            input_window_size=kwargs.get('input_window_size', 10),
            output_window_size=kwargs.get('output_window_size', 10),
            target_col_indices=kwargs.get('target_col_indices', None),
            stride=train_stride,
            **{k: v for k, v in kwargs.items() if k not in ['input_window_size', 'output_window_size', 'target_col_indices', 'train_stride', 'val_stride', 'val_only']}
        )
    else:
        train_ds = None
    
    val_stride = kwargs.get('val_stride', 1)
    # Create validation dataset
    val_ds = dataset_class(
        data_path=patch_data_dir,
        dataset='val', 
        coord_transform=coord_transform, 
        obs_transform=obs_transform,
        input_window_size=kwargs.get('input_window_size', 10),
        output_window_size=kwargs.get('output_window_size', 10),
        target_col_indices=kwargs.get('target_col_indices', None),
        stride=val_stride,
        **{k: v for k, v in kwargs.items() if k not in ['input_window_size', 'output_window_size', 'target_col_indices', 'train_stride', 'val_stride', 'val_only']}
    )

    return train_ds, val_ds


def make_collate_fn(args, coord_dim=3):
    """
    Create a collate function for the 4D space-time FNO architecture.

    The sampler ensures a batch contains indices from a single `patch_id`.
    This collate function:
    1. Constructs 4D (x, y, z, t) coordinates from spatial coords + time values
    2. Assembles input features: [obs, current_forcings, future_forcings] = 10 channels
    3. Builds the 4D latent query grid (Nx, Ny, Nz, Nt)
    4. Provides separate input and output coordinate tensors

    Args:
        args (argparse.Namespace): Argument namespace containing device, latent grid dims,
            input/output window sizes, and dataset reference for time_values
        coord_dim (int): Spatial coordinate dimensionality (default: 3 for 3D)

    Returns:
        function: Collate function for DataLoader
    """
    def collate_fn(batch_samples):
        """
        Collate function that combines samples into a batch for 4D FNO.

        Args:
            batch_samples (list): List of sample dictionaries from the same patch

        Returns:
            dict: Batch dictionary with 4D coordinates, paired features, and targets
        """
        # All samples in the batch come from the same patch (by sampler design)
        core_coords = batch_samples[0]['core_coords']
        ghost_coords = batch_samples[0]['ghost_coords']
        patch_id = batch_samples[0]['patch_id']

        # Single spatial point cloud per batch: [N_pts, 3]
        spatial_coords = torch.cat([core_coords, ghost_coords], dim=0).float()
        n_pts = spatial_coords.shape[0]

        T_in = args.input_window_size
        T_out = args.output_window_size

        # --- Retrieve time values for this window ---
        # time_indices: (T_in + T_out,) — global indices into the dataset's time_values
        time_indices = batch_samples[0]['time_indices']  # same for all samples in batch
        time_values = getattr(args, '_time_values', None)

        if time_values is not None:
            # Clamp indices to valid range (safety)
            
            assert( ((time_indices > 0) & (time_indices < len(time_values))).all(), f"Time indices out of bounds (should be within range [0, {len(time_values)}])" )
            window_times = time_values[time_indices].float()  # (T_in + T_out,)
        else:
            # Fallback: use indices directly
            window_times = time_indices.float()

        # Normalize time using mean and std
        t_mean = time_values.mean()
        t_std = time_values.std()
        time_norm = (window_times - t_mean)/(t_std + 1e-10) # Avoid divide by zero

        input_time_norm = time_norm[:T_in]    # (T_in,)
        output_time_norm = time_norm[T_in:]   # (T_out,)

        # T_min and T_max for grid
        t_min = input_time_norm.min()
        t_max = input_time_norm.max()

        # --- Build 4D coordinates ---
        # Input coords: tile spatial coords across T_in time steps → (N_pts × T_in, 4)
        # For each time step t: [x_1,y_1,z_1,t; x_2,y_2,z_2,t; ...]
        spatial_tiled_in = spatial_coords.unsqueeze(1).expand(-1, T_in, -1)     # (N_pts, T_in, 3)
        time_tiled_in = input_time_norm.unsqueeze(0).expand(n_pts, -1).unsqueeze(-1)  # (N_pts, T_in, 1)
        input_coords_4d = torch.cat([spatial_tiled_in, time_tiled_in], dim=-1)  # (N_pts, T_in, 4)
        input_coords_4d = input_coords_4d.reshape(n_pts * T_in, 4)              # (N_pts×T_in, 4)

        # Output coords: tile spatial coords across T_out time steps → (N_pts × T_out, 4)
        spatial_tiled_out = spatial_coords.unsqueeze(1).expand(-1, T_out, -1)
        time_tiled_out = output_time_norm.unsqueeze(0).expand(n_pts, -1).unsqueeze(-1)
        output_coords_4d = torch.cat([spatial_tiled_out, time_tiled_out], dim=-1)
        output_coords_4d = output_coords_4d.reshape(n_pts * T_out, 4)

        # --- Build 4D latent query grid ---
        coords_min = torch.min(spatial_coords, dim=0).values
        coords_max = torch.max(spatial_coords, dim=0).values
        latent_query_arr = [
            torch.linspace(coords_min[i], coords_max[i], args.latent_query_dims[i], device=args.device)
            for i in range(coord_dim)
        ]
        # Temporal grid dimension: use full normalized time range [0, 1]
        n_t_grid = args.latent_query_dims[coord_dim]  # e.g., 10
        latent_query_arr.append(torch.linspace(t_min, t_max, n_t_grid, device=args.device))

        # Create meshgrid → (Nx, Ny, Nz, Nt, 4)
        latent_queries = torch.stack(
            torch.meshgrid(*latent_query_arr, indexing='ij'), dim=-1
        )

        # --- Assemble input features ---
        # For each sample, concat core+ghost, then assemble [obs, cur_forcings, fut_forcings]
        x_list, y_list = [], []
        for sample in batch_samples:
            n_core = sample['core_in_obs'].shape[0]
            n_ghost = sample['ghost_in_obs'].shape[0]

            # Observations: (N_pts, T_in, C_obs)
            in_obs = torch.cat([sample['core_in_obs'], sample['ghost_in_obs']], dim=0).float()

            if hasattr(args, 'forcings_required') and args.forcings_required:
                # Current forcings: (N_pts, T_in, C_forc)
                in_forc = torch.cat([sample['core_in_forcings'], sample['ghost_in_forcings']], dim=0).float()
                # Future forcings (paired by index): (N_pts, T_out, C_forc)
                out_forc = torch.cat([sample['core_out_forcings'], sample['ghost_out_forcings']], dim=0).float()
                # Assemble: [obs, cur_forcings, fut_forcings] → (N_pts, T_in, C_obs + C_forc + C_forc)
                sample_x = torch.cat([in_obs, in_forc, out_forc], dim=-1)  # (N_pts, T_in, 10)
            else:
                sample_x = in_obs  # (N_pts, T_in, C_obs)

            # Reshape to (N_pts × T_in, C_in) and add batch dim
            C_in = sample_x.shape[-1]
            sample_x = sample_x.reshape((n_core + n_ghost) * T_in, C_in)  # (N_pts×T_in, C_in)
            x_list.append(sample_x.unsqueeze(0))

            # Output targets: (N_pts, T_out, C_obs) → (N_pts × T_out, C_obs)
            out_obs = torch.cat([sample['core_out'], sample['ghost_out']], dim=0).float()
            C_obs = out_obs.shape[-1]
            sample_y = out_obs.reshape((n_core + n_ghost) * T_out, C_obs)
            y_list.append(sample_y.unsqueeze(0))

        x = torch.cat(x_list, dim=0)  # [B, N_pts × T_in, C_in]
        y = torch.cat(y_list, dim=0)  # [B, N_pts × T_out, C_obs]

        # Weights: tile across time steps for the output
        weights = batch_samples[0]['weights']
        if not isinstance(weights, torch.Tensor):
            weights = torch.from_numpy(weights)
        weights = weights.float()
        # Tile weights across T_out time steps: (N_core,) → (N_core × T_out,)
        core_len = len(core_coords)
        weights_tiled = weights.unsqueeze(1).expand(-1, T_out).reshape(-1)  # (N_core × T_out,)

        batch = {
            'patch_id': patch_id,
            'input_coords': input_coords_4d,        # [N_pts × T_in, 4]
            'output_coords': output_coords_4d,       # [N_pts × T_out, 4]
            'latent_queries': latent_queries,         # [Nx, Ny, Nz, Nt, 4]
            'x': x,                                   # [B, N_pts × T_in, C_in]
            'y': y,                                   # [B, N_pts × T_out, C_obs]
            'core_len': core_len,                     # number of core spatial points
            'n_pts': n_pts,                           # total spatial points (core + ghost)
            'T_in': T_in,
            'T_out': T_out,
            'weights': weights_tiled,                 # [N_core × T_out]
            'spatial_coords': spatial_coords,         # [N_pts, 3] for reference
        }
        return batch

    return collate_fn


def reshape_multi_col_predictions(predictions, output_window_size, n_target_cols):
    """
    Reshape predictions from the 4D FNO output format.

    The model outputs (B, N_pts × T_out, C_obs) which is then concatenated across
    batches to (N_samples, N_pts × T_out, C_obs). This function reshapes to
    separate the spatial and temporal dimensions.

    Args:
        predictions: Array of shape [N_samples, N_pts × T_out, n_target_cols]
            OR [N_samples, N_pts, output_window_size * n_target_cols] (legacy)
        output_window_size: Number of output timesteps
        n_target_cols: Number of target columns

    Returns:
        Array of shape [N_samples, N_pts, output_window_size, n_target_cols]
    """
    import numpy as np
    n_samples = predictions.shape[0]
    total_pts = predictions.shape[1]
    last_dim = predictions.shape[2]

    if last_dim == n_target_cols:
        # New format: (N_samples, N_pts × T_out, C_obs)
        # Total points = N_pts × T_out, so N_pts = total_pts / T_out
        n_points = total_pts // output_window_size
        reshaped = predictions.reshape(n_samples, n_points, output_window_size, n_target_cols)
    else:
        # Legacy format: (N_samples, N_pts, T_out × C_obs)
        n_points = total_pts
        if last_dim != output_window_size * n_target_cols:
            raise ValueError(
                f"Expected predictions shape [..., {output_window_size * n_target_cols}] "
                f"or [..., {n_target_cols}], got [..., {last_dim}]"
            )
        reshaped = predictions.reshape(n_samples, n_points, output_window_size, n_target_cols)

    return reshaped
