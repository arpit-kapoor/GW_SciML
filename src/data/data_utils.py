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


def calculate_obs_transform(raw_data_dir, target_obs_cols, representative_file='0000.csv'):
    """
    Calculate mean and std of output variables and create observation transform.

    Normalizes all listed columns for consistency. This is generic and can work
    with any set of observation columns.
    
    Args:
        raw_data_dir (str): Directory containing raw CSV files
        target_obs_cols (list): List of observation column names to normalize
        representative_file (str): Name of representative file to use for statistics
        
    Returns:
        Normalize: Transform object for observation normalization
    """
    # Read representative data file
    file_path = os.path.join(raw_data_dir, representative_file)
    df = pd.read_csv(file_path)

    # Calculate normalization statistics
    obs_mean = df[target_obs_cols].mean().values
    obs_std = df[target_obs_cols].std().values

    # Print normalization statistics for debugging
    print(f"Output mean: {obs_mean}")
    print(f"Output std: {obs_std}")

    # Define output transform
    obs_transform = Normalize(mean=obs_mean, std=obs_std)

    # Clean up memory
    del df
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
    # Create training dataset
    train_ds = dataset_class(
        data_path=patch_data_dir,
        dataset='train', 
        coord_transform=coord_transform, 
        obs_transform=obs_transform,
        input_window_size=kwargs.get('input_window_size', 10),
        output_window_size=kwargs.get('output_window_size', 10),
        target_col_indices=kwargs.get('target_col_indices', None),
        **{k: v for k, v in kwargs.items() if k not in ['input_window_size', 'output_window_size', 'target_col_indices']}
    )
    
    # Create validation dataset
    val_ds = dataset_class(
        data_path=patch_data_dir,
        dataset='val', 
        coord_transform=coord_transform, 
        obs_transform=obs_transform,
        input_window_size=kwargs.get('input_window_size', 10),
        output_window_size=kwargs.get('output_window_size', 10),
        target_col_indices=kwargs.get('target_col_indices', None),
        **{k: v for k, v in kwargs.items() if k not in ['input_window_size', 'output_window_size', 'target_col_indices']}
    )

    return train_ds, val_ds


def make_collate_fn(args, coord_dim=3):
    """
    Create a collate function that batches samples from the same patch.

    The sampler ensures a batch contains indices from a single `patch_id`.
    We build one point cloud per batch (core+ghost), a latent grid over its
    bounding box, and then stack input/output sequences along the batch dim.
    
    This is generic and works for any model that uses patch-based batching
    with core and ghost points.
    
    Args:
        args (argparse.Namespace): Argument namespace containing device and latent grid dimensions
        coord_dim (int): Coordinate dimensionality (default: 3 for 3D)
        
    Returns:
        function: Collate function for DataLoader
    """
    def collate_fn(batch_samples):
        """
        Collate function that combines samples into a batch.
        
        Args:
            batch_samples (list): List of sample dictionaries from the same patch
            
        Returns:
            dict: Batch dictionary with combined point cloud and sequences
        """
        # All samples in the batch come from the same patch (by sampler design)
        core_coords = batch_samples[0]['core_coords']
        ghost_coords = batch_samples[0]['ghost_coords']
        patch_id = batch_samples[0]['patch_id']  # Extract patch ID from first sample

        # Single point cloud per batch: [N_core+N_ghost, coord_dim]
        # Concatenate core and ghost points to form complete spatial domain
        point_coords = torch.concat([core_coords, ghost_coords], dim=0).float()

        # Create latent queries grid over the per-batch bounding box
        # This provides a regular grid for the FNO component
        coords_min = torch.min(point_coords, dim=0).values
        coords_max = torch.max(point_coords, dim=0).values
        latent_query_arr = [
            torch.linspace(coords_min[i], coords_max[i], args.latent_query_dims[i], device=args.device)
            for i in range(coord_dim)
        ]
        # Create meshgrid and stack to get [Qx, Qy, Qz, coord_dim] tensor
        latent_queries = torch.stack(torch.meshgrid(*latent_query_arr, indexing='ij'), dim=-1)

        # Build batched sequences: concat along points (dim=0), batch along dim=0
        x_list, y_list = [], []
        for sample in batch_samples:
            # Combine core and ghost inputs/outputs for each sample
            # Note: sequences are already concatenated across target columns in the dataset
            sample_input = torch.concat([sample['core_in'], sample['ghost_in']], dim=0).float().unsqueeze(0)
            sample_output = torch.concat([sample['core_out'], sample['ghost_out']], dim=0).float().unsqueeze(0)
            x_list.append(sample_input)
            y_list.append(sample_output)

        # Stack all sequences into batch tensors
        x = torch.cat(x_list, dim=0)  # [B, N_points, input_channels]
        y = torch.cat(y_list, dim=0)  # [B, N_points, output_channels]

        # Extract weights (same for all samples in the batch since they're from the same patch)
        weights = batch_samples[0]['weights']  # [N_points]
        if not isinstance(weights, torch.Tensor):
            weights = torch.from_numpy(weights)
        weights = weights.float()

        # Return batch dictionary
        batch = {
            'patch_id': patch_id,             # Patch identifier for tracking results
            'point_coords': point_coords,      # [N_points, coord_dim]
            'latent_queries': latent_queries,  # [Qx, Qy, Qz, coord_dim]
            'x': x,                           # [B, N_points, input_channels]
            'y': y,                           # [B, N_points, output_channels]
            'core_len': len(core_coords),     # Number of core points (for loss masking)
            'weights': weights,               # [N_points] - pre-computed variance-aware weights
        }
        return batch
    
    return collate_fn


def reshape_multi_col_predictions(predictions, output_window_size, n_target_cols):
    """
    Reshape concatenated predictions to separate target columns.
    
    The dataset concatenates data as: [t0_var0, t0_var1, t1_var0, t1_var1, t2_var0, t2_var1, ...]
    This is because _concat_sequence does: seq.reshape(n_points, -1) on [n_points, window_size, n_target_cols]
    which flattens in row-major order, interleaving timesteps and variables.
    
    Args:
        predictions: Array of shape [N_samples, N_points, output_window_size * n_target_cols]
        output_window_size: Number of timesteps
        n_target_cols: Number of target columns
        
    Returns:
        Array of shape [N_samples, N_points, output_window_size, n_target_cols]
    """
    import numpy as np
    n_samples, n_points, total_size = predictions.shape
    
    # Verify dimensions match
    if total_size != output_window_size * n_target_cols:
        raise ValueError(
            f"Expected predictions shape [..., {output_window_size * n_target_cols}], "
            f"got [..., {total_size}]"
        )
    
    # The data is stored as [t0_v0, t0_v1, t1_v0, t1_v1, ...] for each point
    # So we reshape to [N_samples, N_points, output_window_size, n_target_cols] directly
    # This naturally de-interleaves the timesteps and variables
    reshaped = predictions.reshape(n_samples, n_points, output_window_size, n_target_cols)
    return reshaped
