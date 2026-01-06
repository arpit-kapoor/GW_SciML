"""
Generate predictions using trained GINO multi-column model on the entire groundwater dataset.

This script extends generate_gino_predictions.py to support multi-column models that predict
multiple target variables simultaneously (e.g., mass_concentration and head together).

Key differences from single-column version:
- Uses GWPatchDatasetMultiCol for loading data
- Handles concatenated predictions across multiple target columns
- Separates predictions by target column for independent analysis
- Creates combined visualizations showing all target columns together
- Generates per-column statistics and metrics

Prediction tensor shapes:
- Raw predictions: [N_samples, N_points, output_window_size * n_target_cols]
- Reshaped predictions: [N_samples, N_points, output_window_size, n_target_cols]
- Per-column predictions: [N_samples, N_points, output_window_size] for each target

Usage:
    python generate_gino_predictions_multi_col.py \
        --model-path /path/to/model.pth \
        --results-dir /path/to/results \
        --target-cols mass_concentration head
"""

import argparse
import os
import datetime as dt
import pickle
import glob
import cv2

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.transform import Normalize
from src.data.patch_dataset_multi_col import GWPatchDatasetMultiCol
from src.data.batch_sampler import PatchBatchSampler
from src.models.neuralop.gino import GINO
from src.models.neuralop.losses import LpLoss


def setup_arguments():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description='Generate predictions using trained GINO multi-column model')
    
    # Model and data paths
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained GINO model state dict (.pth file)')
    parser.add_argument('--base-data-dir', type=str, 
                       default='/Users/arpitkapoor/Library/CloudStorage/OneDrive-UNSW/Shared/Projects/01_PhD/05_groundwater/data/FEFLOW/variable_density',
                       help='Base data directory')
    parser.add_argument('--raw-data-subdir', type=str, default='all',
                       help='Raw data subdirectory name')
    parser.add_argument('--patch-data-subdir', type=str, default='filter_patch',
                       help='Patch data subdirectory name')
    parser.add_argument('--results-dir', type=str, 
                       default='/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO_predictions',
                       help='Directory to save predictions and plots')
    
    # Model parameters (should match training configuration)
    parser.add_argument('--target-cols', type=str,
                       default='mass_concentration head',
                       help='Space-separated list of target observation column names')
    parser.add_argument('--input-window-size', type=int, default=10,
                       help='Number of time steps in each input sequence')
    parser.add_argument('--output-window-size', type=int, default=10,
                       help='Number of time steps in each output sequence')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for inference (cuda, cpu, or auto)')
    
    # Visualization options
    parser.add_argument('--create-3d-plots', action='store_true', default=False,
                       help='Create 3D scatter plots and videos (disabled by default to save time/storage)')
    
    args = parser.parse_args()
    
    # Construct full paths
    args.raw_data_dir = os.path.join(args.base_data_dir, args.raw_data_subdir)
    args.patch_data_dir = os.path.join(args.base_data_dir, args.patch_data_subdir)
    
    # Create results directory with timestamp
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Create a clean directory name using the full target column names
    # target_cols_str = '__'.join(args.target_cols.split(' '))
    args.results_dir = os.path.join(args.results_dir, f'run_{timestamp}')
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Configure model parameters (must match training configuration)
    args = configure_model_parameters(args)
    args = configure_target_col_indices(args)
    args = configure_device(args)
    
    print(f"Model path: {args.model_path}")
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Patch data directory: {args.patch_data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Target columns: {args.target_cols}")
    
    return args


def configure_model_parameters(args):
    """Configure model parameters to match training configuration."""
    # Number of target columns
    args.n_target_cols = len(args.target_cols)
    print(f"Number of target columns from args: {args.n_target_cols}")
    
    # These parameters must match the training configuration exactly
    args.coord_dim = 3
    args.gno_radius = 0.15
    args.in_gno_out_channels = args.input_window_size * args.n_target_cols
    args.in_gno_channel_mlp_layers = [32, 64, 32]
    args.fno_n_layers = 4
    args.fno_n_modes = (12, 12, 8)
    args.fno_hidden_channels = 64
    args.lifting_channels = 64
    args.out_gno_channel_mlp_layers = [32, 64, 32]
    args.projection_channel_ratio = 2
    args.out_channels = args.output_window_size * args.n_target_cols
    args.latent_query_dims = (32, 32, 24)
    
    # Print key parameters for debugging
    print(f"Input window size: {args.input_window_size}")
    print(f"Output window size: {args.output_window_size}")
    print(f"Number of target columns: {args.n_target_cols}")
    print(f"Input GNO out channels: {args.in_gno_out_channels}")
    print(f"Output channels: {args.out_channels}")
    return args


def configure_device(args):
    """Configure computation device."""
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Please use 'auto' or specify 'cpu'.")
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available. Please use 'auto' or specify 'cpu'.")
    print(f"Using device: {args.device}")
    return args


def configure_target_col_indices(args):
    """Configure target column indices from names."""
    names_to_idx = {
        'mass_concentration': 0,
        'head': 1,
        'pressure': 2
    }
    # Handle case where target_cols comes as a space-separated string
    if isinstance(args.target_cols, str):
        args.target_cols = args.target_cols.split()
    print(f"Processing target columns: {args.target_cols}")
    args.target_col_indices = [names_to_idx[col] for col in args.target_cols]
    print(f"Target column indices: {args.target_col_indices}")
    return args


def calculate_coord_transform(raw_data_dir):
    """Calculate coordinate normalization transform (same as training)."""
    df = pd.read_csv(os.path.join(raw_data_dir, '0000.csv'))
    coord_mean = df[['X', 'Y', 'Z']].mean().values
    coord_std = df[['X', 'Y', 'Z']].std().values
    print(f"Coordinate mean: {coord_mean}")
    print(f"Coordinate std: {coord_std}")
    coord_transform = Normalize(mean=coord_mean, std=coord_std)
    del df
    return coord_transform


def calculate_obs_transform(raw_data_dir, target_obs_cols=['mass_concentration', 'head', 'pressure']):
    """Calculate observation normalization transform (same as training)."""
    df = pd.read_csv(os.path.join(raw_data_dir, '0000.csv'))
    obs_mean = df[target_obs_cols].mean().values
    obs_std = df[target_obs_cols].std().values
    print(f"Output mean: {obs_mean}")
    print(f"Output std: {obs_std}")
    obs_transform = Normalize(mean=obs_mean, std=obs_std)
    del df
    return obs_transform


def create_patch_datasets(patch_data_dir, coord_transform, obs_transform, **kwargs):
    """Create train/val datasets for inference."""
    train_ds = GWPatchDatasetMultiCol(
        data_path=patch_data_dir,
        dataset='train', 
        coord_transform=coord_transform, 
        obs_transform=obs_transform,
        input_window_size=kwargs.get('input_window_size', 10),
        output_window_size=kwargs.get('output_window_size', 10),
        target_col_indices=kwargs.get('target_col_indices', None),
    )
    
    val_ds = GWPatchDatasetMultiCol(
        data_path=patch_data_dir,
        dataset='val', 
        coord_transform=coord_transform, 
        obs_transform=obs_transform,
        input_window_size=kwargs.get('input_window_size', 10),
        output_window_size=kwargs.get('output_window_size', 10),
        target_col_indices=kwargs.get('target_col_indices', None),
    )
    
    return train_ds, val_ds


def load_gino_model(model_path, args):
    """Load trained GINO model from checkpoint file."""
    print(f"\nLoading checkpoint from: {model_path}")
    
    # Load full checkpoint
    checkpoint = torch.load(model_path, map_location=args.device)
    
    # Extract saved configuration
    saved_args = checkpoint['args']
    print("\nSaved model configuration:")
    print(f"- FNO modes: {saved_args.fno_n_modes}")
    print(f"- FNO layers: {saved_args.fno_n_layers}")
    print(f"- Hidden channels: {saved_args.fno_hidden_channels}")
    print(f"- GNO radius: {saved_args.gno_radius}")
    print(f"- Latent dims: {saved_args.latent_query_dims}")
    print(f"- Target columns: {saved_args.target_cols}")
    print(f"- Number of target columns: {saved_args.n_target_cols}")
    
    # Initialize model with saved configuration
    model = GINO(
        # Input GNO configuration
        in_gno_coord_dim=saved_args.coord_dim,
        in_gno_radius=saved_args.gno_radius,
        in_gno_out_channels=saved_args.in_gno_out_channels,
        in_gno_channel_mlp_layers=saved_args.in_gno_channel_mlp_layers,
        
        # FNO configuration
        fno_n_layers=saved_args.fno_n_layers,
        fno_n_modes=saved_args.fno_n_modes,
        fno_hidden_channels=saved_args.fno_hidden_channels,
        lifting_channels=saved_args.lifting_channels,
        
        # Output GNO configuration
        out_gno_coord_dim=saved_args.coord_dim,
        out_gno_radius=saved_args.gno_radius,
        out_gno_channel_mlp_layers=saved_args.out_gno_channel_mlp_layers,
        projection_channel_ratio=saved_args.projection_channel_ratio,
        out_channels=saved_args.out_channels,
    ).to(args.device)
    
    # Load just the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Print training progress from checkpoint
    print(f"\nCheckpoint training progress:")
    print(f"- Epoch: {checkpoint['epoch'] + 1}")
    print(f"- Train loss: {checkpoint['train_losses'][-1]:.4f}")
    print(f"- Val loss: {checkpoint['val_losses'][-1]:.4f}")
    
    print(f"\nModel loaded successfully from: {model_path}")
    return model


def make_collate_fn(args):
    """Create collate function for batch processing (same as training)."""
    def collate_fn(batch_samples):
        # Get shared point cloud from first sample
        core_coords = batch_samples[0]['core_coords']
        ghost_coords = batch_samples[0]['ghost_coords']
        
        # Combine core and ghost points
        point_coords = torch.concat([core_coords, ghost_coords], dim=0).float()
        
        # Create latent queries grid over the per-batch bounding box
        coords_min = torch.min(point_coords, dim=0).values
        coords_max = torch.max(point_coords, dim=0).values
        latent_query_arr = [
            torch.linspace(coords_min[i], coords_max[i], args.latent_query_dims[i], device=args.device)
            for i in range(args.coord_dim)
        ]
        latent_queries = torch.stack(torch.meshgrid(*latent_query_arr, indexing='ij'), dim=-1)
        
        # Build batched sequences
        x_list, y_list = [], []
        for sample in batch_samples:
            sample_input = torch.concat([sample['core_in'], sample['ghost_in']], dim=0).float().unsqueeze(0)
            sample_output = torch.concat([sample['core_out'], sample['ghost_out']], dim=0).float().unsqueeze(0)
            x_list.append(sample_input)
            y_list.append(sample_output)
        
        x = torch.cat(x_list, dim=0)  # [B, N_points, input_window_size * n_target_cols]
        y = torch.cat(y_list, dim=0)  # [B, N_points, output_window_size * n_target_cols]
        
        return {
            'point_coords': point_coords,
            'latent_queries': latent_queries,
            'x': x,
            'y': y,
            'core_len': len(core_coords),
            'patch_id': sample['patch_id']
        }
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
    n_samples, n_points, total_size = predictions.shape
    # From [N_samples, N_points, T*C] to [N_samples, N_points, T, C]
    # where T = output_window_size and C = n_target_cols
    if total_size != output_window_size * n_target_cols:
        raise ValueError(f"Expected total size {output_window_size * n_target_cols} (output_window_size={output_window_size} * n_target_cols={n_target_cols}), but got {total_size}")
    
    # The data is stored as [t0_v0, t0_v1, t1_v0, t1_v1, ...] for each point
    # So we reshape to [N_samples, N_points, output_window_size, n_target_cols] directly
    # This naturally deinterleaves the timesteps and variables
    reshaped = predictions.reshape(n_samples, n_points, output_window_size, n_target_cols)
    return reshaped


def generate_predictions(model, dataset, args, dataset_name):
    """Generate predictions for a dataset."""
    print(f"Generating predictions for {dataset_name} dataset...")
    
    # Create sampler and data loader
    sampler = PatchBatchSampler(
        dataset, 
        batch_size=args.batch_size,
        shuffle_within_batches=False,
        shuffle_patches=False,
        seed=None
    )
    
    collate_fn = make_collate_fn(args)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    
    all_predictions = {}
    all_targets = {}
    all_coords = {}
    all_patch_metadata = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Processing {dataset_name} batches")):
            # Move to device
            point_coords = batch['point_coords'].to(args.device).float()
            latent_queries = batch['latent_queries'].to(args.device).float()
            x = batch['x'].to(args.device).float()
            y = batch['y'].to(args.device).float()
            patch_id = batch['patch_id']
            
            # Generate predictions
            outputs = model(
                input_geom=point_coords,
                latent_queries=latent_queries,
                x=x,
                output_queries=point_coords,
            )
            
            # Extract core points only (exclude ghost points)
            core_len = batch['core_len']
            core_output = outputs[:, :core_len].cpu().numpy()
            core_target = y[:, :core_len].cpu().numpy()
            core_coords = point_coords[:core_len].cpu().numpy()
            core_coords = np.repeat(np.expand_dims(core_coords, axis=0), core_output.shape[0], axis=0)
            
            # Store results
            if all_predictions.get(patch_id) is None:
                all_predictions[patch_id] = []
                all_targets[patch_id] = []
                all_coords[patch_id] = []
            
            all_predictions[patch_id].append(core_output)
            all_targets[patch_id].append(core_target)
            all_coords[patch_id].append(core_coords)
            
            # Store metadata for this batch
            batch_size = x.shape[0]
            for i in range(batch_size):
                all_patch_metadata.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'dataset': dataset_name,
                    'core_len': core_len
                })
    
    # Concatenate all results
    for patch_id in all_predictions.keys():
        all_predictions[patch_id] = np.concatenate(all_predictions[patch_id], axis=0)
        all_targets[patch_id] = np.concatenate(all_targets[patch_id], axis=0)
        all_coords[patch_id] = np.concatenate(all_coords[patch_id], axis=0)
    
    predictions = np.concatenate(list(all_predictions.values()), axis=1)
    targets = np.concatenate(list(all_targets.values()), axis=1)
    coords = np.concatenate(list(all_coords.values()), axis=1)
    
    print(f"{dataset_name} predictions shape (concatenated): {predictions.shape}")
    print(f"{dataset_name} targets shape (concatenated): {targets.shape}")
    print(f"{dataset_name} coords shape: {coords.shape}")
    
    # Get the number of target columns from the model's configuration
    n_target_cols = len(args.target_cols)  # Use the actual number of target columns
    print(f"Using {n_target_cols} target columns for reshaping")
    
    # Reshape to separate target columns
    predictions_reshaped = reshape_multi_col_predictions(
        predictions, args.output_window_size, n_target_cols
    )
    targets_reshaped = reshape_multi_col_predictions(
        targets, args.output_window_size, n_target_cols
    )
    
    print(f"{dataset_name} predictions shape (reshaped): {predictions_reshaped.shape}")
    print(f"{dataset_name} targets shape (reshaped): {targets_reshaped.shape}")
    
    return {
        'predictions': predictions_reshaped,  # [N_samples, N_points, output_window_size, n_target_cols]
        'targets': targets_reshaped,
        'coords': coords,
        'metadata': all_patch_metadata
    }


def save_results(results_dict, args, obs_transform, coord_transform):
    """Save predictions and metadata to files."""
    print("Saving results...")
    
    # Denormalize observations (predictions and targets)
    train_predictions_denorm = denormalize_observations(
        results_dict['train']['predictions'], obs_transform, args.target_col_indices
    )
    train_targets_denorm = denormalize_observations(
        results_dict['train']['targets'], obs_transform, args.target_col_indices
    )
    val_predictions_denorm = denormalize_observations(
        results_dict['val']['predictions'], obs_transform, args.target_col_indices
    )
    val_targets_denorm = denormalize_observations(
        results_dict['val']['targets'], obs_transform, args.target_col_indices
    )
    
    # Denormalize coordinates
    # coords shape: [N_samples, N_points, 3]
    train_coords = results_dict['train']['coords']
    val_coords = results_dict['val']['coords']
    
    # Extract mean and std for coordinates
    coord_mean = coord_transform.mean
    coord_std = coord_transform.std
    
    # Convert to numpy if they are tensors
    if isinstance(coord_mean, torch.Tensor):
        coord_mean = coord_mean.cpu().numpy()
    if isinstance(coord_std, torch.Tensor):
        coord_std = coord_std.cpu().numpy()
    
    # Reshape for broadcasting: [1, 1, 3]
    coord_mean = coord_mean.reshape(1, 1, -1)
    coord_std = coord_std.reshape(1, 1, -1)
    
    # Denormalize: original = normalized * std + mean
    train_coords_denorm = train_coords * coord_std + coord_mean
    val_coords_denorm = val_coords * coord_std + coord_mean
    
    # Save train results (denormalized)
    np.save(os.path.join(args.results_dir, 'train_predictions.npy'), train_predictions_denorm)
    np.save(os.path.join(args.results_dir, 'train_targets.npy'), train_targets_denorm)
    
    # Save validation results (denormalized)
    np.save(os.path.join(args.results_dir, 'val_predictions.npy'), val_predictions_denorm)
    np.save(os.path.join(args.results_dir, 'val_targets.npy'), val_targets_denorm)
    
    # Save coordinates (denormalized)
    with open(os.path.join(args.results_dir, 'train_coords.pkl'), 'wb') as f:
        pickle.dump(train_coords_denorm, f)
    with open(os.path.join(args.results_dir, 'val_coords.pkl'), 'wb') as f:
        pickle.dump(val_coords_denorm, f)
    
    # Save metadata
    with open(os.path.join(args.results_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({
            'train_metadata': results_dict['train']['metadata'],
            'val_metadata': results_dict['val']['metadata'],
            'args': vars(args)
        }, f)
    
    print(f"Results saved to: {args.results_dir} (denormalized)")


def denormalize_observations(normalized_data, obs_transform, target_col_indices):
    """
    Denormalize observations back to original scale.
    
    Args:
        normalized_data: Array of shape [N_samples, N_points, output_window_size, n_target_cols]
        obs_transform: Normalize transform object with mean and std
        target_col_indices: List of indices for target columns
        
    Returns:
        Array in original scale with same shape
    """
    # Extract mean and std for target columns
    mean = obs_transform.mean[target_col_indices]
    std = obs_transform.std[target_col_indices]
    
    # Convert to numpy if they are tensors
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(std, torch.Tensor):
        std = std.cpu().numpy()
    
    print(f"Denormalizing with mean: {mean}, std: {std}")
    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    print(f"Normalized data shape: {normalized_data.shape}")
    print(f"Normalized data sample values: min={normalized_data.min():.4f}, max={normalized_data.max():.4f}, mean={normalized_data.mean():.4f}")
    
    # The Normalize class uses broadcasting: (sample - mean) / std
    # So denormalization is: sample * std + mean
    # For shape [N_samples, N_points, output_window_size, n_target_cols] and mean/std of shape [n_target_cols],
    # NumPy will broadcast from the right, applying each mean/std to its corresponding column
    
    # No need to reshape - NumPy/PyTorch broadcasting automatically aligns from the rightmost dimension
    denormalized = normalized_data * std + mean
    
    print(f"Denormalized sample values: min={denormalized.min():.4f}, max={denormalized.max():.4f}, mean={denormalized.mean():.4f}")
    
    return denormalized


def compute_kge(simulations, observations):
    """
    Compute Kling-Gupta Efficiency (KGE).
    
    KGE = 1 - sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    where:
    - r: Pearson correlation coefficient
    - alpha: Relative variability (std_sim / std_obs)
    - beta: Bias ratio (mean_sim / mean_obs)
    
    Args:
        simulations: Predicted values (flattened array)
        observations: Target/observed values (flattened array)
        
    Returns:
        Dictionary with KGE and its components (r, alpha, beta)
    """
    epsilon = 1e-10
    
    # Mean values
    sim_mean = np.mean(simulations)
    obs_mean = np.mean(observations)
    
    # Standard deviations
    sim_std = np.std(simulations)
    obs_std = np.std(observations)
    
    # Pearson correlation coefficient
    if obs_std < epsilon or sim_std < epsilon:
        r = 0.0
    else:
        r = np.corrcoef(observations, simulations)[0, 1]
        if np.isnan(r):
            r = 0.0
    
    # Alpha (variability ratio)
    alpha = sim_std / (obs_std + epsilon)
    
    # Beta (bias ratio)
    beta = sim_mean / (obs_mean + epsilon)
    
    # KGE
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return {
        'kge': kge,
        'r': r,
        'alpha': alpha,
        'beta': beta
    }


def compute_metrics(results_dict, args, obs_transform):
    """
    Compute relative L2 error, R2 score, and KGE for each target column on train and val sets.
    
    All metrics are computed on denormalized (original scale) data.
    
    Args:
        results_dict: Dictionary containing train and val predictions and targets
        args: Argument namespace
        obs_transform: Normalize transform object for denormalization
        
    Returns:
        Dictionary containing metrics for each dataset and target column
    """
    from sklearn.metrics import r2_score
    
    metrics = {}
    
    for dataset_name in ['train', 'val']:
        # Get normalized data
        predictions_norm = results_dict[dataset_name]['predictions']  # [N_samples, N_points, output_window_size, n_target_cols]
        targets_norm = results_dict[dataset_name]['targets']
        
        # Denormalize to original scale
        predictions = denormalize_observations(predictions_norm, obs_transform, args.target_col_indices)
        targets = denormalize_observations(targets_norm, obs_transform, args.target_col_indices)
        
        metrics[dataset_name] = {}
        
        # Compute metrics for each target column
        for col_idx, col_name in enumerate(args.target_cols):
            print(f"Computing metrics for {col_name} ({dataset_name})...")
            col_predictions = predictions[:, :, :, col_idx]  # [N_samples, N_points, output_window_size]
            col_targets = targets[:, :, :, col_idx]
            
            # Flatten for metric computation
            col_predictions_flat = col_predictions.flatten()
            col_targets_flat = col_targets.flatten()
            
            # 1. Relative L2 Error (using numpy implementation)
            # Relative L2 = ||pred - target||_2 / ||target||_2
            diff = col_predictions_flat - col_targets_flat
            l2_error = np.linalg.norm(diff)
            l2_norm_target = np.linalg.norm(col_targets_flat)
            rel_l2_error = l2_error / (l2_norm_target + 1e-10)
            
            # 2. R² Score
            r2 = r2_score(col_targets_flat, col_predictions_flat)
            
            # 3. Kling-Gupta Efficiency (KGE)
            kge_results = compute_kge(col_predictions_flat, col_targets_flat)
            
            metrics[dataset_name][col_name] = {
                'rel_l2_error': rel_l2_error,
                'r2_score': r2,
                'kge': kge_results['kge'],
                'kge_r': kge_results['r'],
                'kge_alpha': kge_results['alpha'],
                'kge_beta': kge_results['beta']
            }
    
    return metrics


def save_metrics(metrics, args):
    """
    Save computed metrics to a text file.
    
    Args:
        metrics: Dictionary containing metrics for each dataset and target column
        args: Argument namespace
    """
    metrics_file = os.path.join(args.results_dir, 'metrics.txt')
    
    with open(metrics_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Model Performance Metrics (Relative & Dimensionless)\n")
        f.write("All metrics computed on denormalized (original scale) data\n")
        f.write("=" * 80 + "\n\n")
        
        for dataset_name in ['train', 'val']:
            f.write(f"\n{dataset_name.upper()} SET\n")
            f.write("-" * 80 + "\n")
            
            for col_name in args.target_cols:
                col_metrics = metrics[dataset_name][col_name]
                
                f.write(f"\n{col_name}:\n")
                f.write(f"  Relative L2 Error:  {col_metrics['rel_l2_error']:.6f}  (0.0 is perfect)\n")
                f.write(f"  R² Score:           {col_metrics['r2_score']:.6f}  (1.0 is perfect)\n")
                f.write(f"\n")
                f.write(f"  Kling-Gupta Efficiency:\n")
                f.write(f"    KGE:              {col_metrics['kge']:.6f}  (1.0 is perfect)\n")
                f.write(f"      - r (correlation):     {col_metrics['kge_r']:.4f}  (1.0 is perfect)\n")
                f.write(f"      - alpha (variability): {col_metrics['kge_alpha']:.4f}  (1.0 is perfect)\n")
                f.write(f"      - beta (bias ratio):   {col_metrics['kge_beta']:.4f}  (1.0 is perfect)\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("\nMetric Interpretation:\n")
        f.write("  - Relative L2 Error: Dimensionless error normalized by target magnitude\n")
        f.write("  - R² Score: Coefficient of determination (fraction of variance explained)\n")
        f.write("  - KGE: Combines correlation, variability, and bias (preferred in hydrogeology)\n")
        f.write("=" * 80 + "\n")
    
    print(f"Metrics saved to: {metrics_file}")


def create_visualizations(results_dict, args, obs_transform):
    """Create visualizations comparing predictions vs observations for multi-column predictions."""
    print("Creating visualizations...")
    
    # Create visualizations for both train and val datasets
    for dataset_name in ['train', 'val']:
        print(f"\n{'='*60}")
        print(f"Creating visualizations for {dataset_name} dataset...")
        print(f"{'='*60}")
        
        # Create dataset-specific directory
        dataset_dir = os.path.join(args.results_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Get data for this dataset (in normalized scale)
        predictions_norm = results_dict[dataset_name]['predictions']  # [N_samples, N_points, output_window_size, n_target_cols]
        targets_norm = results_dict[dataset_name]['targets']
        coords_data = results_dict[dataset_name]['coords']
        
        # Denormalize to original scale for visualization
        predictions = denormalize_observations(predictions_norm, obs_transform, args.target_col_indices)
        targets = denormalize_observations(targets_norm, obs_transform, args.target_col_indices)
        
        print(f"Denormalized {dataset_name} predictions to original scale")
        print(f"  Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"  Targets range: [{targets.min():.4f}, {targets.max():.4f}]")
        
        # Create visualizations for this dataset
        create_dataset_visualizations(predictions, targets, coords_data, dataset_name, dataset_dir, args)
    
    print(f"\nAll visualizations saved to: {args.results_dir}")


def create_dataset_visualizations(predictions, targets, coords_data, dataset_name, dataset_dir, args):
    """Create visualizations for a specific dataset (train or val)."""
    n_ts, n_nodes, output_window_size, n_target_cols = predictions.shape

    # Identify node indices with least to highest variance in each target column
    # Compute variance for each node for each target column
    node_variance = np.var(targets[:, :, 0, :], axis=0)
    print(f"Node variance shape: {node_variance.shape}")

    # # node_variance has shape [n_nodes, n_target_cols]
    # # Compute percentiles along the node dimension
    # p5 = np.percentile(node_variance, 5, axis=0)
    # p25 = np.percentile(node_variance, 25, axis=0)
    # p50 = np.percentile(node_variance, 50, axis=0)
    # p75 = np.percentile(node_variance, 75, axis=0)
    # p95 = np.percentile(node_variance, 95, axis=0)

    # # Compute absolute differences from these percentile values
    # diff_5 = np.abs(node_variance - p5)
    # diff_25 = np.abs(node_variance - p25)
    # diff_50 = np.abs(node_variance - p50)
    # diff_75 = np.abs(node_variance - p75)
    # diff_95 = np.abs(node_variance - p95)

    # # Find indices (along nodes) that are closest to each percentile per target column
    # idx_5 = np.argmin(diff_5, axis=0)
    # idx_25 = np.argmin(diff_25, axis=0)
    # idx_50 = np.argmin(diff_50, axis=0)
    # idx_75 = np.argmin(diff_75, axis=0)
    # idx_95 = np.argmin(diff_95, axis=0)

    # selected_node_idx = np.stack([idx_5, idx_25, idx_50, idx_75, idx_95], axis=0)

    # Alternatively, select nodes at specific percentiles directly
    p95 = np.percentile(node_variance, 95, axis=0)
    p99 = np.percentile(node_variance, 99, axis=0)
    selected_node_idx = []
    for col_idx, col_name in enumerate(args.target_cols):
        
        print(f"{col_name} - 95th percentile variance: {p95[col_idx]:.6f}")
        print(f"{col_name} - 99th percentile variance: {p99[col_idx]:.6f}")

        # Identify nodes above 99th percentile for this column
        node_idx_in_range = np.where(node_variance[:, col_idx] > p99[col_idx])[0]

        # node_idx_in_range = np.where((node_variance[:, col_idx] >= p95[col_idx]) & (node_variance[:, col_idx] <= p99[col_idx]))[0]

        # print(f"{col_name} - Number of nodes between 95th and 99th percentile: {len(node_idx_in_range)}")

        # Randomly select 5 nodes from those above 95th percentile for each target column
        np.random.seed(42)  # For reproducibility
        # select first 3 and last 2 nodes for consistency
        # selected_node_idx.append(np.concatenate((node_idx_in_range[:3], node_idx_in_range[-2:])).reshape(5, 1))
        selected_node_idx.append(np.random.choice(node_idx_in_range, size=(5, 1), replace=False)) 
        
        # Take first 5 for consistency
    
    selected_node_idx = np.hstack(selected_node_idx)  # Shape: [5, n_target_cols]

    print(f"Selected nodes shape: {selected_node_idx.shape}")

    # # Create scatter plots for first timestep of each target column (stacked as rows)
    # print("Creating combined first-timestep scatter plots...")
    # create_first_timestep_scatter_plots(predictions, targets, dataset_dir, args)
    
    # Create combined 3D scatter plots for all target columns (if enabled)
    if args.create_3d_plots:
        print(f"Creating combined 3D scatter plots for {dataset_name}...")
        create_combined_3d_scatter_plots(predictions, targets, coords_data, dataset_dir, args)
        
        # Create video from combined 3D scatter plots
        create_video_from_combined_scatter_plots(dataset_dir, args)
    else:
        print(f"Skipping 3D scatter plots for {dataset_name} (disabled via --create-3d-plots flag)")
    
    # Create per-column analyses
    for col_idx, col_name in enumerate(args.target_cols):
        print(f"Creating visualizations for {col_name} ({dataset_name})...")
        
        # Extract predictions and targets for this column
        col_predictions = predictions[:, :, :, col_idx]  # [N_samples, N_points, output_window_size]
        col_targets = targets[:, :, :, col_idx]
        
        # Create column-specific directory
        col_dir = os.path.join(dataset_dir, col_name)
        os.makedirs(col_dir, exist_ok=True)
        
        # Create scatter plots at different timesteps
        create_timestep_scatter_plots(col_predictions, col_targets, col_name, col_dir, selected_node_idx[:, col_idx], args)
        
        # Create time series plots
        create_time_series_plots(col_predictions, col_targets, col_name, col_dir, selected_node_idx[:, col_idx], args)
        
        # Create error analysis
        create_error_analysis(col_predictions, col_targets, col_name, col_dir, args)
        
        # Print error statistics
        errors = col_predictions - col_targets
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        print(f"{col_name} ({dataset_name}) - Overall MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    print(f"Visualizations for {dataset_name} saved to: {dataset_dir}")


def create_first_timestep_scatter_plots(predictions, targets, args):
    """
    Create combined scatter plots showing first timestep of all target columns.
    Each target column is shown as a row in the figure.
    """
    n_samples_to_plot = min(5, predictions.shape[0])
    n_target_cols = predictions.shape[3]
    
    fig, axes = plt.subplots(n_samples_to_plot, n_target_cols, 
                            figsize=(6*n_target_cols, 4*n_samples_to_plot))
    
    # Handle single sample or single column case
    if n_samples_to_plot == 1 and n_target_cols == 1:
        axes = np.array([[axes]])
    elif n_samples_to_plot == 1:
        axes = axes.reshape(1, -1)
    elif n_target_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for sample_idx in range(n_samples_to_plot):
        for col_idx, col_name in enumerate(args.target_cols):
            ax = axes[sample_idx, col_idx]
            
            # Get predictions and targets for first timestep (t=0) of this column
            pred_t0 = predictions[sample_idx, :, 0, col_idx]  # [N_points]
            target_t0 = targets[sample_idx, :, 0, col_idx]
            
            # Create scatter plot
            ax.scatter(target_t0, pred_t0, alpha=0.6, s=1)
            
            # Add perfect prediction line
            min_val = min(target_t0.min(), pred_t0.min())
            max_val = max(target_t0.max(), pred_t0.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect')
            
            # Calculate correlation
            correlation = np.corrcoef(target_t0, pred_t0)[0, 1]
            
            ax.set_xlabel('Observed', fontsize=20)
            ax.set_ylabel('Predicted', fontsize=20)
            ax.set_title(f'{col_name}\nSample {sample_idx+1}, t=1\nCorr: {correlation:.3f}', fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.grid(True, alpha=0.3)
            
            if sample_idx == 0 and col_idx == 0:
                ax.legend(fontsize=18)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'first_timestep_all_columns.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_timestep_scatter_plots(predictions, targets, col_name, col_dir, selected_node_idx, args):
    """Create scatter plots at different timesteps for a single column."""
    output_window_size = predictions.shape[2]
    
    # Adapt timesteps to plot based on output window size
    if output_window_size == 1:
        timesteps_to_plot = [0]  # Only first timestep
    elif output_window_size == 2:
        timesteps_to_plot = [0, 1]  # First two timesteps
    elif output_window_size == 3:
        timesteps_to_plot = [0, 1, 2]  # First three timesteps
    else:
        timesteps_to_plot = [0, output_window_size//2, output_window_size-1]  # First, middle, last
    
    n_samples_to_plot = selected_node_idx.shape[0]
    sample_indices = selected_node_idx
    
    fig, axes = plt.subplots(n_samples_to_plot, len(timesteps_to_plot), 
                            figsize=(7*len(timesteps_to_plot), 5*n_samples_to_plot))
    
    # Ensure axes is always 2D for consistent indexing
    if n_samples_to_plot == 1 and len(timesteps_to_plot) == 1:
        axes = np.array([[axes]])
    elif n_samples_to_plot == 1:
        axes = axes.reshape(1, -1)
    elif len(timesteps_to_plot) == 1:
        axes = axes.reshape(-1, 1)
    
    for axes_idx, sample_idx in enumerate(sample_indices):
        for plot_idx, timestep in enumerate(timesteps_to_plot):
            ax = axes[axes_idx, plot_idx]
            
            pred_t = predictions[:, sample_idx, timestep]
            target_t = targets[:, sample_idx, timestep]
            
            ax.scatter(target_t, pred_t, alpha=0.6, s=5)
            
            min_val = min(target_t.min(), pred_t.min())
            max_val = max(target_t.max(), pred_t.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2.5, label='Perfect')
            
            correlation = np.corrcoef(target_t, pred_t)[0, 1]
            
            ax.set_xlabel('Observed', fontsize=24)
            ax.set_ylabel('Predicted', fontsize=24)
            ax.set_title(f'Node {sample_idx}, Timestep {timestep+1}\nCorr: {correlation:.3f}', fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.grid(True, alpha=0.3)
            
            if axes_idx == 0 and plot_idx == 0:
                ax.legend(fontsize=22)
    
    plt.tight_layout()
    plt.savefig(os.path.join(col_dir, f'{col_name}_pred_vs_obs_scatter.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_time_series_plots(predictions, targets, col_name, col_dir, selected_node_idx, args):
    """
    Create time series plots for a single column.
    
    Plots all timesteps in the prediction horizon for each selected sample.
    Saves individual PNG files for each selected node/sample.
    
    Args:
        predictions: Array of shape [N_samples, N_points, output_window_size]
        targets: Array of shape [N_samples, N_points, output_window_size]
        col_name: Name of the target column
        col_dir: Directory to save plots
        selected_node_idx: Array of selected node indices
        args: Argument namespace
    """
    n_samples_to_plot = selected_node_idx.shape[0]
    sample_indices = selected_node_idx
    output_window_size = predictions.shape[2]
    
    # Create directory for time series plots
    timeseries_dir = os.path.join(col_dir, 'time_series_plots')
    os.makedirs(timeseries_dir, exist_ok=True)
    
    # Iterate over all timesteps in the prediction horizon
    for timestep_idx in range(output_window_size):
        # Create a figure with subplots for all selected samples
        fig, axes = plt.subplots(n_samples_to_plot, 1, figsize=(12, 5*n_samples_to_plot))
        
        if n_samples_to_plot == 1:
            axes = [axes]
        
        for axes_idx, sample_idx in enumerate(sample_indices):
            ax = axes[axes_idx]
            
            # Get values at this specific timestep for this sample/node across all samples
            # predictions shape: [N_samples, N_points, output_window_size]
            pred_at_timestep = predictions[:, sample_idx, timestep_idx]  # [N_samples]
            target_at_timestep = targets[:, sample_idx, timestep_idx]  # [N_samples]

            # Create time array
            timesteps = np.arange(len(pred_at_timestep))
            ax.plot(timesteps, pred_at_timestep, 'r--', label='Predicted', linewidth=2)
            ax.plot(timesteps, target_at_timestep, 'b-', label='Observed', linewidth=2)
            ax.set_xlabel('Simulation Time-step', fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.legend(fontsize=18)
            
            # Calculate MAE at this timestep
            mae = np.mean(np.abs(pred_at_timestep - target_at_timestep))
            
            ax.set_ylabel(f'{col_name}', fontsize=20)
            ax.set_title(f'Node {sample_idx} - Timestep {timestep_idx+1}/{output_window_size}\n' + 
                        f'MAE: {mae:.4f}', fontsize=20)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save individual timestep plot
        timestep_filename = f'{col_name}_pred_vs_obs_lineplot_t{timestep_idx+1:02d}.png'
        plt.savefig(os.path.join(timeseries_dir, timestep_filename), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {output_window_size} time series plots to: {timeseries_dir}")


def create_error_analysis(predictions, targets, col_name, col_dir, args):
    """Create error analysis plots for a single column."""
    errors = predictions - targets
    mae_by_timestep = np.mean(np.abs(errors), axis=(1, 2))
    rmse_by_timestep = np.sqrt(np.mean(errors**2, axis=(1, 2)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    timesteps = np.arange(len(mae_by_timestep))
    
    ax1.plot(timesteps, mae_by_timestep, 'b-', marker='o', linewidth=2)
    ax1.set_xlabel('Timestep', fontsize=20)
    ax1.set_ylabel('Mean Absolute Error', fontsize=20)
    ax1.set_title(f'{col_name} - MAE by Timestep', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(timesteps, rmse_by_timestep, 'r-', marker='s', linewidth=2)
    ax2.set_xlabel('Timestep', fontsize=20)
    ax2.set_ylabel('Root Mean Square Error', fontsize=20)
    ax2.set_title(f'{col_name} - RMSE by Timestep', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(col_dir, 'error_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_combined_3d_scatter_plots(predictions, targets, coords_data, dataset_dir, args):
    """
    Create combined 3D scatter plots showing all target columns together.
    
    Layout:
    - Rows: Target columns
    - Columns: Observations, Predictions, Error
    
    Args:
        predictions: Array of shape [N_samples, N_points, output_window_size, n_target_cols]
        targets: Array of shape [N_samples, N_points, output_window_size, n_target_cols]
        coords_data: Array of shape [N_samples, N_points, 3]
        dataset_dir: Directory to save plots (train or val specific)
        args: Argument namespace
    """
    n_samples = predictions.shape[0]
    n_target_cols = predictions.shape[3]
    
    # Create directory for combined plots
    combined_plots_dir = os.path.join(dataset_dir, 'combined_3d_scatter_plots')
    os.makedirs(combined_plots_dir, exist_ok=True)
    
    print(f"Creating combined 3D scatter plots for {n_samples} samples with {n_target_cols} target columns...")
    
    for sample_idx in range(n_samples):
        coords = coords_data[sample_idx]
        
        # Create figure with subplots: rows = target columns, cols = obs/pred/error
        fig = plt.figure(figsize=(18, 6 * n_target_cols))
        
        for col_idx, col_name in enumerate(args.target_cols):
            # Get predictions and targets for first timestep (t=0) of this column
            pred_first_timestep = predictions[sample_idx, :len(coords), 0, col_idx]
            target_first_timestep = targets[sample_idx, :len(coords), 0, col_idx]
            
            # Calculate vmin and vmax based on observations for consistent color scale
            vmin = np.min(target_first_timestep)
            vmax = np.max(target_first_timestep)
            
            # Subplot for observations
            ax1 = fig.add_subplot(n_target_cols, 3, col_idx * 3 + 1, projection='3d')
            scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                                  c=target_first_timestep, cmap='viridis', s=5, alpha=0.7,
                                  vmin=vmin, vmax=vmax)
            ax1.set_title(f'{col_name} - Observations\nSample {sample_idx+1}, t=1', fontsize=10)
            ax1.set_xlabel('X', fontsize=8)
            ax1.set_ylabel('Y', fontsize=8)
            ax1.set_zlabel('Z', fontsize=8)
            plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=20, pad=0.1)
            
            # Subplot for predictions
            ax2 = fig.add_subplot(n_target_cols, 3, col_idx * 3 + 2, projection='3d')
            scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                                  c=pred_first_timestep, cmap='viridis', s=5, alpha=0.7,
                                  vmin=vmin, vmax=vmax)
            ax2.set_title(f'{col_name} - Predictions\nSample {sample_idx+1}, t=1', fontsize=10)
            ax2.set_xlabel('X', fontsize=8)
            ax2.set_ylabel('Y', fontsize=8)
            ax2.set_zlabel('Z', fontsize=8)
            plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=20, pad=0.1)
            
            # Calculate error
            error = target_first_timestep - pred_first_timestep
            
            # Subplot for error
            ax3 = fig.add_subplot(n_target_cols, 3, col_idx * 3 + 3, projection='3d')
            error_range = max(abs(error.min()), abs(error.max()))
            if error_range == 0:
                error_range = 1.0  # Avoid division by zero
            scatter3 = ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                                  c=error, cmap='RdBu_r', s=5, alpha=0.7,
                                  vmin=-error_range, vmax=error_range)
            ax3.set_title(f'{col_name} - Error (Obs - Pred)\nSample {sample_idx+1}, t=1', fontsize=10)
            ax3.set_xlabel('X', fontsize=8)
            ax3.set_ylabel('Y', fontsize=8)
            ax3.set_zlabel('Z', fontsize=8)
            plt.colorbar(scatter3, ax=ax3, shrink=0.5, aspect=20, pad=0.1)
        
        plt.tight_layout()
        
        # Save combined plot
        sample_filename = f'combined_3d_scatter_sample_{sample_idx+1:03d}.png'
        plt.savefig(os.path.join(combined_plots_dir, sample_filename), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        if (sample_idx + 1) % 10 == 0:
            print(f"  Created {sample_idx + 1}/{n_samples} combined plots")
    
    print(f"Combined 3D scatter plots saved to: {combined_plots_dir}")


def create_video_from_combined_scatter_plots(dataset_dir, args):
    """Create a single video from combined 3D scatter plot images."""
    print("Creating video from combined 3D scatter plots...")
    
    # Path to the combined scatter plots directory
    combined_plots_dir = os.path.join(dataset_dir, 'combined_3d_scatter_plots')
    
    # Check if directory exists
    if not os.path.exists(combined_plots_dir):
        print(f"Warning: Combined scatter plots directory not found: {combined_plots_dir}")
        return
    
    # Get all PNG files and sort them
    image_pattern = os.path.join(combined_plots_dir, 'combined_3d_scatter_sample_*.png')
    image_files = glob.glob(image_pattern)
    
    if not image_files:
        print(f"Warning: No combined scatter plot images found in {combined_plots_dir}")
        return
    
    # Sort files by filename to ensure proper order
    image_files.sort()
    
    print(f"Found {len(image_files)} combined scatter plot images")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape
    
    # Define video parameters
    fps = 10
    video_filename = os.path.join(dataset_dir, 'combined_3d_scatter_plots_video.mp4')
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    # Write each image to the video
    for image_file in tqdm(image_files, desc="Creating combined video frames"):
        frame = cv2.imread(image_file)
        video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()
    
    print(f"Combined video saved to: {video_filename}")
    print(f"Video details: {len(image_files)} frames at {fps} fps")


def create_3d_scatter_plots(predictions, targets, coords_data, col_name, col_dir, args):
    """Create 3D scatter plots for a single column."""
    n_samples_to_plot_3d = predictions.shape[0]
    
    scatter_plots_dir = os.path.join(col_dir, '3d_scatter_plots')
    os.makedirs(scatter_plots_dir, exist_ok=True)
    
    for sample_idx in range(n_samples_to_plot_3d):
        coords = coords_data[sample_idx]
        
        # Get predictions and targets for first timestep (t=0)
        pred_first_timestep = predictions[sample_idx, :len(coords), 0]
        target_first_timestep = targets[sample_idx, :len(coords), 0]
        
        # Calculate vmin and vmax based on observations
        vmin = np.min(target_first_timestep)
        vmax = np.max(target_first_timestep)
        
        # Create figure with three subplots
        fig = plt.figure(figsize=(20, 6))
        
        # Subplot for observations
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                              c=target_first_timestep, cmap='viridis', s=5, alpha=0.7,
                              vmin=vmin, vmax=vmax)
        ax1.set_title(f'{col_name} - Observations\nSample {sample_idx+1}, First Timestep')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=20, 
                    label=f'{col_name} (observed)')
        
        # Subplot for predictions
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                              c=pred_first_timestep, cmap='viridis', s=5, alpha=0.7,
                              vmin=vmin, vmax=vmax)
        ax2.set_title(f'{col_name} - Predictions\nSample {sample_idx+1}, First Timestep')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=20, 
                    label=f'{col_name} (predicted)')
        
        # Calculate error
        error = target_first_timestep - pred_first_timestep
        
        # Subplot for error
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        error_range = max(abs(error.min()), abs(error.max()))
        scatter3 = ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                              c=error, cmap='RdBu_r', s=5, alpha=0.7,
                              vmin=-error_range, vmax=error_range)
        ax3.set_title(f'{col_name} - Error (Obs - Pred)\nSample {sample_idx+1}, First Timestep')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        plt.colorbar(scatter3, ax=ax3, shrink=0.5, aspect=20, 
                    label=f'{col_name} error')
        
        plt.tight_layout()
        
        sample_filename = f'3d_scatter_sample_{sample_idx+1:03d}.png'
        plt.savefig(os.path.join(scatter_plots_dir, sample_filename), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"3D scatter plots for {col_name} saved to: {scatter_plots_dir}")


def create_video_from_scatter_plots(col_name, col_dir, args):
    """Create a video from the 3D scatter plot images."""
    print(f"Creating video from 3D scatter plots for {col_name}...")
    
    scatter_plots_dir = os.path.join(col_dir, '3d_scatter_plots')
    
    if not os.path.exists(scatter_plots_dir):
        print(f"Warning: Scatter plots directory not found: {scatter_plots_dir}")
        return
    
    image_pattern = os.path.join(scatter_plots_dir, '3d_scatter_sample_*.png')
    image_files = glob.glob(image_pattern)
    
    if not image_files:
        print(f"Warning: No scatter plot images found in {scatter_plots_dir}")
        return
    
    image_files.sort()
    print(f"Found {len(image_files)} scatter plot images for {col_name}")
    
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape
    
    fps = 10
    video_filename = os.path.join(col_dir, f'3d_scatter_plots_video_{col_name}.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    for image_file in tqdm(image_files, desc=f"Creating video frames for {col_name}"):
        frame = cv2.imread(image_file)
        video_writer.write(frame)
    
    video_writer.release()
    
    print(f"Video saved to: {video_filename}")


def main():
    """Main function to generate predictions and create visualizations."""
    # Parse arguments
    args = setup_arguments()
    
    # Calculate transforms (same as training)
    coord_transform = calculate_coord_transform(args.raw_data_dir)
    obs_transform = calculate_obs_transform(args.raw_data_dir)
    
    # Create datasets
    train_ds, val_ds = create_patch_datasets(
        args.patch_data_dir,
        coord_transform,
        obs_transform,
        target_col_indices=args.target_col_indices,
        input_window_size=args.input_window_size,
        output_window_size=args.output_window_size,
    )
    
    print(f"Train dataset length: {len(train_ds)}")
    print(f"Val dataset length: {len(val_ds)}")
    print(f"Target columns: {args.target_cols} (indices: {args.target_col_indices})")
    
    # Load model
    model = load_gino_model(args.model_path, args)
    
    # Generate predictions
    train_results = generate_predictions(model, train_ds, args, 'train')
    val_results = generate_predictions(model, val_ds, args, 'val')
    
    results_dict = {
        'train': train_results,
        'val': val_results
    }
    
    # Save results (with denormalization)
    save_results(results_dict, args, obs_transform, coord_transform)
    
    # Compute and save metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(results_dict, args, obs_transform)
    save_metrics(metrics, args)
    
    # Create visualizations (pass obs_transform for denormalization)
    create_visualizations(results_dict, args, obs_transform)
    
    print("Prediction generation and visualization complete!")


if __name__ == "__main__":
    main()

