"""
Generate predictions using trained GINO model on the entire groundwater dataset.

This script:
- Loads a trained GINO model from a saved state dict
- Creates datasets for both training and validation patches
- Generates predictions on all patches using the same infrastructure as training
- Saves predictions, ground truth, and metadata for analysis
- Creates visualizations comparing predictions vs observations at different timesteps
- Generates video from 3D scatter plots

Usage:
    python generate_gino_predictions.py --model-path /path/to/model.pth --results-dir /path/to/results
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
from src.data.patch_dataset import GWPatchDataset
from src.data.batch_sampler import PatchBatchSampler
from src.models.neuralop.gino import GINO


def setup_arguments():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description='Generate predictions using trained GINO model')
    
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
    parser.add_argument('--target-col', type=str, default='head',
                       help='Target observation column name')
    parser.add_argument('--input-window-size', type=int, default=10,
                       help='Number of time steps in each input sequence')
    parser.add_argument('--output-window-size', type=int, default=10,
                       help='Number of time steps in each output sequence')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for inference (cuda, cpu, or auto)')
    
    args = parser.parse_args()
    
    # Construct full paths
    args.raw_data_dir = os.path.join(args.base_data_dir, args.raw_data_subdir)
    args.patch_data_dir = os.path.join(args.base_data_dir, args.patch_data_subdir)
    
    # Create results directory with timestamp
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.results_dir = os.path.join(args.results_dir, f'gino_predictions_{timestamp}')
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Configure model parameters (must match training configuration)
    args = configure_model_parameters(args)
    args = configure_target_col_idx(args)
    args = configure_device(args)
    
    print(f"Model path: {args.model_path}")
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Patch data directory: {args.patch_data_dir}")
    print(f"Results directory: {args.results_dir}")
    
    return args


def configure_model_parameters(args):
    """Configure model parameters to match training configuration."""
    # These parameters must match the training configuration exactly
    args.coord_dim = 3
    args.gno_radius = 0.15  # Increased for better aggregation
    args.in_gno_out_channels = args.input_window_size
    args.in_gno_channel_mlp_layers = [32, 64, 32]
    args.fno_n_layers = 4
    args.fno_n_modes = (12, 12, 8)  # Increased from 8
    args.fno_hidden_channels = 64  # Increased from 32
    args.lifting_channels = 64     # Increased from 32
    args.out_gno_channel_mlp_layers = [32, 64, 32]
    args.projection_channel_ratio = 2
    args.out_channels = args.output_window_size
    args.latent_query_dims = (32, 32, 24)  # Updated to match training
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


def configure_target_col_idx(args):
    """Configure target column index from name."""
    names_to_idx = {
        'mass_concentration': 0,
        'head': 1,
        'pressure': 2
    }
    args.target_col_idx = names_to_idx[args.target_col]
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
    train_ds = GWPatchDataset(
        data_path=patch_data_dir,
        dataset='train', 
        coord_transform=coord_transform, 
        obs_transform=obs_transform,
        input_window_size=kwargs.get('input_window_size', 10),
        output_window_size=kwargs.get('output_window_size', 10),
        target_col_idx=kwargs.get('target_col_idx', None),
    )
    
    val_ds = GWPatchDataset(
        data_path=patch_data_dir,
        dataset='val', 
        coord_transform=coord_transform, 
        obs_transform=obs_transform,
        input_window_size=kwargs.get('input_window_size', 10),
        output_window_size=kwargs.get('output_window_size', 10),
        target_col_idx=kwargs.get('target_col_idx', None),
    )
    
    return train_ds, val_ds


def load_gino_model(model_path, args):
    """Load trained GINO model from checkpoint file.
    
    The checkpoint contains the full training state including:
    - model_state_dict: The actual model parameters we need
    - optimizer_state_dict: Not needed for inference
    - scheduler_state_dict: Not needed for inference
    - training history and args: Used to validate configuration
    """
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
        # This provides a regular grid for the FNO component
        coords_min = torch.min(point_coords, dim=0).values
        coords_max = torch.max(point_coords, dim=0).values
        latent_query_arr = [
            torch.linspace(coords_min[i], coords_max[i], args.latent_query_dims[i], device=args.device)
            for i in range(args.coord_dim)
        ]
        # Create meshgrid and stack to get [Qx, Qy, Qz, 3] tensor
        latent_queries = torch.stack(torch.meshgrid(*latent_query_arr, indexing='ij'), dim=-1)
        
        # Build batched sequences: concat along points (dim=0), batch along dim=0
        x_list, y_list = [], []
        for sample in batch_samples:
            # Combine core and ghost inputs/outputs for each sample
            sample_input = torch.concat([sample['core_in'], sample['ghost_in']], dim=0).float().unsqueeze(0)
            sample_output = torch.concat([sample['core_out'], sample['ghost_out']], dim=0).float().unsqueeze(0)
            x_list.append(sample_input)
            y_list.append(sample_output)
        
        # Stack all sequences into batch tensors
        x = torch.cat(x_list, dim=0)  # [B, N_points, input_window_size]
        y = torch.cat(y_list, dim=0)  # [B, N_points, output_window_size]
        
        return {
            'point_coords': point_coords,      # [N_points, 3]
            'latent_queries': latent_queries,  # [Qx, Qy, Qz, 3]
            'x': x,                           # [B, N_points, input_window_size]
            'y': y,                           # [B, N_points, output_window_size]
            'core_len': len(core_coords),     # Number of core points (for loss masking)
            'patch_id': sample['patch_id']
        }
    return collate_fn


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

            print(f"{dataset_name} patch_id: {patch_id}")
            print(f"{dataset_name} core_output shape: {core_output.shape}")
            print(f"{dataset_name} core_target shape: {core_target.shape}")
            print(f"{dataset_name} core_coords shape: {core_coords.shape}")
            
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
    
    print(f"{dataset_name} predictions shape: {predictions.shape}")
    print(f"{dataset_name} targets shape: {targets.shape}")
    print(f"{dataset_name} coords shape: {coords.shape}")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'coords': coords,
        'metadata': all_patch_metadata
    }


def save_results(results_dict, args):
    """Save predictions and metadata to files."""
    print("Saving results...")
    
    # Save train results
    np.save(os.path.join(args.results_dir, 'train_predictions.npy'), results_dict['train']['predictions'])
    np.save(os.path.join(args.results_dir, 'train_targets.npy'), results_dict['train']['targets'])
    
    # Save validation results
    np.save(os.path.join(args.results_dir, 'val_predictions.npy'), results_dict['val']['predictions'])
    np.save(os.path.join(args.results_dir, 'val_targets.npy'), results_dict['val']['targets'])
    
    # Save coordinates
    with open(os.path.join(args.results_dir, 'train_coords.pkl'), 'wb') as f:
        pickle.dump(results_dict['train']['coords'], f)
    with open(os.path.join(args.results_dir, 'val_coords.pkl'), 'wb') as f:
        pickle.dump(results_dict['val']['coords'], f)
    
    # Save metadata
    with open(os.path.join(args.results_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({
            'train_metadata': results_dict['train']['metadata'],
            'val_metadata': results_dict['val']['metadata'],
            'args': vars(args)
        }, f)
    
    print(f"Results saved to: {args.results_dir}")


def create_visualizations(results_dict, args):
    """Create visualizations comparing predictions vs observations."""
    print("Creating visualizations...")
    
    # Use validation data for visualization
    predictions = results_dict['val']['predictions']  # [N_samples, N_points, output_window_size]
    targets = results_dict['val']['targets']
    
    # Create plots for different timesteps
    timesteps_to_plot = [0, 2, 4]  # First, middle, and last timesteps
    n_samples_to_plot = min(5, predictions.shape[0])  # Plot first 5 samples
    
    fig, axes = plt.subplots(n_samples_to_plot, len(timesteps_to_plot), 
                            figsize=(20, 4*n_samples_to_plot))
    
    if n_samples_to_plot == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(n_samples_to_plot):
        for plot_idx, timestep in enumerate(timesteps_to_plot):
            ax = axes[sample_idx, plot_idx]
            
            # Get predictions and targets for this sample and timestep
            pred_t = predictions[sample_idx, :, timestep]  # [N_points]
            target_t = targets[sample_idx, :, timestep]
            
            # Create scatter plot comparing predictions vs targets
            ax.scatter(target_t, pred_t, alpha=0.6, s=1)
            
            # Add perfect prediction line
            min_val = min(target_t.min(), pred_t.min())
            max_val = max(target_t.max(), pred_t.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect prediction')
            
            # Calculate correlation
            correlation = np.corrcoef(target_t, pred_t)[0, 1]
            
            ax.set_xlabel('Observed')
            ax.set_ylabel('Predicted')
            ax.set_title(f'Sample {sample_idx+1}, Timestep {timestep+1}\nCorr: {correlation:.3f}')
            ax.grid(True, alpha=0.3)
            
            if sample_idx == 0 and plot_idx == 0:
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'predictions_vs_observations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create time series plots for first few samples
    fig, axes = plt.subplots(n_samples_to_plot, 1, figsize=(12, 3*n_samples_to_plot))
    
    if n_samples_to_plot == 1:
        axes = [axes]
    
    for sample_idx in range(n_samples_to_plot):
        ax = axes[sample_idx]
        
        # Average predictions and targets across all points for each timestep
        pred_timeseries = predictions[sample_idx].mean(axis=0)  # [output_window_size]
        target_timeseries = targets[sample_idx].mean(axis=0)
        
        timesteps = np.arange(len(pred_timeseries))
        
        ax.plot(timesteps, target_timeseries, 'b-', label='Observed', linewidth=2)
        ax.plot(timesteps, pred_timeseries, 'r--', label='Predicted', linewidth=2)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'{args.target_col} (spatial mean)')
        ax.set_title(f'Time Series Comparison - Sample {sample_idx+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'time_series_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create error statistics
    errors = predictions - targets
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # Error statistics by timestep
    mae_by_timestep = np.mean(np.abs(errors), axis=(0, 1))  # [output_window_size]
    rmse_by_timestep = np.sqrt(np.mean(errors**2, axis=(0, 1)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    timesteps = np.arange(len(mae_by_timestep))
    
    ax1.plot(timesteps, mae_by_timestep, 'b-', marker='o', linewidth=2)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('MAE by Timestep')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(timesteps, rmse_by_timestep, 'r-', marker='s', linewidth=2)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Root Mean Square Error')
    ax2.set_title('RMSE by Timestep')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'error_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create 3D scatter plots for first timestep across samples
    coords_data = results_dict['val']['coords']  # Dictionary with patch coordinates
    n_samples_to_plot_3d = predictions.shape[0]  # Plot all samples

    print(f"Observations shape: {targets.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Coords shape: {coords_data.shape}")
    
    # Create subdirectory for 3D scatter plots
    scatter_plots_dir = os.path.join(args.results_dir, '3d_scatter_plots')
    os.makedirs(scatter_plots_dir, exist_ok=True)
    
    
    for sample_idx in range(n_samples_to_plot_3d):
        
        # Get coordinates for this patch - take first sample if multiple samples per patch
        coords = coords_data[sample_idx]  # [N_points, 3]
        
        # Get predictions and targets for first timestep (t=0) of this sample
        pred_first_timestep = predictions[sample_idx, :len(coords), 0]  # [N_points]
        target_first_timestep = targets[sample_idx, :len(coords), 0]
        
        # Calculate vmin and vmax based on observations for consistent color scale
        vmin = np.min(target_first_timestep)
        vmax = np.max(target_first_timestep)
        
        # Create figure with three subplots for this sample
        fig = plt.figure(figsize=(20, 6))
        
        # Subplot for observations
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                              c=target_first_timestep, cmap='viridis', s=5, alpha=0.7,
                              vmin=vmin, vmax=vmax)
        ax1.set_title(f'Observations - Sample {sample_idx+1}\nFirst Timestep')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=20, 
                    label=f'{args.target_col} (observed)')
        
        # Subplot for predictions
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                              c=pred_first_timestep, cmap='viridis', s=5, alpha=0.7,
                              vmin=vmin, vmax=vmax)
        ax2.set_title(f'Predictions - Sample {sample_idx+1}\nFirst Timestep')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=20, 
                    label=f'{args.target_col} (predicted)')
        
        # Calculate error
        error = target_first_timestep - pred_first_timestep
        
        # Subplot for error
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        # Use diverging colormap centered at 0 for error
        error_range = max(abs(error.min()), abs(error.max()))
        scatter3 = ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                              c=error, cmap='RdBu_r', s=5, alpha=0.7,
                              vmin=-error_range, vmax=error_range)
        ax3.set_title(f'Error (Obs - Pred) - Sample {sample_idx+1}\nFirst Timestep')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        plt.colorbar(scatter3, ax=ax3, shrink=0.5, aspect=20, 
                    label=f'{args.target_col} error')
        
        plt.tight_layout()
        
        # Save each sample to a separate file
        sample_filename = f'3d_scatter_sample_{sample_idx+1:03d}.png'
        plt.savefig(os.path.join(scatter_plots_dir, sample_filename), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"3D scatter plots saved to: {scatter_plots_dir}")
    
    print(f"Overall MAE: {mae:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Visualizations saved to: {args.results_dir}")


def create_video_from_scatter_plots(args):
    """Create a video from the 3D scatter plot images."""
    print("Creating video from 3D scatter plots...")
    
    # Path to the scatter plots directory
    scatter_plots_dir = os.path.join(args.results_dir, '3d_scatter_plots')
    
    # Check if scatter plots directory exists
    if not os.path.exists(scatter_plots_dir):
        print(f"Warning: Scatter plots directory not found: {scatter_plots_dir}")
        return
    
    # Get all PNG files in the scatter plots directory and sort them
    image_pattern = os.path.join(scatter_plots_dir, '3d_scatter_sample_*.png')
    image_files = glob.glob(image_pattern)
    
    if not image_files:
        print(f"Warning: No scatter plot images found in {scatter_plots_dir}")
        return
    
    # Sort files by filename to ensure proper order
    image_files.sort()
    
    print(f"Found {len(image_files)} scatter plot images")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape
    
    # Define video parameters
    fps = 10
    video_filename = os.path.join(args.results_dir, '3d_scatter_plots_video.mp4')
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    # Write each image to the video
    for image_file in tqdm(image_files, desc="Creating video frames"):
        frame = cv2.imread(image_file)
        video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()
    
    print(f"Video saved to: {video_filename}")
    print(f"Video details: {len(image_files)} frames at {fps} fps")


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
        target_col_idx=args.target_col_idx,
        input_window_size=args.input_window_size,
        output_window_size=args.output_window_size,
    )
    
    print(f"Train dataset length: {len(train_ds)}")
    print(f"Val dataset length: {len(val_ds)}")
    print(f"Target column: {args.target_col} (index: {args.target_col_idx})")
    
    # Load model
    model = load_gino_model(args.model_path, args)
    
    # Generate predictions
    train_results = generate_predictions(model, train_ds, args, 'train')
    val_results = generate_predictions(model, val_ds, args, 'val')
    
    results_dict = {
        'train': train_results,
        'val': val_results
    }
    
    # Save results
    save_results(results_dict, args)
    
    # Create visualizations
    create_visualizations(results_dict, args)
    
    # Create video from scatter plots
    create_video_from_scatter_plots(args)
    
    print("Prediction generation and visualization complete!")


if __name__ == "__main__":
    main()
