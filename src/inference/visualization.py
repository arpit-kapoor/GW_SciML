"""
Visualization utilities for model predictions.

Create various plots comparing predictions vs observations including:
- Scatter plots
- Time series comparisons
- Error analysis
- 3D spatial visualizations
- Video generation
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from tqdm import tqdm


def create_scatter_comparison_plots(predictions, targets, output_path, 
                                    n_samples=5, timesteps=[0, 2, 4], 
                                    title_prefix=''):
    """
    Create scatter plots comparing predictions vs observations.
    
    Args:
        predictions (np.ndarray): Shape [N_samples, N_points, output_window_size]
        targets (np.ndarray): Shape [N_samples, N_points, output_window_size]
        output_path (str): Path to save plot
        n_samples (int): Number of samples to plot
        timesteps (list): Timesteps to visualize
        title_prefix (str): Prefix for plot titles
    """
    n_samples = min(n_samples, predictions.shape[0])
    timesteps = [t for t in timesteps if t < predictions.shape[2]]
    
    fig, axes = plt.subplots(n_samples, len(timesteps), 
                            figsize=(20, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(n_samples):
        for plot_idx, timestep in enumerate(timesteps):
            ax = axes[sample_idx, plot_idx]
            
            # Get predictions and targets for this sample and timestep
            pred_t = predictions[sample_idx, :, timestep]
            target_t = targets[sample_idx, :, timestep]
            
            # Create scatter plot
            ax.scatter(target_t, pred_t, alpha=0.6, s=1)
            
            # Add perfect prediction line
            min_val = min(target_t.min(), pred_t.min())
            max_val = max(target_t.max(), pred_t.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                   alpha=0.8, label='Perfect prediction')
            
            # Calculate correlation
            correlation = np.corrcoef(target_t, pred_t)[0, 1]
            
            ax.set_xlabel('Observed')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{title_prefix}Sample {sample_idx+1}, Timestep {timestep+1}\n'
                        f'Corr: {correlation:.3f}')
            ax.grid(True, alpha=0.3)
            
            if sample_idx == 0 and plot_idx == 0:
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter comparison plots saved to: {output_path}")


def create_timeseries_comparison_plots(predictions, targets, output_path,
                                       n_samples=5, variable_name='Variable'):
    """
    Create time series plots showing mean predictions vs observations over time.
    
    Args:
        predictions (np.ndarray): Shape [N_samples, N_points, output_window_size]
        targets (np.ndarray): Shape [N_samples, N_points, output_window_size]
        output_path (str): Path to save plot
        n_samples (int): Number of samples to plot
        variable_name (str): Name of the variable being predicted
    """
    n_samples = min(n_samples, predictions.shape[0])
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
    
    if n_samples == 1:
        axes = [axes]
    
    for sample_idx in range(n_samples):
        ax = axes[sample_idx]
        
        # Average across all points for each timestep
        pred_timeseries = predictions[sample_idx].mean(axis=0)
        target_timeseries = targets[sample_idx].mean(axis=0)
        
        timesteps = np.arange(len(pred_timeseries))
        
        ax.plot(timesteps, target_timeseries, 'b-', label='Observed', linewidth=2)
        ax.plot(timesteps, pred_timeseries, 'r--', label='Predicted', linewidth=2)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'{variable_name} (spatial mean)')
        ax.set_title(f'Time Series Comparison - Sample {sample_idx+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time series comparison plots saved to: {output_path}")


def create_error_analysis_plots(predictions, targets, output_path):
    """
    Create error analysis plots showing MAE and RMSE over time.
    
    Args:
        predictions (np.ndarray): Shape [N_samples, N_points, output_window_size]
        targets (np.ndarray): Shape [N_samples, N_points, output_window_size]
        output_path (str): Path to save plot
        
    Returns:
        dict: Dictionary with error statistics
    """
    errors = predictions - targets
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # Error statistics by timestep
    mae_by_timestep = np.mean(np.abs(errors), axis=(0, 1))
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error analysis plots saved to: {output_path}")
    print(f"Overall MAE: {mae:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mae_by_timestep': mae_by_timestep,
        'rmse_by_timestep': rmse_by_timestep
    }


def create_3d_spatial_plots(predictions, targets, coords, output_dir,
                           variable_name='Variable', timestep=0):
    """
    Create 3D scatter plots showing spatial distribution of predictions vs observations.
    
    Args:
        predictions (np.ndarray): Shape [N_samples, N_points, output_window_size]
        targets (np.ndarray): Shape [N_samples, N_points, output_window_size]
        coords (np.ndarray): Shape [N_samples, N_points, 3]
        output_dir (str): Directory to save plots
        variable_name (str): Name of the variable being predicted
        timestep (int): Which timestep to visualize
        
    Returns:
        str: Path to directory containing plots
    """
    # Create subdirectory for 3D plots
    scatter_dir = os.path.join(output_dir, '3d_scatter_plots')
    os.makedirs(scatter_dir, exist_ok=True)
    
    n_samples = predictions.shape[0]
    
    print(f"Creating 3D spatial plots for {n_samples} samples...")
    
    for sample_idx in tqdm(range(n_samples), desc="Creating 3D plots"):
        # Get data for this sample
        coords_sample = coords[sample_idx]
        pred_sample = predictions[sample_idx, :len(coords_sample), timestep]
        target_sample = targets[sample_idx, :len(coords_sample), timestep]
        
        # Calculate color scale based on observations
        vmin = np.min(target_sample)
        vmax = np.max(target_sample)
        
        # Create figure with three subplots
        fig = plt.figure(figsize=(20, 6))
        
        # Observations
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        scatter1 = ax1.scatter(coords_sample[:, 0], coords_sample[:, 1], coords_sample[:, 2],
                              c=target_sample, cmap='viridis', s=5, alpha=0.7,
                              vmin=vmin, vmax=vmax)
        ax1.set_title(f'Observations - Sample {sample_idx+1}\nTimestep {timestep+1}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=20,
                    label=f'{variable_name} (observed)')
        
        # Predictions
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        scatter2 = ax2.scatter(coords_sample[:, 0], coords_sample[:, 1], coords_sample[:, 2],
                              c=pred_sample, cmap='viridis', s=5, alpha=0.7,
                              vmin=vmin, vmax=vmax)
        ax2.set_title(f'Predictions - Sample {sample_idx+1}\nTimestep {timestep+1}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=20,
                    label=f'{variable_name} (predicted)')
        
        # Error
        error = target_sample - pred_sample
        error_range = max(abs(error.min()), abs(error.max()))
        
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        scatter3 = ax3.scatter(coords_sample[:, 0], coords_sample[:, 1], coords_sample[:, 2],
                              c=error, cmap='RdBu_r', s=5, alpha=0.7,
                              vmin=-error_range, vmax=error_range)
        ax3.set_title(f'Error (Obs - Pred) - Sample {sample_idx+1}\nTimestep {timestep+1}')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        plt.colorbar(scatter3, ax=ax3, shrink=0.5, aspect=20,
                    label=f'{variable_name} error')
        
        plt.tight_layout()
        
        # Save plot
        filename = f'3d_scatter_sample_{sample_idx+1:03d}.png'
        plt.savefig(os.path.join(scatter_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"3D spatial plots saved to: {scatter_dir}")
    return scatter_dir


def create_video_from_images(image_dir, output_path, fps=10, pattern='3d_scatter_sample_*.png'):
    """
    Create video from sequence of images.
    
    Args:
        image_dir (str): Directory containing images
        output_path (str): Path to save video file
        fps (int): Frames per second
        pattern (str): Glob pattern to match image files
    """
    print("Creating video from images...")
    
    # Get all matching image files
    image_pattern = os.path.join(image_dir, pattern)
    image_files = sorted(glob.glob(image_pattern))
    
    if not image_files:
        print(f"Warning: No images found matching pattern: {image_pattern}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for image_file in tqdm(image_files, desc="Writing video frames"):
        frame = cv2.imread(image_file)
        video_writer.write(frame)
    
    video_writer.release()
    
    print(f"Video saved to: {output_path}")
    print(f"Video details: {len(image_files)} frames at {fps} fps")


def create_all_visualizations(results_dict, args):
    """
    Orchestrate creation of all visualizations.
    
    Args:
        results_dict (dict): Dictionary with 'train' and 'val' results
        args (argparse.Namespace): Arguments containing configuration
    """
    print("\nCreating visualizations...")
    
    # Use validation data for visualization
    predictions = results_dict['val']['predictions']
    targets = results_dict['val']['targets']
    coords = results_dict['val']['coords']
    
    variable_name = getattr(args, 'target_col', 'Variable')
    
    # 1. Scatter comparison plots
    scatter_path = os.path.join(args.results_dir, 'predictions_vs_observations.png')
    create_scatter_comparison_plots(predictions, targets, scatter_path)
    
    # 2. Time series comparison plots
    timeseries_path = os.path.join(args.results_dir, 'time_series_comparison.png')
    create_timeseries_comparison_plots(predictions, targets, timeseries_path,
                                       variable_name=variable_name)
    
    # 3. Error analysis plots
    error_path = os.path.join(args.results_dir, 'error_analysis.png')
    error_stats = create_error_analysis_plots(predictions, targets, error_path)
    
    # 4. 3D spatial plots
    scatter_dir = create_3d_spatial_plots(predictions, targets, coords,
                                          args.results_dir, variable_name=variable_name)
    
    # 5. Create video from 3D plots
    video_path = os.path.join(args.results_dir, '3d_scatter_plots_video.mp4')
    create_video_from_images(scatter_dir, video_path)
    
    print(f"\nAll visualizations saved to: {args.results_dir}")
    
    return error_stats
