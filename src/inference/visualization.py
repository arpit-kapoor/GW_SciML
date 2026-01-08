"""
Visualization utilities for model predictions.

Create various plots comparing predictions vs observations including:
- Scatter plots
- Time series comparisons
- Error analysis
- 3D spatial visualizations
- Video generation
- Variance-based node selection
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from tqdm import tqdm


def select_nodes_by_variance(targets, n_target_cols, target_cols):
    """
    Select nodes with high variance (above 99th percentile) for each target column.
    
    Args:
        targets: Array of shape [N_samples, N_points, output_window_size, n_target_cols]
        n_target_cols: Number of target columns
        target_cols: List of target column names
        
    Returns:
        Array of shape [5, n_target_cols] with selected node indices
    """
    # Compute variance for each node for each target column
    # Use first timestep for consistency
    node_variance = np.var(targets[:, :, 0, :], axis=0)  # [n_nodes, n_target_cols]
    print(f"Node variance shape: {node_variance.shape}")
    
    selected_node_idx = []
    for col_idx, col_name in enumerate(target_cols):
        p99 = np.percentile(node_variance[:, col_idx], 99)
        print(f"{col_name} - 99th percentile variance: {p99:.6f}")
        
        # Identify nodes above 99th percentile for this column
        node_idx_in_range = np.where(node_variance[:, col_idx] > p99)[0]
        print(f"{col_name} - Number of nodes above 99th percentile: {len(node_idx_in_range)}")
        
        # Randomly select 5 nodes from those above 99th percentile
        np.random.seed(42)  # For reproducibility
        if len(node_idx_in_range) >= 5:
            selected = np.random.choice(node_idx_in_range, size=(5, 1), replace=False)
        else:
            # If fewer than 5 nodes, take all and pad with random high-variance nodes
            selected = node_idx_in_range.reshape(-1, 1)
            # Pad with additional high-variance nodes
            sorted_indices = np.argsort(node_variance[:, col_idx])[::-1]
            additional_needed = 5 - len(node_idx_in_range)
            for idx in sorted_indices:
                if idx not in node_idx_in_range and additional_needed > 0:
                    selected = np.vstack([selected, idx])
                    additional_needed -= 1
        
        selected_node_idx.append(selected)
    
    selected_node_idx = np.hstack(selected_node_idx)  # Shape: [5, n_target_cols]
    print(f"Selected nodes shape: {selected_node_idx.shape}")
    
    return selected_node_idx


def create_scatter_comparison_plots(predictions, targets, output_path, 
                                    selected_node_idx=None, output_window_size=1,
                                    title_prefix=''):
    """
    Create scatter plots comparing predictions vs observations at different timesteps.
    
    Uses variance-based selected nodes and adapts layout based on output window size.
    Matches legacy GINO implementation.
    
    Args:
        predictions (np.ndarray): Shape [N_samples, N_points, output_window_size]
        targets (np.ndarray): Shape [N_samples, N_points, output_window_size]
        output_path (str): Path to save plot
        selected_node_idx (np.ndarray): Array of node indices to plot (default: None, will use first 5)
        output_window_size (int): Number of output timesteps
        title_prefix (str): Prefix for plot titles
    """
    # Adapt timesteps to plot based on output window size
    if output_window_size == 1:
        timesteps_to_plot = [0]  # Only first timestep
    elif output_window_size == 2:
        timesteps_to_plot = [0, 1]  # First two timesteps
    elif output_window_size == 3:
        timesteps_to_plot = [0, 1, 2]  # First three timesteps
    else:
        timesteps_to_plot = [0, output_window_size//2, output_window_size-1]  # First, middle, last
    
    # Use provided node indices or default to first 5
    if selected_node_idx is None:
        n_nodes_to_plot = min(5, predictions.shape[1])
        sample_indices = np.arange(n_nodes_to_plot)
    else:
        sample_indices = selected_node_idx
        n_nodes_to_plot = len(sample_indices)
    
    # Adaptable figure size
    fig, axes = plt.subplots(n_nodes_to_plot, len(timesteps_to_plot), 
                            figsize=(7*len(timesteps_to_plot), 5*n_nodes_to_plot))
    
    # Ensure axes is always 2D for consistent indexing
    if n_nodes_to_plot == 1 and len(timesteps_to_plot) == 1:
        axes = np.array([[axes]])
    elif n_nodes_to_plot == 1:
        axes = axes.reshape(1, -1)
    elif len(timesteps_to_plot) == 1:
        axes = axes.reshape(-1, 1)
    
    for axes_idx, sample_idx in enumerate(sample_indices):
        for plot_idx, timestep in enumerate(timesteps_to_plot):
            ax = axes[axes_idx, plot_idx]
            
            # Get predictions and targets across all temporal positions for this node and timestep
            pred_t = predictions[:, sample_idx, timestep]
            target_t = targets[:, sample_idx, timestep]
            
            # Create scatter plot
            ax.scatter(target_t, pred_t, alpha=0.6, s=5)
            
            # Add perfect prediction line
            min_val = min(target_t.min(), pred_t.min())
            max_val = max(target_t.max(), pred_t.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                   alpha=0.8, linewidth=2.5, label='Perfect')
            
            # Calculate correlation
            correlation = np.corrcoef(target_t, pred_t)[0, 1]
            
            ax.set_xlabel('Observed', fontsize=24)
            ax.set_ylabel('Predicted', fontsize=24)
            ax.set_title(f'Node {sample_idx}, Timestep {timestep+1}\nCorr: {correlation:.3f}', 
                        fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.grid(True, alpha=0.3)
            
            if axes_idx == 0 and plot_idx == 0:
                ax.legend(fontsize=22)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter comparison plots saved to: {output_path}")


def create_timeseries_comparison_plots(predictions, targets, col_name, timeseries_dir,
                                       selected_node_idx=None, output_window_size=1):
    """
    Create time series plots showing predictions vs observations over the temporal sequence.
    
    Generates individual PNG files for each timestep in the prediction horizon,
    showing evolution over the temporal sequence (N_samples dimension) for selected spatial nodes.
    Matches legacy GINO implementation.
    
    Args:
        predictions (np.ndarray): Shape [N_samples, N_points, output_window_size]
        targets (np.ndarray): Same shape as predictions
        col_name (str): Name of the target column
        timeseries_dir (str): Directory to save time series plots
        selected_node_idx (np.ndarray): Array of node indices to plot (default: None, will use first 5)
        output_window_size (int): Number of output timesteps
    """
    # Use provided node indices or default to first 5
    if selected_node_idx is None:
        n_nodes_to_plot = min(5, predictions.shape[1])
        sample_indices = np.arange(n_nodes_to_plot)
    else:
        sample_indices = selected_node_idx
        n_nodes_to_plot = len(sample_indices)
    
    # Iterate over all timesteps in the prediction horizon
    for timestep_idx in range(output_window_size):
        # Create a figure with subplots for all selected nodes
        fig, axes = plt.subplots(n_nodes_to_plot, 1, figsize=(12, 5*n_nodes_to_plot))
        
        if n_nodes_to_plot == 1:
            axes = [axes]
        
        for axes_idx, sample_idx in enumerate(sample_indices):
            ax = axes[axes_idx]
            
            # Get values at this specific timestep for this node across all temporal positions
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


def create_error_analysis_plots(predictions, targets, output_path):
    """
    Create error analysis plots showing MAE and RMSE over the temporal sequence.
    
    The first dimension (N_samples) represents the temporal sequence. This function
    computes error metrics for each timestep in the sequence by reducing across
    spatial points (P), output window (T), and columns (C) dimensions.
    
    Args:
        predictions (np.ndarray): Shape [N_samples, N_points, output_window_size] or
                                         [N_samples, N_points, output_window_size, n_cols]
        targets (np.ndarray): Same shape as predictions
        output_path (str): Path to save plot
        
    Returns:
        dict: Dictionary with error statistics
    """
    errors = predictions - targets
    
    # Overall statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # Error statistics for each N (temporal position)
    # Reduce across P (spatial), T (output window), and C (columns if present)
    # This gives us errors as a function of the temporal sequence
    reduce_axes = tuple(range(1, errors.ndim))  # All axes except the first (N)
    mae_by_sequence = np.mean(np.abs(errors), axis=reduce_axes)
    rmse_by_sequence = np.sqrt(np.mean(errors**2, axis=reduce_axes))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    sequence_indices = np.arange(len(mae_by_sequence))
    
    ax1.plot(sequence_indices, mae_by_sequence, 'b-', marker='o', linewidth=2, markersize=3)
    ax1.set_xlabel('Temporal Sequence Position (N)')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('MAE Over Temporal Sequence')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(sequence_indices, rmse_by_sequence, 'r-', marker='s', linewidth=2, markersize=3)
    ax2.set_xlabel('Temporal Sequence Position (N)')
    ax2.set_ylabel('Root Mean Square Error')
    ax2.set_title('RMSE Over Temporal Sequence')
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
        'mae_by_sequence': mae_by_sequence,
        'rmse_by_sequence': rmse_by_sequence
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


def create_per_column_visualizations(results_dict, target_cols, target_col_indices, output_window_size,
                                     results_dir, obs_transform, create_3d_plots=False):
    """
    Create visualizations for each target column separately for both train and val datasets.
    
    Generic function that uses variance-based node selection and adaptable figure sizes.
    Can be reused by FNO, GINO, and other multi-column prediction models.
    
    Args:
        results_dict (dict): Dictionary with 'train' and 'val' results
        target_cols (list): List of target column names
        target_col_indices (list): List of target column indices
        output_window_size (int): Number of output timesteps
        results_dir (str): Base directory for saving results
        obs_transform: Normalize transform object for denormalization
        create_3d_plots (bool): Whether to create 3D plots and videos
    """
    from src.inference.metrics import denormalize_observations
    
    print("\nCreating per-column visualizations...")
    
    # Process both train and val datasets
    for dataset_name in ['train', 'val']:
        print(f"\n{'='*60}")
        print(f"Creating visualizations for {dataset_name} dataset...")
        print(f"{'='*60}")
        
        # Create dataset-specific directory
        dataset_dir = os.path.join(results_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Get normalized data
        predictions_norm = results_dict[dataset_name]['predictions']  # [N_samples, N_points, T, n_cols]
        targets_norm = results_dict[dataset_name]['targets']
        coords_data = results_dict[dataset_name]['coords']
        
        # Denormalize to original scale for visualization
        predictions = denormalize_observations(predictions_norm, obs_transform, target_col_indices)
        targets = denormalize_observations(targets_norm, obs_transform, target_col_indices)
        
        print(f"Denormalized {dataset_name} predictions to original scale")
        print(f"  Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"  Targets range: [{targets.min():.4f}, {targets.max():.4f}]")
        
        # Select nodes based on variance
        n_target_cols = len(target_cols)
        selected_node_idx = select_nodes_by_variance(targets, n_target_cols, target_cols)
        
        # Create visualizations for each target column
        for col_idx, col_name in enumerate(target_cols):
            print(f"\nGenerating visualizations for: {col_name} ({dataset_name})")
            
            # Create column-specific directory
            col_dir = os.path.join(dataset_dir, col_name)
            os.makedirs(col_dir, exist_ok=True)
            
            # Extract predictions and targets for this column
            # Shape: [N_samples, N_points, T, n_cols] -> [N_samples, N_points, T]
            col_predictions = predictions[:, :, :, col_idx]
            col_targets = targets[:, :, :, col_idx]
            
            # Get selected nodes for this column
            col_selected_nodes = selected_node_idx[:, col_idx]
            
            # 1. Scatter comparison plots with variance-based node selection
            scatter_path = os.path.join(col_dir, f'{col_name}_pred_vs_obs_scatter.png')
            create_scatter_comparison_plots(
                col_predictions, col_targets, scatter_path,
                title_prefix=col_name,
                selected_node_idx=col_selected_nodes,
                output_window_size=output_window_size
            )
            
            # 2. Time series comparison plots with variance-based node selection
            timeseries_dir = os.path.join(col_dir, 'time_series_plots')
            os.makedirs(timeseries_dir, exist_ok=True)
            create_timeseries_comparison_plots(
                col_predictions, col_targets, col_name, timeseries_dir,
                selected_node_idx=col_selected_nodes,
                output_window_size=output_window_size
            )
            
            # 3. Error analysis plots
            error_path = os.path.join(col_dir, 'error_analysis.png')
            create_error_analysis_plots(col_predictions, col_targets, error_path)
            
            # Print error statistics
            errors = col_predictions - col_targets
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            print(f"{col_name} ({dataset_name}) - Overall MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            
            # 4. 3D spatial plots (only if enabled)
            if create_3d_plots:
                print(f"Creating 3D plots for {col_name} ({dataset_name})...")
                scatter_dir = create_3d_spatial_plots(col_predictions, col_targets, coords_data,
                                                     col_dir, variable_name=col_name)
                
                # 5. Create video from 3D plots
                video_path = os.path.join(col_dir, f'{col_name}_3d_scatter_plots_video.mp4')
                create_video_from_images(scatter_dir, video_path)
            else:
                print(f"Skipping 3D plots for {col_name} ({dataset_name}) (disabled)")
        
        print(f"Visualizations for {dataset_name} saved to: {dataset_dir}")
    
    print("\nPer-column visualizations complete!")
