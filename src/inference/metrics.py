"""
Metrics computation utilities for model evaluation.

Provides functions for:
- Computing performance metrics (KGE, R², Relative L2 Error)
- Denormalizing predictions
- Saving metrics to files
"""

import os
import numpy as np
from sklearn.metrics import r2_score


def compute_kge(simulations, observations):
    """
    Compute Kling-Gupta Efficiency (KGE) and its components.
    
    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    where:
        r = correlation coefficient
        alpha = std(sim) / std(obs)  (variability ratio)
        beta = mean(sim) / mean(obs)  (bias ratio)
    
    Args:
        simulations (np.ndarray): Simulated/predicted values (1D array)
        observations (np.ndarray): Observed/target values (1D array)
        
    Returns:
        dict: Dictionary with kge, r, alpha, beta
    """
    # Ensure inputs are 1D
    sim_flat = simulations.flatten()
    obs_flat = observations.flatten()
    
    # Correlation coefficient (r)
    r = np.corrcoef(sim_flat, obs_flat)[0, 1]
    
    # Variability ratio (alpha)
    alpha = np.std(sim_flat) / (np.std(obs_flat) + 1e-10)
    
    # Bias ratio (beta)
    beta = np.mean(sim_flat) / (np.mean(obs_flat) + 1e-10)
    
    # KGE
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return {
        'kge': kge,
        'r': r,
        'alpha': alpha,
        'beta': beta
    }


def denormalize_observations(normalized_data, obs_transform, target_col_indices):
    """
    Denormalize observations from normalized scale back to original scale.
    
    Args:
        normalized_data: Array of shape [N_samples, N_points, output_window_size, n_target_cols]
        obs_transform: Normalize transform object with mean and std attributes
        target_col_indices: List of target column indices
        
    Returns:
        Array in original scale with same shape
    """
    # Get mean and std for target columns
    mean_all = obs_transform.mean
    std_all = obs_transform.std
    
    mean = mean_all[target_col_indices]
    std = std_all[target_col_indices]
    
    # Convert to numpy if they are tensors
    import torch
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(std, torch.Tensor):
        std = std.cpu().numpy()
    
    # Reshape mean and std for broadcasting
    # Shape: [1, 1, 1, n_target_cols]
    mean = mean.reshape(1, 1, 1, -1)
    std = std.reshape(1, 1, 1, -1)
    
    # Denormalize: data = normalized * std + mean
    denormalized = normalized_data * std + mean
    
    return denormalized


def compute_metrics(results_dict, target_cols, target_col_indices, obs_transform):
    """
    Compute relative L2 error, R² score, and KGE for each target column on train and val sets.
    
    All metrics are computed on denormalized (original scale) data.
    
    Args:
        results_dict: Dictionary containing train and val predictions and targets
        target_cols: List of target column names
        target_col_indices: List of target column indices
        obs_transform: Normalize transform object for denormalization
        
    Returns:
        Dictionary containing metrics for each dataset and target column
    """
    metrics = {}
    
    for dataset_name in ['train', 'val']:
        # Get normalized data
        predictions_norm = results_dict[dataset_name]['predictions']  # [N_samples, N_points, output_window_size, n_target_cols]
        targets_norm = results_dict[dataset_name]['targets']
        
        # Denormalize to original scale
        predictions = denormalize_observations(predictions_norm, obs_transform, target_col_indices)
        targets = denormalize_observations(targets_norm, obs_transform, target_col_indices)
        
        metrics[dataset_name] = {}
        
        # Compute metrics for each target column
        for col_idx, col_name in enumerate(target_cols):
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


def save_metrics(metrics, target_cols, results_dir):
    """
    Save computed metrics to a text file.
    
    Args:
        metrics: Dictionary containing metrics for each dataset and target column
        target_cols: List of target column names
        results_dir: Directory to save metrics file
    """
    metrics_file = os.path.join(results_dir, 'metrics.txt')
    
    with open(metrics_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Model Performance Metrics (Relative & Dimensionless)\n")
        f.write("All metrics computed on denormalized (original scale) data\n")
        f.write("=" * 80 + "\n\n")
        
        for dataset_name in ['train', 'val']:
            f.write(f"\n{dataset_name.upper()} SET\n")
            f.write("-" * 80 + "\n")
            
            for col_name in target_cols:
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
