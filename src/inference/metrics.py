"""
Metrics computation utilities for model evaluation.

Provides functions for:
- Computing performance metrics (KGE, R², Relative L2 Error)
- Denormalizing predictions
- Saving metrics to files
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def compute_normalized_mae(predictions, observations):
    """
    Compute Mean Absolute Error normalized by standard deviation of observations.
    
    NMAE = MAE / std(observations)
    
    Args:
        predictions (np.ndarray): Predicted values (1D array)
        observations (np.ndarray): Observed/target values (1D array)
        
    Returns:
        float: Normalized MAE
    """
    # Ensure inputs are 1D
    pred_flat = predictions.flatten()
    obs_flat = observations.flatten()
    
    # Compute MAE
    mae = np.mean(np.abs(pred_flat - obs_flat))
    
    # Normalize by std of observations
    obs_std = np.std(obs_flat)
    
    # Avoid division by zero
    nmae = mae / (obs_std + 1e-10)
    
    return nmae


def compute_r2_score(predictions, observations):
    """
    Compute R² score (coefficient of determination).
    
    This function is kept for potential future use but not called by default.
    
    Args:
        predictions (np.ndarray): Predicted values
        observations (np.ndarray): Observed/target values
        
    Returns:
        float: R² score
    """
    pred_flat = predictions.flatten()
    obs_flat = observations.flatten()
    return r2_score(obs_flat, pred_flat)


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


def compute_metrics(results_dict, target_cols, target_col_indices, obs_transform=None):
    """
    Compute relative L2 error, R² score, and KGE for each target column on train and val sets.
    
    All metrics are computed on denormalized (original scale) data.
    
    Args:
        results_dict: Dictionary containing train and val predictions and targets
        target_cols: List of target column names
        target_col_indices: List of target column indices
        obs_transform: Normalize transform object for denormalization (None if data already denormalized)
        
    Returns:
        Dictionary containing metrics for each dataset and target column
    """
    metrics = {}
    
    for dataset_name in ['train', 'val']:
        # Get data (may be normalized or denormalized)
        predictions_data = results_dict[dataset_name]['predictions']  # [N_samples, N_points, output_window_size, n_target_cols]
        targets_data = results_dict[dataset_name]['targets']
        
        # Denormalize to original scale if transform provided
        if obs_transform is not None:
            predictions = denormalize_observations(predictions_data, obs_transform, target_col_indices)
            targets = denormalize_observations(targets_data, obs_transform, target_col_indices)
        else:
            # Data is already denormalized
            predictions = predictions_data
            targets = targets_data
        
        metrics[dataset_name] = {}
        
        # Compute metrics for each target column
        for col_idx, col_name in enumerate(target_cols):
            print(f"Computing metrics for {col_name} ({dataset_name})...")
            col_predictions = predictions[:, :, :, col_idx]  # [N_samples, N_points, output_window_size]
            col_targets = targets[:, :, :, col_idx]
            
            # Flatten for metric computation
            col_predictions_flat = col_predictions.flatten()
            col_targets_flat = col_targets.flatten()
            
            # 1. Normalized MAE
            nmae = compute_normalized_mae(col_predictions_flat, col_targets_flat)
            
            # 2. Relative L2 Error (using numpy implementation)
            # Relative L2 = ||pred - target||_2 / ||target||_2
            diff = col_predictions_flat - col_targets_flat
            l2_error = np.linalg.norm(diff)
            l2_norm_target = np.linalg.norm(col_targets_flat)
            rel_l2_error = l2_error / (l2_norm_target + 1e-10)
            
            # 3. Kling-Gupta Efficiency (KGE)
            kge_results = compute_kge(col_predictions_flat, col_targets_flat)
            
            metrics[dataset_name][col_name] = {
                'nmae': nmae,
                'rel_l2_error': rel_l2_error,
                'kge': kge_results['kge'],
                'kge_r': kge_results['r'],
                'kge_alpha': kge_results['alpha'],
                'kge_beta': kge_results['beta']
            }
    
    return metrics


def save_metrics(metrics, target_cols, results_dir):
    """
    Save computed metrics to a CSV file in long format.
    
    Format (Option B - Long Format):
    metric_name | variable | train | val
    
    Args:
        metrics: Dictionary containing metrics for each dataset and target column
        target_cols: List of target column names
        results_dir: Directory to save metrics file
    """
    metrics_file = os.path.join(results_dir, 'metrics.csv')
    
    # Build list of rows for DataFrame
    rows = []
    
    # Metrics to extract (in desired order)
    metric_names = ['nmae', 'rel_l2_error', 'kge', 'kge_r', 'kge_alpha', 'kge_beta']
    
    for metric_name in metric_names:
        for col_name in target_cols:
            row = {
                'metric_name': metric_name,
                'variable': col_name,
                'train': metrics['train'][col_name][metric_name],
                'val': metrics['val'][col_name][metric_name]
            }
            rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(metrics_file, index=False, float_format='%.6f')
    
    print(f"Saved metrics to {metrics_file}")
    print("\nMetrics Preview:")
    print(df.head(10).to_string(index=False))
