"""
Results management for inference outputs.

Handles saving predictions, metadata, and organizing output directories.
"""

import os
import pickle
import datetime as dt
import numpy as np
from .metrics import denormalize_observations


def create_results_directory(base_dir, experiment_name=None):
    """
    Create timestamped results directory.
    
    Args:
        base_dir (str): Base directory for results
        experiment_name (str, optional): Experiment name prefix
        
    Returns:
        str: Path to created results directory
    """
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if experiment_name:
        dir_name = f'{experiment_name}_{timestamp}'
    else:
        dir_name = f'predictions_{timestamp}'
    
    results_dir = os.path.join(base_dir, dir_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Created results directory: {results_dir}")
    return results_dir


def save_predictions_to_disk(predictions, targets, coords, output_dir, prefix=''):
    """
    Save predictions, targets, and coordinates to disk.
    
    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Ground truth targets
        coords (np.ndarray or dict): Coordinate data
        output_dir (str): Directory to save files
        prefix (str): Filename prefix (e.g., 'train_', 'val_')
    """
    # Save numpy arrays
    np.save(os.path.join(output_dir, f'{prefix}predictions.npy'), predictions)
    np.save(os.path.join(output_dir, f'{prefix}targets.npy'), targets)
    
    # Save coordinates (can be array or dict)
    coords_path = os.path.join(output_dir, f'{prefix}coords.pkl')
    with open(coords_path, 'wb') as f:
        pickle.dump(coords, f)
    
    print(f"Saved {prefix}predictions, targets, and coords to {output_dir}")


def save_metadata(metadata, output_dir, filename='metadata.pkl'):
    """
    Save metadata dictionary to pickle file.
    
    Args:
        metadata (dict): Dictionary containing metadata
        output_dir (str): Directory to save file
        filename (str): Output filename
    """
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Saved metadata to {filepath}")


def organize_and_save_results(results_dict, args):
    """
    Orchestrate saving all results including train/val predictions and metadata.
    
    Args:
        results_dict (dict): Dictionary with 'train' and 'val' results (already denormalized)
        args (argparse.Namespace): Arguments containing configuration
    """
    print("Organizing and saving results...")
    
    # Save train results
    if 'train' in results_dict:
        save_predictions_to_disk(
            results_dict['train']['predictions'],
            results_dict['train']['targets'],
            results_dict['train']['coords'],
            args.results_dir,
            prefix='train_'
        )
    
    # Save validation results
    if 'val' in results_dict:
        save_predictions_to_disk(
            results_dict['val']['predictions'],
            results_dict['val']['targets'],
            results_dict['val']['coords'],
            args.results_dir,
            prefix='val_'
        )
    
    # Compile and save metadata
    metadata = {
        'args': vars(args),
        'timestamp': dt.datetime.now().isoformat(),
    }
    
    if 'train' in results_dict and 'metadata' in results_dict['train']:
        metadata['train_metadata'] = results_dict['train']['metadata']
    
    if 'val' in results_dict and 'metadata' in results_dict['val']:
        metadata['val_metadata'] = results_dict['val']['metadata']
    
    save_metadata(metadata, args.results_dir)
    
    print(f"All results saved to: {args.results_dir}")
