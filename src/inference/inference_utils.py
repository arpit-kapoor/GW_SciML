"""
Generic inference utilities for neural operator models.

Provides model loading, prediction generation, and configuration setup
that works with any model architecture (GINO, FNO, UNO, etc.).
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.batch_sampler import PatchBatchSampler
from ..training.config import configure_device


def setup_inference_arguments(description, default_base_data_dir, default_results_dir,
                              add_model_specific_args=None):
    """
    Setup argument parser for inference with common parameters.
    
    Args:
        description (str): Description for argument parser
        default_base_data_dir (str): Default base data directory
        default_results_dir (str): Default results directory
        add_model_specific_args (callable, optional): Function to add model-specific args
        
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Model and data paths
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--base-data-dir', type=str, default=default_base_data_dir,
                       help='Base data directory')
    parser.add_argument('--raw-data-subdir', type=str, default='all',
                       help='Raw data subdirectory name')
    parser.add_argument('--patch-data-subdir', type=str, default='filter_patch',
                       help='Patch data subdirectory name')
    parser.add_argument('--results-dir', type=str, default=default_results_dir,
                       help='Directory to save predictions and plots')
    
    # Model parameters
    parser.add_argument('--target-col', type=str, default='head',
                       help='Target observation column name')
    parser.add_argument('--input-window-size', type=int, default=10,
                       help='Input sequence length')
    parser.add_argument('--output-window-size', type=int, default=10,
                       help='Output sequence length')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    
    # Resolution parameters for testing at different spatial resolutions
    parser.add_argument('--resolution-ratio', type=float, default=1.0,
                       help='Ratio of nodes to keep in each patch (0 < ratio <= 1.0). Default is 1.0 (no subsampling)')
    parser.add_argument('--resolution-seed', type=int, default=42,
                       help='Random seed for reproducible subsampling. Default is 42')
    
    # Output control
    parser.add_argument('--metrics-only', action='store_true', default=False,
                       help='Only save metrics and metadata (no arrays or plots) to save disk space')
    parser.add_argument('--create-3d-plots', action='store_true', default=False,
                       help='Create 3D scatter plots and videos (disabled by default to save time/storage)')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    
    # Add model-specific arguments if provided
    if add_model_specific_args is not None:
        parser = add_model_specific_args(parser)
    
    args = parser.parse_args()
    
    # Construct full paths
    args.raw_data_dir = f"{args.base_data_dir}/{args.raw_data_subdir}"
    args.patch_data_dir = f"{args.base_data_dir}/{args.patch_data_subdir}"
    
    # Configure device
    args = configure_device(args)
    
    print(f"Model path: {args.model_path}")
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Patch data directory: {args.patch_data_dir}")
    
    return args


def load_checkpoint(checkpoint_path, device='cpu'):
    """
    Load checkpoint file and extract components.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to map checkpoint to
        
    Returns:
        dict: Checkpoint dictionary containing model_state_dict, args, etc.
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Print checkpoint info
    if 'args' in checkpoint:
        print("\nCheckpoint configuration found")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")
    if 'train_losses' in checkpoint and len(checkpoint['train_losses']) > 0:
        print(f"Final train loss: {checkpoint['train_losses'][-1]:.4f}")
    if 'val_losses' in checkpoint and len(checkpoint['val_losses']) > 0:
        print(f"Final val loss: {checkpoint['val_losses'][-1]:.4f}")
    
    return checkpoint


def create_model_from_checkpoint(checkpoint, model_factory, device='cpu'):
    """
    Create and initialize model from checkpoint using provided factory function.
    
    Args:
        checkpoint (dict): Checkpoint dictionary
        model_factory (callable): Function that takes (checkpoint, device) and returns model
        device (str): Device to load model on
        
    Returns:
        torch.nn.Module: Initialized model in eval mode
    """
    # Use factory function to create model with checkpoint config
    model = model_factory(checkpoint, device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully and set to eval mode")
    return model


def generate_predictions(model, dataset, args, dataset_name='dataset',
                        forward_fn=None, collate_fn=None):
    """
    Generate predictions for an entire dataset.
    
    Args:
        model (torch.nn.Module): Model to use for prediction
        dataset: PyTorch dataset
        args (argparse.Namespace): Arguments with batch_size and device
        dataset_name (str): Name for logging
        forward_fn (callable, optional): Custom forward function(model, batch, args)
        collate_fn (callable, optional): Custom collate function
        
    Returns:
        dict: Dictionary with 'predictions', 'targets', 'coords', 'metadata' (normalized)
    """
    print(f"\nGenerating predictions for {dataset_name} dataset...")
    
    # Create sampler and data loader
    sampler = PatchBatchSampler(
        dataset,
        batch_size=args.batch_size,
        shuffle_within_batches=False,
        shuffle_patches=False,
        seed=None
    )
    
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    
    all_predictions = {}
    all_targets = {}
    all_coords = {}
    all_patch_metadata = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Processing {dataset_name}")):
            # Use custom forward function if provided, otherwise use default
            if forward_fn is not None:
                outputs, core_output, core_target, core_coords = forward_fn(model, batch, args)
            else:
                # Default forward pass (assumes GINO-style interface)
                outputs, core_output, core_target, core_coords = _default_forward(model, batch, args)
            
            patch_id = batch['patch_id']
            
            # Store results per patch
            if patch_id not in all_predictions:
                all_predictions[patch_id] = []
                all_targets[patch_id] = []
                all_coords[patch_id] = []
            
            all_predictions[patch_id].append(core_output)
            all_targets[patch_id].append(core_target)
            all_coords[patch_id].append(core_coords)
            
            # Store metadata
            batch_size = core_output.shape[0]
            for i in range(batch_size):
                all_patch_metadata.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'patch_id': patch_id,
                    'dataset': dataset_name,
                    'core_len': batch.get('core_len', core_output.shape[1])
                })
    
    # Concatenate results per patch
    for patch_id in all_predictions.keys():
        all_predictions[patch_id] = np.concatenate(all_predictions[patch_id], axis=0)
        all_targets[patch_id] = np.concatenate(all_targets[patch_id], axis=0)
        all_coords[patch_id] = np.concatenate(all_coords[patch_id], axis=0)
    
    # Concatenate all patches
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


def _default_forward(model, batch, args):
    """
    Default forward pass for GINO-style models.
    
    Args:
        model: Model to run forward pass
        batch (dict): Batch dictionary
        args: Arguments with device
        
    Returns:
        tuple: (outputs, core_output, core_target, core_coords)
    """
    # Move data to device
    point_coords = batch['point_coords'].to(args.device).float()
    latent_queries = batch['latent_queries'].to(args.device).float()
    x = batch['x'].to(args.device).float()
    y = batch['y'].to(args.device).float()
    
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
    
    # Repeat coords for batch dimension
    core_coords = np.repeat(
        np.expand_dims(core_coords, axis=0),
        core_output.shape[0],
        axis=0
    )
    
    return outputs, core_output, core_target, core_coords
