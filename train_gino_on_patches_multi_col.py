"""
Train GINO on variable-density groundwater patches with multi-column support.

This script extends the original train_gino_on_patches.py to support training on
multiple target variables simultaneously. Input and output sequences concatenate
values from all target columns along the last dimension.

Key differences from single-column version:
- Accepts multiple target columns via --target-cols argument
- Input channels = input_window_size * n_target_cols
- Output channels = output_window_size * n_target_cols
- Sequences concatenate multiple variables along the last dimension

Example:
For 2 target columns ['mass_concentration', 'head'] with window size 5:
- Single column: [N_points, 5] per variable
- Multi-column: [N_points, 10] (5 timesteps × 2 variables concatenated)

Tensor shapes (per batch):
- point_coords: [N_points, 3]
- latent_queries: [Qx, Qy, Qz, 3]
- x (inputs): [B, N_points, input_window_size * n_target_cols]
- y (targets): [B, N_points, output_window_size * n_target_cols]
- outputs: [B, N_points, output_window_size * n_target_cols]

Loss is computed only on core points to avoid boundary artifacts from ghost points.

Resume Training:
- Use --resume-from path/to/checkpoint.pth to resume from a specific checkpoint
- Checkpoints are automatically saved every N epochs (configurable with --save-checkpoint-every)
- Training state (model, optimizer, scheduler, losses) is fully restored
- Results directory is inferred from checkpoint path when resuming
"""

import argparse
import os
import json

import pandas as pd
import torch
import datetime as dt
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.transform import Normalize
from src.data.patch_dataset_multi_col import GWPatchDatasetMultiCol
from src.data.batch_sampler import PatchBatchSampler
from src.models.neuralop.gino import GINO
from src.models.neuralop.losses import LpLoss, H1Loss

def setup_arguments():
    """Parse command line arguments for data, model, and training.

    Notable arguments:
    - --target-cols: multiple observation fields to model (mapped to `target_col_indices`)
    - --input-window-size / --output-window-size: sliding window lengths
    - --batch-size: number of sequences per training step (from the same patch)
    
    Returns:
        argparse.Namespace: Parsed arguments with computed paths and configurations
    """
    parser = argparse.ArgumentParser(description='Train GINO model on groundwater patches with multi-column support')
    
    # Data directories
    parser.add_argument('--base-data-dir', type=str, 
                       default='/Users/arpitkapoor/Library/CloudStorage/OneDrive-UNSW/Shared/Projects/01_PhD/05_groundwater/data/FEFLOW/variable_density',
                       help='Base data directory')
    parser.add_argument('--raw-data-subdir', type=str, default='all',
                       help='Raw data subdirectory name')
    parser.add_argument('--patch-data-subdir', type=str, default='filter_patch',
                       help='Patch data subdirectory name')
    parser.add_argument('--results-dir', type=str, default='/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO',
                       help='Directory to save trained models and results')

    # Target observation columns (multiple)
    parser.add_argument('--target-cols', type=str, nargs='+', 
                       default=['mass_concentration', 'head'],
                       help='List of target observation column names (e.g., mass_concentration head pressure)')
    
    # Sequence lengths
    parser.add_argument('--input-window-size', type=int, default=10,
                       help='Number of time steps in each input sequence')
    parser.add_argument('--output-window-size', type=int, default=10,
                       help='Number of time steps in each output sequence')
    
    # Model parameters
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='Learning rate for optimizer')
    parser.add_argument('--lr-gamma', type=float, default=0.98,
                       help='Exponential learning rate decay factor')
    parser.add_argument('--lr-scheduler-interval', type=int, default=10,
                       help='Number of epochs between learning rate scheduler updates')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                       help='Gradient clipping norm value (0 to disable)')
    parser.add_argument('--scheduler-type', type=str, default='exponential', 
                       choices=['exponential', 'cosine'],
                       help='Type of learning rate scheduler to use')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    # Shuffling parameters
    parser.add_argument('--shuffle-within-batches', action='store_true', default=True,
                       help='Shuffle examples within each batch (default: True)')
    parser.add_argument('--no-shuffle-within-batches', dest='shuffle_within_batches', action='store_false',
                       help='Disable shuffling examples within batches')
    parser.add_argument('--shuffle-patches', action='store_true', default=True,
                       help='Shuffle the order of patches between epochs (default: True)')
    parser.add_argument('--no-shuffle-patches', dest='shuffle_patches', action='store_false',
                       help='Disable shuffling patch order between epochs')
    
    # Resume training parameters
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory to save checkpoints (defaults to results_dir/checkpoints)')
    parser.add_argument('--save-checkpoint-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (cuda, cpu, or auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Construct full paths and add them to args
    args.raw_data_dir = os.path.join(args.base_data_dir, args.raw_data_subdir)
    args.patch_data_dir = os.path.join(args.base_data_dir, args.patch_data_subdir)

    # Create results directory with timestamp (unless resuming)
    if args.resume_from is None:
        # Create new timestamped directory for fresh training
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.results_dir = os.path.join(args.results_dir, f'gino_multi_{timestamp}')
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"Starting fresh training, created results directory: {args.results_dir}")
    else:
        # When resuming, extract the actual run directory from checkpoint path
        checkpoint_path = os.path.abspath(args.resume_from)
        
        # Extract the actual results directory (specific run dir) from checkpoint path
        if 'checkpoints' in checkpoint_path:
            # Path like: /path/to/exp_name/gino_timestamp/checkpoints/checkpoint.pth
            actual_results_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        else:
            # Path like: /path/to/exp_name/gino_timestamp/checkpoint.pth
            actual_results_dir = os.path.dirname(checkpoint_path)
        
        # Ensure the checkpoint exists and is accessible
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Use the actual run directory extracted from checkpoint path
        # This ensures we continue writing to the same run directory
        args.results_dir = actual_results_dir
        print(f"Resuming training, using run directory: {args.results_dir}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Validate that we're resuming within the expected experiment structure
        experiment_dir = os.path.dirname(actual_results_dir)
        run_dir_name = os.path.basename(actual_results_dir)
        print(f"Experiment directory: {experiment_dir}")
        print(f"Run directory: {run_dir_name}")
    
    # Set up checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.results_dir, 'checkpoints')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Define model parameters
    args = define_model_parameters(args)

    # Configure target observation column indices
    args = configure_target_col_indices(args)

    # Configure device
    args = configure_device(args)
    
    # Print data directories for verification
    print(f"Base data directory: {args.base_data_dir}")
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Patch filtered data directory: {args.patch_data_dir}")
    print(f"Results directory: {args.results_dir}")
    
    return args

def define_model_parameters(args):
    """Define model parameters for GINO architecture with multi-column support.

    Notes:
    - `in_gno_out_channels` is set to input_window_size * n_target_cols
    - `out_channels` is set to output_window_size * n_target_cols
    - This allows the model to handle concatenated multi-variable sequences
    
    Args:
        args: Argument namespace to modify with model parameters
        
    Returns:
        argparse.Namespace: Modified args with model parameters added
    """
    # Coordinate dimensions (3D: x, y, z)
    args.coord_dim = 3
    
    # Number of target columns
    args.n_target_cols = len(args.target_cols)
    
    # Radius for neighbor search in GNO operations
    args.gno_radius = 0.15
    
    # Output channels of the input GNO block 
    # Multiplied by n_target_cols to handle concatenated variables
    args.in_gno_out_channels = args.input_window_size * args.n_target_cols
    
    # MLP layer dimensions for input GNO channel processing
    args.in_gno_channel_mlp_layers = [32, 64, 32]
    
    # FNO (Fourier Neural Operator) configuration
    args.fno_n_layers = 4
    args.fno_n_modes = (12, 12, 8)  # 3D Fourier modes
    args.fno_hidden_channels = 64
    args.lifting_channels = 64
    
    # Output GNO configuration
    args.out_gno_channel_mlp_layers = [32, 64, 32]
    args.projection_channel_ratio = 2
    
    # Predict the full output window per point for all target columns
    # Multiplied by n_target_cols to handle concatenated variables
    args.out_channels = args.output_window_size * args.n_target_cols
    
    # Latent query grid dimensions
    args.latent_query_dims = (32, 32, 24)
    
    return args

def configure_device(args):
    """Configure computation device: cpu, cuda, or mps.
    
    Args:
        args: Argument namespace with device preference
        
    Returns:
        argparse.Namespace: Modified args with device configured
        
    Raises:
        ValueError: If requested device is not available
    """
    if args.device == 'auto':
        # Auto-detect best available device
        args.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Please use 'auto' or specify 'cpu'.")
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available. Please use 'auto' or specify 'cpu'.")
    return args

def configure_target_col_indices(args):
    """Configure multiple target observation column indices from names.

    Maps human-readable column names to array indices for the observation data.
    
    Args:
        args: Argument namespace with target column names
        
    Returns:
        argparse.Namespace: Modified args with target_col_indices added
    """
    # Mapping from column names to indices in the observation array
    names_to_idx = {
        'mass_concentration': 0,
        'head': 1,
        'pressure': 2
    }
    
    # Convert target column names to indices
    args.target_col_indices = [names_to_idx[col] for col in args.target_cols]
    
    print(f"Target columns: {args.target_cols}")
    print(f"Target column indices: {args.target_col_indices}")
    
    return args

def calculate_coord_transform(raw_data_dir):
    """Calculate mean and std of coordinates and create coordinate transform.

    Uses a representative CSV (0000.csv) to derive normalization stats for
    spatial coordinates. This ensures coordinates are zero-centered with unit variance.
    
    Args:
        raw_data_dir: Directory containing raw CSV files
        
    Returns:
        Normalize: Transform object for coordinate normalization
    """
    # Read representative data file to compute statistics
    df = pd.read_csv(os.path.join(raw_data_dir, '0000.csv'))

    # Calculate mean and std of coordinates (X, Y, Z columns)
    coord_mean = df[['X', 'Y', 'Z']].mean().values
    coord_std = df[['X', 'Y', 'Z']].std().values

    # Print normalization statistics for debugging
    print(f"Coordinate mean: {coord_mean}")
    print(f"Coordinate std: {coord_std}")

    # Create coordinate transform
    coord_transform = Normalize(mean=coord_mean, std=coord_std)

    # Clean up memory
    del df
    return coord_transform

def calculate_obs_transform(raw_data_dir, 
                            target_obs_cols=['mass_concentration', 'head', 'pressure']):
    """Calculate mean and std of output variables and create observation transform.

    Normalizes all listed columns for consistency. The dataset will then select
    multiple targets by `target_col_indices` before sequencing.
    
    Args:
        raw_data_dir: Directory containing raw CSV files
        target_obs_cols: List of observation column names to normalize
        
    Returns:
        Normalize: Transform object for observation normalization
    """
    # Read representative data file
    df = pd.read_csv(os.path.join(raw_data_dir, '0000.csv'))

    # Define output columns to normalize
    obs_cols = target_obs_cols

    # Calculate normalization statistics
    obs_mean = df[obs_cols].mean().values
    obs_std = df[obs_cols].std().values

    # Print normalization statistics for debugging
    print(f"Output mean: {obs_mean}")
    print(f"Output std: {obs_std}")

    # Define output transform
    obs_transform = Normalize(mean=obs_mean, std=obs_std)

    # Clean up memory
    del df
    return obs_transform

def create_patch_datasets(patch_data_dir, coord_transform, obs_transform, **kwargs):
    """Create train/val `GWPatchDatasetMultiCol` with normalization and sequencing.

    Creates datasets that apply coordinate and observation normalization,
    then generate sliding window sequences for training and validation with
    multiple target columns concatenated.
    
    Args:
        patch_data_dir: Directory containing patch data files
        coord_transform: Normalization transform for coordinates
        obs_transform: Normalization transform for observations
        **kwargs: Additional arguments including window sizes and target column indices
        
    Returns:
        tuple: (train_dataset, validation_dataset)
    """
    # Create training dataset
    train_ds = GWPatchDatasetMultiCol(
        data_path=patch_data_dir,
        dataset='train', 
        coord_transform=coord_transform, 
        obs_transform=obs_transform,
        input_window_size=kwargs.get('input_window_size', 10),
        output_window_size=kwargs.get('output_window_size', 10),
        target_col_indices=kwargs.get('target_col_indices', None),
    )
    
    # Create validation dataset
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


def define_ginos_model(args):
    """Define GINO model with 3D coordinates and multi-column sequence projection head.
    
    Initializes a Graph-Informed Neural Operator (GINO) with parameters
    configured for 3D groundwater modeling with temporal sequences across
    multiple target variables.
    
    Args:
        args: Argument namespace containing model hyperparameters
        
    Returns:
        GINO: Configured GINO model moved to specified device
    """
    model = GINO(
        # Input GNO configuration
        in_gno_coord_dim=args.coord_dim,
        in_gno_radius=args.gno_radius,
        in_gno_out_channels=args.in_gno_out_channels,
        in_gno_channel_mlp_layers=args.in_gno_channel_mlp_layers,
        
        # FNO configuration
        fno_n_layers=args.fno_n_layers,
        fno_n_modes=args.fno_n_modes,  # 3D modes
        fno_hidden_channels=args.fno_hidden_channels,
        lifting_channels=args.lifting_channels,
        
        # Output GNO configuration
        out_gno_coord_dim=args.coord_dim,
        out_gno_radius=args.gno_radius,
        out_gno_channel_mlp_layers=args.out_gno_channel_mlp_layers,
        projection_channel_ratio=args.projection_channel_ratio,
        out_channels=args.out_channels,
    ).to(args.device)
    return model

def _make_collate_fn(args):
    """Create a collate function that batches samples from the same patch.

    The sampler ensures a batch contains indices from a single `patch_id`.
    We build one point cloud per batch (core+ghost), a latent grid over its
    bounding box, and then stack input/output sequences along the batch dim.
    
    For multi-column support, the sequences are already concatenated in the dataset.
    
    Args:
        args: Argument namespace containing device and latent grid dimensions
        
    Returns:
        function: Collate function for DataLoader
    """
    def collate_fn(batch_samples):
        """Collate function that combines samples into a batch.
        
        Args:
            batch_samples: List of sample dictionaries from the same patch
            
        Returns:
            dict: Batch dictionary with combined point cloud and sequences
        """
        # All samples in the batch come from the same patch (by sampler design)
        core_coords = batch_samples[0]['core_coords']
        ghost_coords = batch_samples[0]['ghost_coords']

        # Single point cloud per batch: [N_core+N_ghost, 3]
        # Concatenate core and ghost points to form complete spatial domain
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
            # Note: sequences are already concatenated across target columns in the dataset
            sample_input = torch.concat([sample['core_in'], sample['ghost_in']], dim=0).float().unsqueeze(0)
            sample_output = torch.concat([sample['core_out'], sample['ghost_out']], dim=0).float().unsqueeze(0)
            x_list.append(sample_input)
            y_list.append(sample_output)

        # Stack all sequences into batch tensors
        x = torch.cat(x_list, dim=0)  # [B, N_points, input_window_size * n_target_cols]
        y = torch.cat(y_list, dim=0)  # [B, N_points, output_window_size * n_target_cols]

        # Return batch dictionary
        batch = {
            'point_coords': point_coords,      # [N_points, 3]
            'latent_queries': latent_queries,  # [Qx, Qy, Qz, 3]
            'x': x,                           # [B, N_points, input_window_size * n_target_cols]
            'y': y,                           # [B, N_points, output_window_size * n_target_cols]
            'core_len': len(core_coords),     # Number of core points (for loss masking)
        }
        return batch
    return collate_fn
 

def evaluate_model_on_patches(val_loader, model, loss_fn, args):
    """Evaluate model on validation loader.

    Computes relative L2 loss on core points across validation batches.
    Ghost points are excluded from loss computation to avoid boundary artifacts.
    
    Args:
        val_loader: DataLoader for validation data
        model: GINO model to evaluate
        loss_fn: Loss function (typically LpLoss)
        args: Argument namespace containing device info
        
    Returns:
        float: Average validation loss across all batches
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch in val_loader:
            # Move batch data to device
            point_coords = batch['point_coords'].to(args.device).float()
            latent_queries = batch['latent_queries'].to(args.device).float()
            x = batch['x'].to(args.device).float()
            y = batch['y'].to(args.device).float()

            # Forward pass through model
            outputs = model(
                input_geom=point_coords,
                latent_queries=latent_queries,
                x=x,
                output_queries=point_coords,
            )

            # Extract core points only for loss computation
            core_len = batch['core_len']
            core_output = outputs[:, :core_len]
            core_target = y[:, :core_len]

            # Compute loss on core points only
            loss = loss_fn(core_output, core_target)
            
            # Dynamically infer batch size for loss computation
            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    model.train()  # Return model to training mode
    return total_loss / max(total_samples, 1)


def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, args, filename=None):
    """Save a training checkpoint to enable resuming training.
    
    Args:
        model: GINO model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        epoch: Current epoch number (0-indexed)
        train_losses: List of training losses up to current epoch
        val_losses: List of validation losses up to current epoch
        args: Argument namespace containing checkpoint directory
        filename: Optional custom filename for checkpoint
    """
    if filename is None:
        filename = f'checkpoint_epoch_{epoch:04d}.pth'
    
    checkpoint_path = os.path.join(args.checkpoint_dir, filename)
    
    # Save all training state
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'args': args,  # Save training configuration for compatibility checking
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Also save as latest checkpoint for easy resuming
    latest_path = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    print(f"Latest checkpoint saved: {latest_path}")
    
    # Save accumulated loss history for continuous training curves
    # Save both in results_dir and next to the checkpoint for better resumption
    accumulated_train, accumulated_val = get_accumulated_losses(train_losses, val_losses, args, checkpoint_path)
    
    # Also save loss history next to the checkpoint file
    checkpoint_loss_history = os.path.join(os.path.dirname(checkpoint_path), 'loss_history.json')
    loss_history = {
        'train_losses': accumulated_train,
        'val_losses': accumulated_val,
        'last_updated': dt.datetime.now().isoformat(),
        'total_epochs': len(accumulated_train),
        'checkpoint_file': os.path.basename(checkpoint_path)
    }
    with open(checkpoint_loss_history, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"Loss history saved with checkpoint: {checkpoint_loss_history}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, args):
    """Load a training checkpoint to resume training.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: GINO model to load state into
        optimizer: Optimizer to load state into  
        scheduler: Learning rate scheduler to load state into
        args: Current argument namespace for compatibility checking
        
    Returns:
        tuple: (start_epoch, train_losses, val_losses)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # Validate compatibility
    saved_args = checkpoint['args']
    _validate_checkpoint_compatibility(saved_args, args)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state loaded successfully")
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Optimizer state loaded successfully")
    
    # Load scheduler state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print("Scheduler state loaded successfully")
    
    # Get training progress
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    
    print(f"Resuming training from epoch {start_epoch}")
    print(f"Loaded {len(train_losses)} training losses and {len(val_losses)} validation losses")
    
    return start_epoch, train_losses, val_losses


def _validate_checkpoint_compatibility(saved_args, current_args):
    """Validate that checkpoint is compatible with current training setup.
    
    Args:
        saved_args: Arguments from saved checkpoint
        current_args: Current training arguments
        
    Raises:
        ValueError: If incompatible configuration detected
    """
    # Check critical model parameters
    critical_params = [
        'coord_dim', 'gno_radius', 'in_gno_out_channels', 'fno_n_layers', 
        'fno_n_modes', 'fno_hidden_channels', 'out_channels', 'latent_query_dims',
        'input_window_size', 'output_window_size', 'target_col_indices', 'n_target_cols'
    ]
    
    for param in critical_params:
        if hasattr(saved_args, param) and hasattr(current_args, param):
            saved_val = getattr(saved_args, param)
            current_val = getattr(current_args, param)
            if saved_val != current_val:
                raise ValueError(f"Incompatible {param}: saved={saved_val}, current={current_val}")
    
    print("Checkpoint compatibility validated successfully")


def save_loss_history(accumulated_train_losses, accumulated_val_losses, args):
    """Save training and validation loss history to a persistent JSON file.
    
    This function saves the complete accumulated history of losses across all training sessions.
    
    Args:
        accumulated_train_losses: Complete list of accumulated training losses
        accumulated_val_losses: Complete list of accumulated validation losses
        args: Argument namespace containing results directory
    """
    loss_history_path = os.path.join(args.results_dir, 'loss_history.json')
    
    # Create loss history dictionary
    loss_history = {
        'train_losses': accumulated_train_losses,
        'val_losses': accumulated_val_losses,
        'last_updated': dt.datetime.now().isoformat(),
        'total_epochs': len(accumulated_train_losses)
    }
    
    # Save to JSON file
    with open(loss_history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"Loss history saved to '{loss_history_path}' (total epochs: {len(accumulated_train_losses)})")


def load_loss_history(args, checkpoint_path=None):
    """Load training and validation loss history from persistent JSON file.
    
    When resuming training (checkpoint_path provided), tries to load loss history from:
    1. The checkpoint directory first (most accurate for resuming)
    2. Falls back to results directory if not found
    3. Returns empty history if neither exists
    
    Args:
        args: Argument namespace containing results directory
        checkpoint_path: Optional path to checkpoint file being loaded
        
    Returns:
        dict: Dictionary containing train_losses, val_losses, and metadata
    """
    # First try to load from checkpoint directory if resuming
    if checkpoint_path is not None:
        checkpoint_loss_history = os.path.join(os.path.dirname(checkpoint_path), 'loss_history.json')
        if os.path.exists(checkpoint_loss_history):
            try:
                with open(checkpoint_loss_history, 'r') as f:
                    history = json.load(f)
                print(f"Loaded loss history from checkpoint directory: {checkpoint_loss_history}")
                print(f"Total epochs in history: {history.get('total_epochs', 0)}")
                return history
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load checkpoint loss history from '{checkpoint_loss_history}': {e}")
                # Continue to try results directory
    
    # Try results directory as fallback
    results_loss_history = os.path.join(args.results_dir, 'loss_history.json')
    if os.path.exists(results_loss_history):
        try:
            with open(results_loss_history, 'r') as f:
                history = json.load(f)
            print(f"Loaded loss history from results directory: {results_loss_history}")
            print(f"Total epochs in history: {history.get('total_epochs', 0)}")
            return history
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load results loss history from '{results_loss_history}': {e}")
    
    # Return empty history if no valid history found
    print("No existing loss history found, starting fresh.")
    return {'train_losses': [], 'val_losses': [], 'total_epochs': 0}


def get_accumulated_losses(train_losses, val_losses, args, checkpoint_path=None):
    """Get accumulated losses from all training sessions.
    
    This function loads the persistent loss history and properly merges it with
    the current session's losses to maintain continuity across all training sessions.
    It uses delta-merging to prevent duplication of epochs, whether resuming or in
    a fresh run.
    
    Args:
        train_losses: Current session's training losses
        val_losses: Current session's validation losses  
        args: Argument namespace containing results directory
        checkpoint_path: Optional path to checkpoint file being loaded when resuming
        
    Returns:
        tuple: (accumulated_train_losses, accumulated_val_losses)
    """
    
    # Load existing loss history, prioritizing checkpoint directory when resuming
    existing_history = load_loss_history(args, checkpoint_path)
    existing_train = existing_history.get('train_losses', [])
    existing_val = existing_history.get('val_losses', [])
    
    # Get the number of existing epochs
    existing_epoch_count = len(existing_train)
    current_epoch_count = len(train_losses)
    
    # If we have more epochs than what's saved, append only the new ones
    if current_epoch_count > existing_epoch_count:
        # Extract only the new epochs that aren't in the existing history
        new_train_losses = train_losses[existing_epoch_count:]
        new_val_losses = val_losses[existing_epoch_count:]
        
        # Merge existing with new
        accumulated_train = existing_train + new_train_losses
        accumulated_val = existing_val + new_val_losses
    else:
        # Current session has same or fewer epochs - use existing
        accumulated_train = existing_train
        accumulated_val = existing_val
    
    return accumulated_train, accumulated_val


def plot_training_curves(train_losses, val_losses, args):
    """Plot training and validation loss curves and save to results directory.
    
    This function now plots accumulated losses from all training sessions,
    providing a continuous view of training progress across resume operations.
    
    Args:
        train_losses: List of training losses per epoch (current session)
        val_losses: List of validation losses per epoch (current session)
        args: Argument namespace containing results directory
    """
    # Get accumulated losses from all training sessions
    # Pass checkpoint path if we're resuming training
    checkpoint_path = args.resume_from if hasattr(args, 'resume_from') else None
    accumulated_train, accumulated_val = get_accumulated_losses(train_losses, val_losses, args, checkpoint_path)
    
    # Create the plot with accumulated data
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(accumulated_train) + 1)
    
    plt.plot(epochs, accumulated_train, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, accumulated_val, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    
    # Add visual indicators for resume points if this is a resumed session
    if args.resume_from is not None and len(accumulated_train) > len(train_losses):
        resume_points = []
        
        # Find potential resume points by looking for discontinuities or patterns
        # For now, we'll mark the boundary between existing and new losses
        existing_count = len(accumulated_train) - len(train_losses)
        if existing_count > 0:
            resume_points.append(existing_count)
        
        # Add vertical lines for resume points
        for resume_point in resume_points:
            plt.axvline(x=resume_point + 0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            plt.text(resume_point + 0.5, max(max(accumulated_train), max(accumulated_val)) * 0.9, 
                    'Resume', rotation=90, ha='right', va='top', fontsize=10, alpha=0.7)
    
    plt.title('Training and Validation Loss Over Epochs (Accumulated)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics as text
    if accumulated_train:
        min_train_loss = min(accumulated_train)
        min_val_loss = min(accumulated_val)
        total_epochs = len(accumulated_train)
        
        stats_text = f'Total Epochs: {total_epochs}\nMin Train Loss: {min_train_loss:.4f}\nMin Val Loss: {min_val_loss:.4f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add some styling
    plt.tight_layout()
    
    # Save plot to results directory
    plot_path = os.path.join(args.results_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to '{plot_path}' (total epochs: {len(accumulated_train)})")
    
    # Save the accumulated loss history for future sessions
    save_loss_history(accumulated_train, accumulated_val, args)


def train_gino_on_patches(train_patch_ds, val_patch_ds, model, optimizer, scheduler, args):
    """Train GINO with true batch training using DataLoader and PatchBatchSampler.

    Each training step uses multiple sequences from the same patch to share the
    same point cloud and latent grid, improving neighbor search and cache reuse.
    
    Supports resuming training from checkpoints.
    
    Args:
        train_patch_ds: Training dataset
        val_patch_ds: Validation dataset
        model: GINO model to train
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        args: Argument namespace with training hyperparameters
        
    Returns:
        GINO: Trained model
    """
    # Create training sampler (will reshuffle automatically each epoch)
    train_sampler = PatchBatchSampler(
        train_patch_ds, 
        batch_size=args.batch_size,
        shuffle_within_batches=args.shuffle_within_batches,
        shuffle_patches=args.shuffle_patches,
        seed=args.seed
    )
    
    # Create validation sampler (no shuffling for deterministic evaluation)
    val_sampler = PatchBatchSampler(
        val_patch_ds, 
        batch_size=args.batch_size,
        shuffle_within_batches=False,
        shuffle_patches=False,
        seed=None
    )

    # Custom collate function to handle patch-based batching
    collate_fn = _make_collate_fn(args)

    # Create data loaders
    train_loader = DataLoader(train_patch_ds, batch_sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_patch_ds, batch_sampler=val_sampler, collate_fn=collate_fn)

    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")

    # Use relative L2 loss with appropriate dimensionality
    loss_fn = LpLoss(d=1, p=2, reduce_dims=[0, 1], reductions='mean')

    # Initialize training state
    start_epoch = 0
    train_losses = []
    val_losses = []
    
    # Load checkpoint if resuming training
    if args.resume_from is not None:
        start_epoch, train_losses, val_losses = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, args
        )

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Training Epoch {epoch+1} of {args.epochs}")
        print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Log shuffling configuration for this epoch
        print(f"Epoch {epoch+1} shuffling: within_batches={args.shuffle_within_batches}, patches={args.shuffle_patches}")

        # Track training loss for this epoch
        epoch_train_loss = 0.0
        epoch_train_samples = 0

        # Training step loop
        for step_idx, batch in enumerate(train_loader, start=1):
            # Move batch data to device
            point_coords = batch['point_coords'].to(args.device).float()
            latent_queries = batch['latent_queries'].to(args.device).float()
            x = batch['x'].to(args.device).float()
            y = batch['y'].to(args.device).float()
            
            # Dynamically infer batch size for this batch
            batch_size = x.shape[0]
            n_points = x.shape[1]

            # Forward pass through GINO model
            outputs = model(
                input_geom=point_coords,      # Point cloud coordinates
                latent_queries=latent_queries, # Regular grid for FNO
                x=x,                          # Input sequences (concatenated across target cols)
                output_queries=point_coords,  # Query points (same as input geometry)
            )
            
            # Debug: Print shapes of key variables
            # print(f"DEBUG - Step {step_idx}:")
            # print(f"  point_coords shape: {point_coords.shape}")
            # print(f"  latent_queries shape: {latent_queries.shape}")
            # print(f"  x shape: {x.shape}")
            # print(f"  y shape: {y.shape}")
            # print(f"  outputs shape: {outputs.shape}")

            # Extract core points only for loss computation
            # Ghost points are excluded to avoid boundary artifacts
            core_len = batch['core_len']
            core_output = outputs[:, :core_len]
            core_target = y[:, :core_len]

            # Compute loss on core points only
            loss = loss_fn(core_output, core_target)

            # Accumulate training loss weighted by batch size
            epoch_train_loss += loss.item() * batch_size
            epoch_train_samples += batch_size

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping if enabled
            if hasattr(args, 'grad_clip_norm') and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            
            optimizer.step()

            # Log training progress
            print(f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Step: {step_idx}/{len(train_loader)} Batch Size: {batch_size} N Points: {n_points} Loss: {loss.item():.4f}")

        # Calculate average training loss for this epoch
        avg_train_loss = epoch_train_loss / max(epoch_train_samples, 1)
        train_losses.append(avg_train_loss)

        # Evaluate model on validation set after each epoch
        val_loss = evaluate_model_on_patches(val_loader, model, loss_fn, args)
        val_losses.append(val_loss)
        
        print(f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_checkpoint_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, args)
        
        # Step the learning rate scheduler based on type
        if args.scheduler_type == 'cosine':
            # Cosine scheduler steps every epoch
            scheduler.step()
            print(f"Learning rate updated to: {scheduler.get_last_lr()[0]:.6f}")
        else:
            # Exponential scheduler steps at specified intervals
            if (epoch + 1) % args.lr_scheduler_interval == 0:
                scheduler.step()
                print(f"Learning rate updated to: {scheduler.get_last_lr()[0]:.6f}")

    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, train_losses, val_losses, args, 'final_checkpoint.pth')
    
    # Plot and save training curves
    plot_training_curves(train_losses, val_losses, args)

    return model
        

if __name__ == "__main__":
    
    # Parse command line arguments
    args = setup_arguments()
    print(f"Args: {args}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    if args.device == 'mps':
        torch.mps.manual_seed(args.seed)
    
    # Calculate coordinate normalization transform from raw data
    # This ensures spatial coordinates are properly normalized
    coord_transform = calculate_coord_transform(args.raw_data_dir)
    
    # Calculate observation normalization transform from raw data
    # This ensures observation values are properly normalized
    obs_transform = calculate_obs_transform(args.raw_data_dir)
    
    # Create patch datasets with normalization transforms
    # These datasets handle sliding window sequence generation with multi-column support
    train_patch_ds, val_patch_ds = create_patch_datasets(
        args.patch_data_dir, 
        coord_transform, 
        obs_transform,
        target_col_indices=args.target_col_indices,
        input_window_size=args.input_window_size,
        output_window_size=args.output_window_size,
    )
    
    print(f"Train dataset length: {len(train_patch_ds)}")
    print(f"Val dataset length: {len(val_patch_ds)}")

    print(f"Target columns: {args.target_cols}, target column indices: {args.target_col_indices}")
    print(f"Number of target columns: {args.n_target_cols}")
    print(f"Input channels: {args.in_gno_out_channels} = {args.input_window_size} * {args.n_target_cols}")
    print(f"Output channels: {args.out_channels} = {args.output_window_size} * {args.n_target_cols}")
    print(f"Shuffling configuration: within_batches={args.shuffle_within_batches}, patches={args.shuffle_patches}")

    # Define and initialize GINO model
    model = define_ginos_model(args)
    print(f"Model: {model}")
    
    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Choose scheduler type based on arguments
    if args.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
        )
    else:  # exponential
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    
    # Train the model on patch data
    model = train_gino_on_patches(train_patch_ds, val_patch_ds, model, optimizer, scheduler, args)

    # Save trained model state dict to results directory
    model_path = os.path.join(args.results_dir, 'gino_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")

