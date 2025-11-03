"""
Train GFNO on 2D plane sequences with multi-GPU support.

This script trains a Graph-Fourier Neural Operator (GFNO) on 2D plane sequence data.
The GFNO combines:
- GNO encoder: Processes boundary conditions on irregular geometries
- FNO core: Processes features in latent space on regular grids
- Direct prediction: Outputs uniform grid predictions from the FNO

Key differences from GINO:
- GFNO predicts directly on uniform grids (no output GNO needed)
- Input: boundary conditions on irregular points
- Output: predictions on regular latent grid
- Designed for 2D plane sequences with temporal evolution

Multi-GPU Support:
- Uses DataParallel for multi-GPU training
- Automatically detects available GPUs
- Handles static inputs (geometries) efficiently

Tensor shapes (per batch):
- input_geom: [B, N_bc_points, 3] - boundary condition coordinates (S, Z, T)
- input_data: [B, N_bc_points, 2] - boundary values (head, mass_conc)
- latent_queries: [B, alpha, H, W, 3] - latent grid coordinates
- latent_features: [B, alpha, H, W, 4] - latent grid features (X, Y, head, mass_conc)
- output: [B, alpha, H, W, 2] - predictions (head, mass_conc)

Resume Training:
- Use --resume-from path/to/checkpoint.pth to resume from a specific checkpoint
- Checkpoints are automatically saved every N epochs (configurable with --save-checkpoint-every)
- Training state (model, optimizer, scheduler, losses) is fully restored
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datetime as dt
import json
import pickle

from src.data.plane_dataset import GWPlaneDatasetFromFiles
from src.data.batch_sampler import PatchBatchSampler
from src.models import GFNO
from src.models.neuralop.losses import LpLoss


def setup_arguments():
    """Parse command line arguments for data, model, and training.
    
    Returns:
        argparse.Namespace: Parsed arguments with computed paths and configurations
    """
    parser = argparse.ArgumentParser(description='Train GFNO model on 2D plane sequences with multi-GPU support')
    
    # Data directories
    parser.add_argument('--data-dir', type=str,
                       default='/Users/arpitkapoor/data/GW/2d_plane_sequences',
                       help='Directory containing 2D plane sequence data')
    parser.add_argument('--results-dir', type=str, 
                       default='/srv/scratch/z5370003/projects/results/04_groundwater/2d_planes/GFNO',
                       help='Directory to save trained models and results')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Ratio of data to use for validation (default: 0.2)')
    
    # Model parameters
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate for optimizer')
    parser.add_argument('--lr-gamma', type=float, default=0.95,
                       help='Exponential learning rate decay factor')
    parser.add_argument('--lr-scheduler-interval', type=int, default=5,
                       help='Number of epochs between learning rate scheduler updates')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                       help='Gradient clipping norm value (0 to disable)')
    parser.add_argument('--scheduler-type', type=str, default='exponential',
                       choices=['exponential', 'cosine'],
                       help='Type of learning rate scheduler to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
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
    parser.add_argument('--save-checkpoint-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (cuda, cpu, or auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--use-multi-gpu', action='store_true', default=True,
                       help='Use DataParallel for multi-GPU training if available')
    
    args = parser.parse_args()
    
    # Create results directory with timestamp (unless resuming)
    if args.resume_from is None:
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.results_dir = os.path.join(args.results_dir, f'run_{timestamp}')
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"Created results directory: {args.results_dir}")
    else:
        # Extract results directory from checkpoint path
        checkpoint_path = os.path.abspath(args.resume_from)
        
        # Navigate up from checkpoint to find results directory
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.basename(checkpoint_dir) == 'checkpoints':
            args.results_dir = os.path.dirname(checkpoint_dir)
        else:
            args.results_dir = checkpoint_dir
        
        print(f"Resuming training from: {checkpoint_path}")
        print(f"Using results directory: {args.results_dir}")
        
        # Validate that results directory exists
        if not os.path.exists(args.results_dir):
            raise ValueError(f"Results directory does not exist: {args.results_dir}")
    
    # Set up checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.results_dir, 'checkpoints')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Define model parameters
    args = define_model_parameters(args)
    
    # Configure device
    args = configure_device(args)
    
    # Print configuration
    print(f"\nTraining Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Validation ratio: {args.val_ratio}")
    print(f"  Multi-GPU: {args.use_multi_gpu and torch.cuda.device_count() > 1}")
    
    return args


def define_model_parameters(args):
    """Define model parameters for GFNO architecture.
    
    Args:
        args: Argument namespace to modify with model parameters
        
    Returns:
        argparse.Namespace: Modified args with model parameters added
    """
    # Coordinate dimensions (3D: S, Z, T)
    args.coord_dim = 3
    
    # Number of target columns (head, mass_concentration)
    args.n_target_cols = 2
    
    # GNO encoder parameters
    args.gno_radius = 0.15
    args.gno_out_channels = 16  # Encoded boundary condition features
    args.gno_channel_mlp_layers = [32, 64, 32]
    args.gno_pos_embed_type = 'transformer'
    args.gno_pos_embed_channels = 32
    
    # Latent feature channels (X, Y, head, mass_conc)
    args.latent_feature_channels = 4
    
    # FNO parameters
    args.fno_n_layers = 4
    args.fno_n_modes = (6, 8, 8)  # 3D modes (alpha, S, Z)
    args.fno_hidden_channels = 64
    args.lifting_channels = 64
    
    # Output channels (head, mass_concentration)
    args.out_channels = 2
    
    # Projection parameters
    args.projection_channel_ratio = 2
    
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
        if torch.cuda.is_available():
            args.device = torch.device('cuda')
            print(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
            if torch.cuda.device_count() > 1:
                print(f"Found {torch.cuda.device_count()} GPUs available for training")
        elif torch.backends.mps.is_available():
            args.device = torch.device('mps')
            print("Auto-detected MPS device (Apple Silicon)")
        else:
            args.device = torch.device('cpu')
            print("No GPU detected, using CPU")
    elif args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but not available")
    else:
        args.device = torch.device(args.device)
    
    return args


def create_datasets(data_dir, val_ratio=0.2, **kwargs):
    """Create train/val datasets from 2D plane sequence files.
    
    Args:
        data_dir: Directory containing plane sequence data
        val_ratio: Ratio of sequences to use for validation (default: 0.2)
        **kwargs: Additional arguments (currently unused)
        
    Returns:
        tuple: (train_dataset, validation_dataset)
    """
    print(f"\nLoading datasets from {data_dir}...")
    print(f"Train/Val split ratio: {1-val_ratio:.1%} / {val_ratio:.1%}")
    
    # Create training dataset
    train_ds = GWPlaneDatasetFromFiles(
        data_dir=data_dir,
        dataset='train',
        val_ratio=val_ratio,
        fill_nan_value=-999.0
    )
    
    # Create validation dataset
    val_ds = GWPlaneDatasetFromFiles(
        data_dir=data_dir,
        dataset='val',
        val_ratio=val_ratio,
        fill_nan_value=-999.0
    )
    
    print(f"Training dataset: {len(train_ds)} sequences")
    print(f"Validation dataset: {len(val_ds)} sequences")
    print(f"Total sequences: {len(train_ds) + len(val_ds)}")
    
    return train_ds, val_ds


def define_gfno_model(args):
    """Define GFNO model with 3D coordinates.
    
    Initializes a Graph-Fourier Neural Operator (GFNO) with parameters
    configured for 2D plane modeling with temporal sequences.
    
    Args:
        args: Argument namespace containing model hyperparameters
        
    Returns:
        GFNO: Configured GFNO model moved to specified device
    """
    model = GFNO(
        # GNO encoder parameters
        gno_coord_dim=args.coord_dim,
        gno_radius=args.gno_radius,
        gno_out_channels=args.gno_out_channels,
        gno_channel_mlp_layers=args.gno_channel_mlp_layers,
        gno_pos_embed_type=args.gno_pos_embed_type,
        gno_pos_embed_channels=args.gno_pos_embed_channels,
        # Latent features
        latent_feature_channels=args.latent_feature_channels,
        # FNO parameters
        fno_n_layers=args.fno_n_layers,
        fno_n_modes=args.fno_n_modes,
        fno_hidden_channels=args.fno_hidden_channels,
        lifting_channels=args.lifting_channels,
        # Projection parameters
        projection_channel_ratio=args.projection_channel_ratio,
        out_channels=args.out_channels,
    ).to(args.device)
    
    return model


# ------------------------------
# DataParallel adapter + helpers
# ------------------------------
class GFNODataParallelAdapter(torch.nn.Module):
    """
    Thin wrapper for DataParallel to handle GFNO's multiple input tensors.
    
    DataParallel splits along batch dimension by default. This adapter ensures
    proper handling of all GFNO inputs including geometries and features.
    """
    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, *, input_geom, latent_queries, x, latent_features):
        """Forward pass with keyword arguments."""
        return self.inner(
            input_geom=input_geom,
            latent_queries=latent_queries,
            x=x,
            latent_features=latent_features
        )


def _unwrap_model_for_state_dict(model: torch.nn.Module) -> torch.nn.Module:
    """
    Extract the underlying GFNO module from DataParallel wrapper.
    
    Args:
        model: Potentially wrapped model
        
    Returns:
        torch.nn.Module: Unwrapped GFNO model
    """
    m = model
    # Unwrap DataParallel
    if hasattr(m, 'module'):
        m = m.module
    # Unwrap adapter
    if hasattr(m, 'inner'):
        m = m.inner
    return m


def evaluate_model(val_loader, model, loss_fn, args):
    """Evaluate model on validation loader.
    
    Computes loss on validation batches.
    
    Args:
        val_loader: DataLoader for validation data
        model: GFNO model to evaluate
        loss_fn: Loss function (typically LpLoss)
        args: Argument namespace containing device info
        
    Returns:
        float: Average validation loss across all batches
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move batch to device
            input_geom = batch['input_geom'].to(args.device)
            input_data = batch['input_data'].to(args.device)
            latent_geom = batch['latent_geom'].to(args.device)
            latent_features = batch['latent_features'].to(args.device)
            output_latent_features = batch['output_latent_features'].to(args.device)
            
            # Forward pass
            predictions = model(
                input_geom=input_geom,
                latent_queries=latent_geom,
                x=input_data,
                latent_features=latent_features
            )
            
            # Compute loss
            # Extract only the target columns (head, mass_conc) from output features
            targets = output_latent_features[..., -args.n_target_cols:]
            
            loss = loss_fn(predictions, targets)
            
            # Accumulate loss
            batch_size = input_geom.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    model.train()
    return total_loss / max(total_samples, 1)


def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, args, filename=None):
    """Save a training checkpoint to enable resuming training.
    
    Args:
        model: GFNO model to save
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
        'model_state_dict': _unwrap_model_for_state_dict(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'args': args,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Also save as latest checkpoint for easy resuming
    latest_path = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    print(f"Latest checkpoint saved: {latest_path}")
    
    # Save accumulated loss history
    accumulated_train, accumulated_val = get_accumulated_losses(train_losses, val_losses, args, checkpoint_path)
    
    # Save loss history next to checkpoint
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
        model: GFNO model to load state into
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
    _unwrap_model_for_state_dict(model).load_state_dict(checkpoint['model_state_dict'])
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
        'coord_dim', 'gno_radius', 'gno_out_channels', 'fno_n_layers',
        'fno_n_modes', 'fno_hidden_channels', 'out_channels',
        'latent_feature_channels', 'n_target_cols'
    ]
    
    for param in critical_params:
        saved_value = getattr(saved_args, param, None)
        current_value = getattr(current_args, param, None)
        if saved_value != current_value:
            raise ValueError(
                f"Checkpoint incompatibility: {param} mismatch. "
                f"Saved: {saved_value}, Current: {current_value}"
            )
    
    print("Checkpoint compatibility validated successfully")


def save_loss_history(accumulated_train_losses, accumulated_val_losses, args):
    """Save training and validation loss history to a persistent JSON file.
    
    Args:
        accumulated_train_losses: Complete list of accumulated training losses
        accumulated_val_losses: Complete list of accumulated validation losses
        args: Argument namespace containing results directory
    """
    loss_history_path = os.path.join(args.results_dir, 'loss_history.json')
    
    loss_history = {
        'train_losses': accumulated_train_losses,
        'val_losses': accumulated_val_losses,
        'last_updated': dt.datetime.now().isoformat(),
        'total_epochs': len(accumulated_train_losses)
    }
    
    with open(loss_history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"Loss history saved to '{loss_history_path}' (total epochs: {len(accumulated_train_losses)})")


def load_loss_history(args, checkpoint_path=None):
    """Load training and validation loss history from persistent JSON file.
    
    Args:
        args: Argument namespace containing results directory
        checkpoint_path: Optional path to checkpoint file being loaded
        
    Returns:
        dict: Dictionary containing train_losses, val_losses, and metadata
    """
    # Try checkpoint directory first if resuming
    if checkpoint_path is not None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_loss_history = os.path.join(checkpoint_dir, 'loss_history.json')
        if os.path.exists(checkpoint_loss_history):
            with open(checkpoint_loss_history, 'r') as f:
                history = json.load(f)
            print(f"Loaded loss history from checkpoint directory: {len(history['train_losses'])} epochs")
            return history
    
    # Try results directory as fallback
    results_loss_history = os.path.join(args.results_dir, 'loss_history.json')
    if os.path.exists(results_loss_history):
        with open(results_loss_history, 'r') as f:
            history = json.load(f)
        print(f"Loaded loss history from results directory: {len(history['train_losses'])} epochs")
        return history
    
    # Return empty history if no valid history found
    print("No existing loss history found, starting fresh.")
    return {'train_losses': [], 'val_losses': [], 'total_epochs': 0}


def get_accumulated_losses(train_losses, val_losses, args, checkpoint_path=None):
    """Get accumulated losses from all training sessions.
    
    Args:
        train_losses: Current session's training losses
        val_losses: Current session's validation losses
        args: Argument namespace containing results directory
        checkpoint_path: Optional path to checkpoint file being loaded
        
    Returns:
        tuple: (accumulated_train_losses, accumulated_val_losses)
    """
    # Load existing loss history
    existing_history = load_loss_history(args, checkpoint_path)
    existing_train = existing_history.get('train_losses', [])
    existing_val = existing_history.get('val_losses', [])
    
    # Get counts
    existing_epoch_count = len(existing_train)
    current_epoch_count = len(train_losses)
    
    # Merge: append only new epochs
    if current_epoch_count > existing_epoch_count:
        new_train = train_losses[existing_epoch_count:]
        new_val = val_losses[existing_epoch_count:]
        accumulated_train = existing_train + new_train
        accumulated_val = existing_val + new_val
    else:
        accumulated_train = existing_train
        accumulated_val = existing_val
    
    return accumulated_train, accumulated_val


def plot_training_curves(train_losses, val_losses, args):
    """Plot training and validation loss curves and save to results directory.
    
    Args:
        train_losses: List of training losses per epoch (current session)
        val_losses: List of validation losses per epoch (current session)
        args: Argument namespace containing results directory
    """
    # Get accumulated losses from all training sessions
    checkpoint_path = args.resume_from if hasattr(args, 'resume_from') else None
    accumulated_train, accumulated_val = get_accumulated_losses(train_losses, val_losses, args, checkpoint_path)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(accumulated_train) + 1)
    
    plt.plot(epochs, accumulated_train, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, accumulated_val, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    
    # Add visual indicators for resume points if this is a resumed session
    if args.resume_from is not None and len(accumulated_train) > len(train_losses):
        resume_epoch = len(accumulated_train) - len(train_losses)
        plt.axvline(x=resume_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Resume Point')
    
    plt.title('Training and Validation Loss Over Epochs (GFNO)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics
    if accumulated_train:
        stats_text = f"Final Train Loss: {accumulated_train[-1]:.6f}\n"
        stats_text += f"Final Val Loss: {accumulated_val[-1]:.6f}\n"
        stats_text += f"Best Val Loss: {min(accumulated_val):.6f} (Epoch {accumulated_val.index(min(accumulated_val))+1})"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(args.results_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to '{plot_path}' (total epochs: {len(accumulated_train)})")
    
    # Save the accumulated loss history
    save_loss_history(accumulated_train, accumulated_val, args)


def train_gfno(train_ds, val_ds, model, optimizer, scheduler, args):
    """Train GFNO with multi-GPU support using DataParallel.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        model: GFNO model to train
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        args: Argument namespace with training hyperparameters
        
    Returns:
        GFNO: Trained model
    """
    # Wrap model with DataParallel if multiple GPUs available
    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"\nUsing DataParallel with {torch.cuda.device_count()} GPUs")
        model = GFNODataParallelAdapter(model)
        model = nn.DataParallel(model)
        print("Model wrapped with DataParallel")
    
    # Create training sampler
    train_sampler = PatchBatchSampler(
        train_ds,
        batch_size=args.batch_size,
        shuffle_within_batches=args.shuffle_within_batches,
        shuffle_patches=args.shuffle_patches,
        seed=args.seed
    )
    
    # Create validation sampler
    val_sampler = PatchBatchSampler(
        val_ds,
        batch_size=args.batch_size,
        shuffle_within_batches=False,
        shuffle_patches=False,
        seed=None
    )
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler)
    
    print(f"\nDataLoader Configuration:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Use relative L2 loss
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
    print(f"\nStarting training from epoch {start_epoch + 1}/{args.epochs}")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Training phase
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_geom = batch['input_geom'].to(args.device)
            input_data = batch['input_data'].to(args.device)
            latent_geom = batch['latent_geom'].to(args.device)
            latent_features = batch['latent_features'].to(args.device)
            output_latent_features = batch['output_latent_features'].to(args.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(
                input_geom=input_geom,
                latent_queries=latent_geom,
                x=input_data,
                latent_features=latent_features
            )
            
            # Compute loss
            # Extract only the target columns (head, mass_conc) from output features
            targets = output_latent_features[..., -args.n_target_cols:]
            
            loss = loss_fn(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f} Avg: {avg_loss:.6f}")
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss = evaluate_model(val_loader, model, loss_fn, args)
        val_losses.append(val_loss)
        
        # Learning rate scheduler step
        if (epoch + 1) % args.lr_scheduler_interval == 0:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate updated to: {current_lr:.6e}")
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6e}")
        print("-" * 80)
        
        # Save checkpoint
        if (epoch + 1) % args.save_checkpoint_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, args)
        
        # Plot training curves
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            plot_training_curves(train_losses, val_losses, args)
    
    # Final checkpoint and plots
    print("\n" + "=" * 80)
    print("Training completed!")
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, train_losses, val_losses, args, 
                   filename='final_checkpoint.pth')
    plot_training_curves(train_losses, val_losses, args)
    
    return model


if __name__ == "__main__":
    # Parse arguments
    args = setup_arguments()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Create datasets
    train_ds, val_ds = create_datasets(args.data_dir, val_ratio=args.val_ratio)
    
    # Create model
    print("\nInitializing GFNO model...")
    model = define_gfno_model(args)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create learning rate scheduler
    if args.scheduler_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Train model
    trained_model = train_gfno(train_ds, val_ds, model, optimizer, scheduler, args)
    
    print("\n" + "=" * 80)
    print("Training pipeline completed successfully!")
    print(f"Results saved to: {args.results_dir}")
    print("=" * 80)
