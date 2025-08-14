"""
Train GINO on variable-density groundwater patches with true batch training.

This script:
- Builds `GWPatchDataset` that yields sliding-window sequences per patch
- Uses `PatchBatchSampler` to group batch indices from the same `patch_id`
- Collates a batch into a single point cloud (core+ghost) and batches sequences
- Generates a per-batch latent grid over the point cloud bounding box
- Trains GINO with batched forward/backward passes

Tensor shapes (per batch):
- point_coords: [N_points, 3]
- latent_queries: [Qx, Qy, Qz, 3]
- x (inputs): [B, N_points, input_window_size]
- y (targets): [B, N_points, output_window_size]
- outputs: [B, N_points, output_window_size]

Loss is computed only on core points to avoid boundary artifacts from ghost points.
"""

import argparse
import os

import pandas as pd
import torch
import datetime as dt
from torch.utils.data import DataLoader

from src.data.transform import Normalize
from src.data.patch_dataset import GWPatchDataset
from src.data.batch_sampler import PatchBatchSampler
from src.models.neuralop.gino import GINO
from src.models.neuralop.losses import LpLoss, H1Loss

def setup_arguments():
    """Parse command line arguments for data, model, and training.

    Notable arguments:
    - --target-col: single observation field to model (mapped to `target_col_idx`)
    - --input-window-size / --output-window-size: sliding window lengths
    - --batch-size: number of sequences per training step (from the same patch)
    
    Returns:
        argparse.Namespace: Parsed arguments with computed paths and configurations
    """
    parser = argparse.ArgumentParser(description='Train GINO model on groundwater patches')
    
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

    # Target observation column (single)
    parser.add_argument('--target-col', type=str, default='mass_concentration',
                       help='Single target observation column name')
    # Sequence lengths
    parser.add_argument('--input-window-size', type=int, default=10,
                       help='Number of time steps in each input sequence')
    parser.add_argument('--output-window-size', type=int, default=10,
                       help='Number of time steps in each output sequence')
    
    # Model parameters (placeholder for future use)
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate for optimizer')
    parser.add_argument('--lr-gamma', type=float, default=0.95,
                       help='Exponential learning rate decay factor')
    parser.add_argument('--lr-scheduler-interval', type=int, default=5,
                       help='Number of epochs between learning rate scheduler updates')
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
    
    # Other parameters (placeholder for future use)
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (cuda, cpu, or auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Construct full paths and add them to args
    args.raw_data_dir = os.path.join(args.base_data_dir, args.raw_data_subdir)
    args.patch_data_dir = os.path.join(args.base_data_dir, args.patch_data_subdir)

    # Create results directory with timestamp
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.results_dir = os.path.join(args.results_dir, f'gino_{timestamp}')
    os.makedirs(args.results_dir, exist_ok=True)

    # Define model parameters
    args = define_model_parameters(args)

    # Configure target observation columns
    args = configure_target_col_idx(args)

    # Configure device
    args = configure_device(args)
    
    # Print data directories for verification
    print(f"Base data directory: {args.base_data_dir}")
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Patch filtered data directory: {args.patch_data_dir}")
    print(f"Results directory: {args.results_dir}")
    
    return args

def define_model_parameters(args):
    """Define model parameters for GINO architecture.

    Notes:
    - `in_gno_out_channels` are feature channels produced by the input GNO.
    - `out_channels` is set to the output window size so the projection head
      predicts a full future sequence per point.
    
    Args:
        args: Argument namespace to modify with model parameters
        
    Returns:
        argparse.Namespace: Modified args with model parameters added
    """
    # Coordinate dimensions (3D: x, y, z)
    args.coord_dim = 3
    
    # Radius for neighbor search in GNO operations
    args.gno_radius = 0.1
    
    # Output channels of the input GNO block (feature channels produced after aggregation)
    args.in_gno_out_channels = args.input_window_size
    
    # MLP layer dimensions for input GNO channel processing
    args.in_gno_channel_mlp_layers = [32, 64, 32]
    
    # FNO (Fourier Neural Operator) configuration
    args.fno_n_layers = 4
    args.fno_n_modes = (8, 8, 8)  # 3D Fourier modes
    args.fno_hidden_channels = 64
    args.lifting_channels = 64
    
    # Output GNO configuration
    args.out_gno_channel_mlp_layers = [32, 64, 32]
    args.projection_channel_ratio = 2
    
    # Predict the full output window per point
    args.out_channels = args.output_window_size
    
    # Latent query grid dimensions (creates a 32x32x32 grid)
    args.latent_query_dims = (32, 32, 32)
    
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

def configure_target_col_idx(args):
    """Configure a single target observation column index from name.

    Maps human-readable column names to array indices for the observation data.
    
    Backward compatibility: if a legacy list is provided in `args.target_cols`,
    the first element is used.
    
    Args:
        args: Argument namespace with target column name
        
    Returns:
        argparse.Namespace: Modified args with target_col_idx added
    """
    # Mapping from column names to indices in the observation array
    names_to_idx = {
        'mass_concentration': 0,
        'head': 1,
        'pressure': 2
    }
    
    if hasattr(args, 'target_col') and args.target_col is not None:
        args.target_col_idx = names_to_idx[args.target_col]
    else:
        # Backward compatibility if a list was provided elsewhere
        target_cols = getattr(args, 'target_cols', ['mass_concentration'])
        if isinstance(target_cols, (list, tuple)):
            args.target_col_idx = names_to_idx[target_cols[0]]
            args.target_col = target_cols[0]
        else:
            args.target_col_idx = names_to_idx[target_cols]
            args.target_col = target_cols
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

    Normalizes all listed columns for consistency. The dataset will then select a
    single target by `target_col_idx` before sequencing.
    
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
    """Create train/val `GWPatchDataset` with normalization and sequencing.

    Creates datasets that apply coordinate and observation normalization,
    then generate sliding window sequences for training and validation.
    
    Args:
        patch_data_dir: Directory containing patch data files
        coord_transform: Normalization transform for coordinates
        obs_transform: Normalization transform for observations
        **kwargs: Additional arguments including window sizes and target column index
        
    Returns:
        tuple: (train_dataset, validation_dataset)
    """
    # Create training dataset
    train_ds = GWPatchDataset(
        data_path=patch_data_dir,
        dataset='train', 
        coord_transform=coord_transform, 
        obs_transform=obs_transform,
        input_window_size=kwargs.get('input_window_size', 10),
        output_window_size=kwargs.get('output_window_size', 10),
        target_col_idx=kwargs.get('target_col_idx', None),
    )
    
    # Create validation dataset
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


def define_ginos_model(args):
    """Define GINO model with 3D coordinates and sequence projection head.
    
    Initializes a Graph-Informed Neural Operator (GINO) with parameters
    configured for 3D groundwater modeling with temporal sequences.
    
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
            sample_input = torch.concat([sample['core_in'], sample['ghost_in']], dim=0).float().unsqueeze(0)
            sample_output = torch.concat([sample['core_out'], sample['ghost_out']], dim=0).float().unsqueeze(0)
            x_list.append(sample_input)
            y_list.append(sample_output)

        # Stack all sequences into batch tensors
        x = torch.cat(x_list, dim=0)  # [B, N_points, input_window_size]
        y = torch.cat(y_list, dim=0)  # [B, N_points, output_window_size]

        # Return batch dictionary
        batch = {
            'point_coords': point_coords,      # [N_points, 3]
            'latent_queries': latent_queries,  # [Qx, Qy, Qz, 3]
            'x': x,                           # [B, N_points, input_window_size]
            'y': y,                           # [B, N_points, output_window_size]
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
            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    model.train()  # Return model to training mode
    return total_loss / max(total_samples, 1)


def train_gino_on_patches(train_patch_ds, val_patch_ds, model, args):
    """Train GINO with true batch training using DataLoader and PatchBatchSampler.

    Each training step uses multiple sequences from the same patch to share the
    same point cloud and latent grid, improving neighbor search and cache reuse.
    
    Args:
        train_patch_ds: Training dataset
        val_patch_ds: Validation dataset
        model: GINO model to train
        args: Argument namespace with training hyperparameters
        
    Returns:
        GINO: Trained model
    """
    # Create training sampler (will reshuffle automatically each epoch)
    # OPTIMIZATION: No need to recreate sampler/loader every epoch!
    train_sampler = PatchBatchSampler(
        train_patch_ds, 
        batch_size=args.batch_size,
        shuffle_within_batches=args.shuffle_within_batches,  # Use command line argument
        shuffle_patches=args.shuffle_patches,                 # Use command line argument
        seed=args.seed                                       # Use the same seed for reproducibility
    )
    
    # Create validation sampler (no shuffling for deterministic evaluation)
    val_sampler = PatchBatchSampler(
        val_patch_ds, 
        batch_size=args.batch_size,
        shuffle_within_batches=False, # No shuffling for validation
        shuffle_patches=False,        # No shuffling for validation
        seed=None                    # No seed for validation (deterministic)
    )

    # Custom collate function to handle patch-based batching
    collate_fn = _make_collate_fn(args)

    # Create data loaders (fixed - no recreation needed)
    train_loader = DataLoader(train_patch_ds, batch_sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_patch_ds, batch_sampler=val_sampler, collate_fn=collate_fn)

    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")

    # Define optimizer and loss (relative L2 over time channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Add exponential learning rate decay scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    
    # Use relative L2 loss with appropriate dimensionality
    loss_fn = LpLoss(d=1, p=2, reduce_dims=[0, 1], reductions='mean')

    # Training loop
    for epoch in range(args.epochs):
        print(f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Training Epoch {epoch+1} of {args.epochs}")
        print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Log shuffling configuration for this epoch
        print(f"Epoch {epoch+1} shuffling: within_batches={args.shuffle_within_batches}, patches={args.shuffle_patches}")

        # Training step loop
        for step_idx, batch in enumerate(train_loader, start=1):
            # Move batch data to device
            point_coords = batch['point_coords'].to(args.device).float()
            latent_queries = batch['latent_queries'].to(args.device).float()
            x = batch['x'].to(args.device).float()
            y = batch['y'].to(args.device).float()
            
            # Extract batch dimensions for logging
            batch_size = x.shape[0]
            n_points = x.shape[1]

            # Forward pass through GINO model
            outputs = model(
                input_geom=point_coords,      # Point cloud coordinates
                latent_queries=latent_queries, # Regular grid for FNO
                x=x,                          # Input sequences
                output_queries=point_coords,  # Query points (same as input geometry)
            )

            # Extract core points only for loss computation
            # Ghost points are excluded to avoid boundary artifacts
            core_len = batch['core_len']
            core_output = outputs[:, :core_len]
            core_target = y[:, :core_len]

            # Compute loss on core points only
            loss = loss_fn(core_output, core_target)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training progress
            print(f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Step: {step_idx}/{len(train_loader)} Batch Size: {batch_size} N Points: {n_points} Loss: {loss.item():.4f}")

        # Evaluate model on validation set after each epoch
        val_loss = evaluate_model_on_patches(val_loader, model, loss_fn, args)
        print(f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Validation Loss: {val_loss:.4f}")
        
        # Step the learning rate scheduler only at the specified interval
        if (epoch + 1) % args.lr_scheduler_interval == 0:
            scheduler.step()
            print(f"Learning rate updated to: {scheduler.get_last_lr()[0]:.6f}")

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
    # These datasets handle sliding window sequence generation
    train_patch_ds, val_patch_ds = create_patch_datasets(
        args.patch_data_dir, 
        coord_transform, 
        obs_transform,
        target_col_idx=args.target_col_idx,
        input_window_size=args.input_window_size,
        output_window_size=args.output_window_size,
    )
    
    print(f"Train dataset length: {len(train_patch_ds)}")
    print(f"Val dataset length: {len(val_patch_ds)}")

    print(f"Target column: {args.target_col}, target column idx: {args.target_col_idx}")
    print(f"Shuffling configuration: within_batches={args.shuffle_within_batches}, patches={args.shuffle_patches}")

    # Define and initialize GINO model
    model = define_ginos_model(args)
    print(f"Model: {model}")
    
    # Train the model on patch data
    model = train_gino_on_patches(train_patch_ds, val_patch_ds, model, args)

    # Save trained model state dict to results directory
    model_path = os.path.join(args.results_dir, 'gino_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")