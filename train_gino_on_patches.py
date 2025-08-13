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
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    
    # Other parameters (placeholder for future use)
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (cuda, cpu, or auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Construct full paths and add them to args
    args.raw_data_dir = os.path.join(args.base_data_dir, args.raw_data_subdir)
    args.patch_data_dir = os.path.join(args.base_data_dir, args.patch_data_subdir)

    # Define model parameters
    args = define_model_parameters(args)

    # Configure target observation columns
    args = configure_target_col_idx(args)

    # Configure device
    args = configure_device(args)
    
    # Print data directories
    print(f"Base data directory: {args.base_data_dir}")
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Patch filtered data directory: {args.patch_data_dir}")
    
    return args

def define_model_parameters(args):
    """Define model parameters.

    Notes:
    - `in_gno_out_channels` are feature channels produced by the input GNO.
    - `out_channels` is set to the output window size so the projection head
      predicts a full future sequence per point.
    """
    args.coord_dim = 3
    args.gno_radius = 0.1
    # Output channels of the input GNO block (feature channels produced after aggregation)
    args.in_gno_out_channels = args.input_window_size
    args.in_gno_channel_mlp_layers = [32, 64, 32]
    args.fno_n_layers = 4
    args.fno_n_modes = (8, 8, 8)
    args.fno_hidden_channels = 64
    args.lifting_channels = 64
    args.out_gno_channel_mlp_layers = [32, 64, 32]
    args.projection_channel_ratio = 2
    # Predict the full output window per point
    args.out_channels = args.output_window_size
    args.latent_query_dims = (32, 32, 32)
    return args

def configure_device(args):
    """Configure device: cpu, cuda, or mps."""
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Please use 'auto' or specify 'cpu'.")
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available. Please use 'auto' or specify 'cpu'.")
    return args

def configure_target_col_idx(args):
    """Configure a single target observation column index from name.

    Backward compatibility: if a legacy list is provided in `args.target_cols`,
    the first element is used.
    """
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

    Uses a representative CSV (0000.csv) to derive normalization stats.
    """
    # Read data
    df = pd.read_csv(os.path.join(raw_data_dir, '0000.csv'))

    # Calculate mean and std of coordinates
    coord_mean = df[['X', 'Y', 'Z']].mean().values
    coord_std = df[['X', 'Y', 'Z']].std().values

    # Print mean and std of coordinates
    print(f"Coordinate mean: {coord_mean}")
    print(f"Coordinate std: {coord_std}")

    # Create coordinate transform
    coord_transform = Normalize(mean=coord_mean, std=coord_std)

    del df
    return coord_transform

def calculate_obs_transform(raw_data_dir, 
                            target_obs_cols=['mass_concentration', 'head', 'pressure']):
    """Calculate mean and std of output variables and create observation transform.

    Normalizes all listed columns for consistency. The dataset will then select a
    single target by `target_col_idx` before sequencing.
    """
    # Read data
    df = pd.read_csv(os.path.join(raw_data_dir, '0000.csv'))

    # Define output columns
    obs_cols = target_obs_cols

    # Mean and std of output
    obs_mean = df[obs_cols].mean().values
    obs_std = df[obs_cols].std().values

    # Print mean and std of output
    print(f"Output mean: {obs_mean}")
    print(f"Output std: {obs_std}")

    # Define output transform
    obs_transform = Normalize(mean=obs_mean, std=obs_std)

    del df
    return obs_transform

def create_patch_datasets(patch_data_dir, coord_transform, obs_transform, **kwargs):
    """Create train/val `GWPatchDataset` with normalization and sequencing.

    Expects kwargs: `input_window_size`, `output_window_size`, `target_col_idx`.
    """
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


def define_ginos_model(args):
    """Define GINO model with 3D coordinates and sequence projection head."""
    model = GINO(
        in_gno_coord_dim=args.coord_dim,
        in_gno_radius=args.gno_radius,
        in_gno_out_channels=args.in_gno_out_channels,
        in_gno_channel_mlp_layers=args.in_gno_channel_mlp_layers,
        fno_n_layers=args.fno_n_layers,
        fno_n_modes=args.fno_n_modes,  # 3D modes
        fno_hidden_channels=args.fno_hidden_channels,
        lifting_channels=args.lifting_channels,
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
    """
    def collate_fn(batch_samples):
        # All samples in the batch come from the same patch (by sampler design)
        core_coords = batch_samples[0]['core_coords']
        ghost_coords = batch_samples[0]['ghost_coords']

        # Single point cloud per batch: [N_core+N_ghost, 3]
        point_coords = torch.concat([core_coords, ghost_coords], dim=0).float()

        # Latent queries grid over the per-batch bounding box
        coords_min = torch.min(point_coords, dim=0).values
        coords_max = torch.max(point_coords, dim=0).values
        latent_query_arr = [
            torch.linspace(coords_min[i], coords_max[i], args.latent_query_dims[i], device=args.device)
            for i in range(args.coord_dim)
        ]
        latent_queries = torch.stack(torch.meshgrid(*latent_query_arr, indexing='ij'), dim=-1)

        # Build batched sequences: concat along points (dim=0), batch along dim=0
        x_list, y_list = [], []
        for sample in batch_samples:
            sample_input = torch.concat([sample['core_in'], sample['ghost_in']], dim=0).float().unsqueeze(0)
            sample_output = torch.concat([sample['core_out'], sample['ghost_out']], dim=0).float().unsqueeze(0)
            x_list.append(sample_input)
            y_list.append(sample_output)

        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)

        batch = {
            'point_coords': point_coords,
            'latent_queries': latent_queries,
            'x': x,
            'y': y,
            'core_len': len(core_coords),
        }
        return batch
    return collate_fn
 

def evaluate_model_on_patches(val_loader, model, loss_fn, args):
    """Evaluate model on validation loader.

    Computes relative L2 loss on core points across validation batches.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            point_coords = batch['point_coords'].to(args.device).float()
            latent_queries = batch['latent_queries'].to(args.device).float()
            x = batch['x'].to(args.device).float()
            y = batch['y'].to(args.device).float()

            outputs = model(
                input_geom=point_coords,
                latent_queries=latent_queries,
                x=x,
                output_queries=point_coords,
            )

            core_len = batch['core_len']
            core_output = outputs[:, :core_len]
            core_target = y[:, :core_len]

            loss = loss_fn(core_output, core_target)
            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    model.train()
    return total_loss / max(total_samples, 1)


def train_gino_on_patches(train_patch_ds, val_patch_ds, model, args):
    """Train GINO with true batch training using DataLoader and PatchBatchSampler.

    Each training step uses multiple sequences from the same patch to share the
    same point cloud and latent grid, improving neighbor search and cache reuse.
    """
    # DataLoaders with batch sampler that groups by patch_id
    train_sampler = PatchBatchSampler(train_patch_ds, batch_size=args.batch_size)
    val_sampler = PatchBatchSampler(val_patch_ds, batch_size=args.batch_size)

    collate_fn = _make_collate_fn(args)

    train_loader = DataLoader(train_patch_ds, batch_sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_patch_ds, batch_sampler=val_sampler, collate_fn=collate_fn)

    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")

    # Define optimizer and loss (relative L2 over time channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = LpLoss(d=3, p=2)

    for epoch in range(args.epochs):
        print(f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Training Epoch {epoch+1} of {args.epochs}")

        for step_idx, batch in enumerate(train_loader, start=1):
            point_coords = batch['point_coords'].to(args.device).float()
            latent_queries = batch['latent_queries'].to(args.device).float()
            x = batch['x'].to(args.device).float()
            y = batch['y'].to(args.device).float()

            outputs = model(
                input_geom=point_coords,
                latent_queries=latent_queries,
                x=x,
                output_queries=point_coords,
            )

            core_len = batch['core_len']
            core_output = outputs[:, :core_len]
            core_target = y[:, :core_len]

            loss = loss_fn(core_output, core_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Step: {step_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        # Evaluate model on validation set
        val_loss = evaluate_model_on_patches(val_loader, model, loss_fn, args)
        print(f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Validation Loss: {val_loss:.4f}")

    return model
        

if __name__ == "__main__":
    
    # Parse command line arguments
    args = setup_arguments()
    print(f"Args: {args}")

    # Set seed
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    if args.device == 'mps':
        torch.mps.manual_seed(args.seed)
    
    # Calculate coordinate normalization transform from raw data
    coord_transform = calculate_coord_transform(args.raw_data_dir)
    
    # Calculate observation normalization transform from raw data
    obs_transform = calculate_obs_transform(args.raw_data_dir)
    
    # Create patch dataset with normalization transforms
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

    # Define GINO model
    model = define_ginos_model(args)
    print(f"Model: {model}")
    
    # Examine the structure of patch data
    model = train_gino_on_patches(train_patch_ds, val_patch_ds, model, args)

    # Save model to file
    torch.save(model.state_dict(), 'gino_model.pth')