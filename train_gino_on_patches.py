import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

from src.data.transform import Normalize
from src.data.patch_dataset import GWPatchDataset
from src.models.neuralop.gino import GINO
from src.models.neuralop.losses import LpLoss, H1Loss

def setup_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GINO model on groundwater patches')
    
    # Data directories
    parser.add_argument('--base-data-dir', type=str, 
                       default='/Users/arpitkapoor/Library/CloudStorage/OneDrive-UNSW/Shared/Projects/01_PhD/05_groundwater/data/FEFLOW/variable_density',
                       help='Base data directory')
    parser.add_argument('--raw-data-subdir', type=str, default='all',
                       help='Raw data subdirectory name')
    parser.add_argument('--patch-data-subdir', type=str, default='filter_all_ts_patch',
                       help='Patch data subdirectory name')

    # Target observation columns
    parser.add_argument('--target-cols', type=str, nargs='+', default=['mass_concentration', 'head'],
                       help='Target observation columns')
    
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
    args = configure_target_cols_idx(args)

    # Configure device
    args = configure_device(args)
    
    # Print data directories
    print(f"Base data directory: {args.base_data_dir}")
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Patch filtered data directory: {args.patch_data_dir}")
    
    return args

def define_model_parameters(args):
    """Define model parameters."""
    args.coord_dim = 3
    args.gno_radius = 0.1
    args.in_channels = len(args.target_cols)
    args.in_gno_channel_mlp_layers = [32, 64, 32]
    args.fno_n_layers = 4
    args.fno_n_modes = (8, 8, 8)
    args.fno_hidden_channels = 64
    args.lifting_channels = 64
    args.out_gno_channel_mlp_layers = [32, 64, 32]
    args.projection_channel_ratio = 2
    args.out_channels = len(args.target_cols)
    args.latent_query_dims = (32, 32, 32)
    return args

def configure_device(args):
    """Configure device: cpu, cuda, or mps"""
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Please use 'auto' or specify 'cpu'.")
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available. Please use 'auto' or specify 'cpu'.")
    return args

def configure_target_cols_idx(args):
    """Configure target observation columns."""
    names_to_idx = {
        'mass_concentration': 0,
        'head': 1,
        'pressure': 2
    }
    args.target_cols_idx = [names_to_idx[col] for col in args.target_cols]
    return args

def calculate_coord_transform(raw_data_dir):
    """Calculate mean and std of coordinates and create coordinate transform."""
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
    """Calculate mean and std of output variables and create observation transform."""
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
    """Create patch dataset."""
    train_ds = GWPatchDataset(data_path=patch_data_dir, dataset='train', 
                              coord_transform=coord_transform, 
                              obs_transform=obs_transform,
                              **kwargs)
    
    val_ds = GWPatchDataset(data_path=patch_data_dir, dataset='val', 
                              coord_transform=coord_transform, 
                              obs_transform=obs_transform,
                              **kwargs)

    return train_ds, val_ds


def define_ginos_model(args):
    """Define GINO model."""
    model = GINO(
        in_gno_coord_dim=args.coord_dim,
        in_gno_radius=args.gno_radius,
        in_gno_out_channels=args.in_channels,  # Match input channels (3)
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

def train_on_batch(model, train_patch_ds, batch_indices, loss_fn, optimizer, args):

    # Batch loss
    batch_loss = 0.0

    for sample_idx in batch_indices:

        # Get patch data
        patch_data = train_patch_ds[sample_idx]
        
        # Concatenate core and ghost coordinates
        point_coords = torch.concat([patch_data['core_coords'], patch_data['ghost_coords']], dim=0).float()
        
        # Calculate min and max of point coordinates
        coords_min = torch.min(point_coords, dim=0).values
        coords_max = torch.max(point_coords, dim=0).values

        # Create latent queries
        latent_query_arr = [torch.linspace(coords_min[i], coords_max[i], args.latent_query_dims[i], device=args.device) for i in range(args.coord_dim)]
        latent_queries = torch.stack(torch.meshgrid(*latent_query_arr, indexing='ij'), dim=-1)

        # Concatenate core and ghost input observations
        input_obs = torch.concat([patch_data['core_in'], patch_data['ghost_in']], dim=1).float()
        output_obs = torch.concat([patch_data['core_out'], patch_data['ghost_out']], dim=1).float()
        
        # Forward pass
        model_output = model(input_geom=point_coords.to(args.device),
                            latent_queries=latent_queries.to(args.device),
                            x=input_obs.to(args.device),
                            output_queries=point_coords.to(args.device))

        # Evaluate loss on core points
        core_len = len(patch_data['core_coords'])
        core_output = model_output[:core_len]
        core_target = output_obs[:core_len].to(args.device)
        core_loss = loss_fn(core_output, core_target)
        print(f"Sample {sample_idx} Core loss: {core_loss.item()}")
        batch_loss += core_loss
    
    # Backward pass
    batch_loss /= len(batch_indices)
    batch_loss.backward()

    # Update model parameters
    optimizer.step()

    # Zero gradients
    optimizer.zero_grad()

    return batch_loss.item()
 

def evaluate_model_on_patches(val_patch_ds, model, loss_fn, args):
    """Evaluate model on patches."""
    with torch.no_grad():
        val_loss = 0.0
        for sample_idx in range(len(val_patch_ds)):
            patch_data = val_patch_ds[sample_idx]
            # Calculate min and max of point coordinates
            coords_min = torch.min(patch_data['core_coords'], dim=0).values
            coords_max = torch.max(patch_data['core_coords'], dim=0).values
            # Concatenate core and ghost coordinates
            point_coords = torch.concat([patch_data['core_coords'], patch_data['ghost_coords']], dim=0).float()
            # Create latent queries
            latent_query_arr = [torch.linspace(coords_min[i], coords_max[i], args.latent_query_dims[i], device=args.device) for i in range(args.coord_dim)]
            latent_queries = torch.stack(torch.meshgrid(*latent_query_arr, indexing='ij'), dim=-1)
            # Concatenate core and ghost input observations
            input_obs = torch.concat([patch_data['core_in'], patch_data['ghost_in']], dim=1).float()
            output_obs = torch.concat([patch_data['core_out'], patch_data['ghost_out']], dim=1).float()
            # Forward pass
            model_output = model(input_geom=point_coords.to(args.device),
                                 latent_queries=latent_queries.to(args.device),
                                 x=input_obs.to(args.device),
                                 output_queries=point_coords.to(args.device))
            
            # Evaluate loss on core points
            core_len = len(patch_data['core_coords'])
            core_output = model_output[:core_len]
            core_target = output_obs[:core_len].to(args.device)
            core_loss = loss_fn(core_output, core_target)
            val_loss += core_loss.item()

        val_loss /= len(val_patch_ds)
        return val_loss


def train_gino_on_patches(train_patch_ds, val_patch_ds, model, args):
    """Train GINO on patches."""
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Define loss function
    loss_fn = LpLoss(d=3, p=2)

    for epoch in range(args.epochs):
        
        print(f"Epoch {epoch+1} of {args.epochs}")

        # Randomly ordered index for patch dataset
        indices = torch.randperm(len(train_patch_ds))

        for batch_idx in range(0, len(indices), args.batch_size):

            # Get batch indices
            batch_indices = indices[batch_idx:batch_idx+args.batch_size]

            # Train on batch
            batch_loss = train_on_batch(model, train_patch_ds, batch_indices, loss_fn, optimizer, args)
            print(f"Epoch {epoch+1} of {args.epochs}, Batch at {batch_idx//args.batch_size}/{len(indices)//args.batch_size} loss: {batch_loss}")

        # Evaluate model on validation set
        val_loss = evaluate_model_on_patches(val_patch_ds, model, args)
        print(f"Epoch {epoch+1} of {args.epochs}, Validation loss: {val_loss}")
        
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
    obs_transform = calculate_obs_transform(args.raw_data_dir, args.target_cols)
    
    # Create patch dataset with normalization transforms
    train_patch_ds, val_patch_ds = create_patch_datasets(args.patch_data_dir, 
                                    coord_transform, obs_transform,
                                    target_cols_idx=args.target_cols_idx)
    
    print(f"Train dataset length: {len(train_patch_ds)}")
    print(f"Val dataset length: {len(val_patch_ds)}")

    print(f"Target columns: {args.target_cols}, target columns idx: {args.target_cols_idx}")

    # Define GINO model
    model = define_ginos_model(args)
    print(f"Model: {model}")
    
    # Examine the structure of patch data
    model = train_gino_on_patches(train_patch_ds, val_patch_ds, model, args)

    # Save model to file
    torch.save(model.state_dict(), 'gino_model.pth')