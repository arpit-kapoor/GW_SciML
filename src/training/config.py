"""
Configuration utilities for training neural operator models.

This module provides generic configuration functions that can be reused across
different model architectures and training scripts.
"""

import argparse
import datetime as dt
import os
import torch


def setup_training_arguments(
    description='Train model on groundwater patches',
    default_base_data_dir=None,
    default_results_dir=None,
    add_model_specific_args=None
):
    """
    Parse command line arguments for generic model training.
    
    This function provides a flexible argument parser that can be extended
    for specific model architectures. It handles common arguments like data paths,
    training hyperparameters, checkpointing, and device configuration.
    
    Args:
        description (str): Description for the argument parser
        default_base_data_dir (str): Default base data directory path
        default_results_dir (str): Default results directory path
        add_model_specific_args (callable): Optional function to add model-specific arguments.
            Should accept parser as an argument and return it.
    
    Returns:
        argparse.Namespace: Parsed arguments with computed paths and configurations
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Data directories
    parser.add_argument('--base-data-dir', type=str, 
                       default=default_base_data_dir,
                       help='Base data directory')
    parser.add_argument('--raw-data-subdir', type=str, default='all',
                       help='Raw data subdirectory name')
    parser.add_argument('--patch-data-subdir', type=str, default='filter_patch_all_ts',
                       help='Patch data subdirectory name')
    parser.add_argument('--results-dir', type=str, default=default_results_dir,
                       help='Directory to save trained models and results')

    # Target observation columns (multiple)
    parser.add_argument('--target-cols', type=str, nargs='+', 
                       default=['mass_concentration', 'head'],
                       help='List of target observation column names (e.g., mass_concentration head pressure)')
    
    parser.add_argument('--forcings-required', action='store_true', default=False,
                       help='Indicates if forcings data is required for the model (default: False)')

    
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
    
    # Variance-aware loss parameters
    parser.add_argument('--lambda-conc-focus', type=float, default=0.5,
                       help='Lambda parameter to focus on concentration in variance-aware loss')
    parser.add_argument('--var-aware-alpha', type=float, default=0.3,
                       help='Alpha parameter for variance-aware loss')
    parser.add_argument('--var-aware-beta', type=float, default=2.0,
                       help='Beta parameter for variance-aware loss')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (cuda, cpu, or auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Add model-specific arguments if provided
    if add_model_specific_args is not None:
        parser = add_model_specific_args(parser)
    
    args = parser.parse_args()
    
    # Construct full paths and add them to args
    if args.base_data_dir is not None:
        args.raw_data_dir = os.path.join(args.base_data_dir, args.raw_data_subdir)
        args.patch_data_dir = os.path.join(args.base_data_dir, args.patch_data_subdir)
    else:
        args.raw_data_dir = None
        args.patch_data_dir = None

    # Create results directory with timestamp (unless resuming)
    if args.results_dir is not None:
        args.results_dir = setup_results_directory(args)
    
    # Set up checkpoint directory
    if args.checkpoint_dir is None and args.results_dir is not None:
        args.checkpoint_dir = os.path.join(args.results_dir, 'checkpoints')
    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Configure device
    args = configure_device(args)
    
    # Print data directories for verification
    if args.base_data_dir is not None:
        print(f"Base data directory: {args.base_data_dir}")
        print(f"Raw data directory: {args.raw_data_dir}")
        print(f"Patch filtered data directory: {args.patch_data_dir}")
    if args.results_dir is not None:
        print(f"Results directory: {args.results_dir}")
    
    return args


def setup_results_directory(args):
    """
    Setup results directory with timestamp for fresh training or extract from checkpoint path when resuming.
    
    Args:
        args (argparse.Namespace): Arguments containing resume_from and results_dir
        
    Returns:
        str: Path to the results directory
    """
    if args.resume_from is None:
        # Create new timestamped directory for fresh training
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(args.results_dir, f'training_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        print(f"Starting fresh training, created results directory: {results_dir}")
        return results_dir
    else:
        # When resuming, extract the actual run directory from checkpoint path
        checkpoint_path = os.path.abspath(args.resume_from)
        
        # Extract the actual results directory (specific run dir) from checkpoint path
        if 'checkpoints' in checkpoint_path:
            # Path like: /path/to/exp_name/training_timestamp/checkpoints/checkpoint.pth
            results_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        else:
            # Path like: /path/to/exp_name/training_timestamp/checkpoint.pth
            results_dir = os.path.dirname(checkpoint_path)
        
        # Ensure the checkpoint exists and is accessible
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Use the actual run directory extracted from checkpoint path
        # This ensures we continue writing to the same run directory
        print(f"Resuming training, using run directory: {results_dir}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Validate that we're resuming within the expected experiment structure
        experiment_dir = os.path.dirname(results_dir)
        run_dir_name = os.path.basename(results_dir)
        print(f"Experiment directory: {experiment_dir}")
        print(f"Run directory: {run_dir_name}")
        
        return results_dir


def configure_device(args):
    """
    Configure computation device: cpu, cuda, or mps (Apple Silicon).
    
    Automatically detects the best available device if 'auto' is specified.
    Validates that the requested device is available.
    
    Args:
        args (argparse.Namespace): Argument namespace with device preference
        
    Returns:
        argparse.Namespace: Modified args with device configured
        
    Raises:
        ValueError: If requested device is not available
    """
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Auto-selected device: {args.device}")
    elif args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but not available")
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError("MPS device requested but not available")
    return args


def configure_target_col_indices(args, available_columns=None):
    """
    Configure target observation column indices from names.

    Maps human-readable column names to array indices for the observation data.
    This is generic and can work with any set of available columns.
    
    Args:
        args (argparse.Namespace): Argument namespace with target column names
        available_columns (dict): Optional mapping from column names to indices.
            If None, uses default groundwater columns.
        
    Returns:
        argparse.Namespace: Modified args with target_col_indices added
        
    Raises:
        ValueError: If a target column is not in available columns
    """
    # Default mapping for groundwater data
    if available_columns is None:
        available_columns = {
            'mass_concentration': 0,
            'head': 1,
            'pressure': 2
        }
    
    # Validate that all target columns are available
    for col in args.target_cols:
        if col not in available_columns:
            raise ValueError(f"Target column '{col}' not found in available columns: {list(available_columns.keys())}")
    
    # Convert target column names to indices
    args.target_col_indices = [available_columns[col] for col in args.target_cols]
    
    print(f"Target columns: {args.target_cols}")
    print(f"Target column indices: {args.target_col_indices}")
    
    return args
