"""
Checkpoint management utilities for neural operator training.

This module provides generic checkpoint save/load functionality that can be reused
across different model architectures and training configurations.
"""

import datetime as dt
import json
import os
import torch


def save_checkpoint(
    model, 
    optimizer, 
    scheduler, 
    epoch, 
    loss_dict,
    args, 
    filename=None,
    unwrap_fn=None
):
    """
    Save a training checkpoint to enable resuming training.
    
    This function is generic and works with any model architecture that follows
    the standard PyTorch training pattern.
    
    Args:
        model (torch.nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer state to save
        scheduler: Learning rate scheduler state to save
        epoch (int): Current epoch number (0-indexed)
        loss_dict (dict): Dictionary of loss histories, e.g.:
            {'train_losses': [...], 'val_losses': [...], 
             'train_global_losses': [...], 'val_global_losses': [...]}
        args (argparse.Namespace): Argument namespace containing checkpoint directory
        filename (str): Optional custom filename for checkpoint
        unwrap_fn (callable): Optional function to unwrap model from DataParallel/adapters
            before saving state_dict. If None, uses default unwrapping.
    """
    if filename is None:
        filename = f'checkpoint_epoch_{epoch:04d}.pth'
    
    checkpoint_path = os.path.join(args.checkpoint_dir, filename)
    
    # Unwrap model if needed (e.g., from DataParallel wrapper)
    if unwrap_fn is not None:
        model_to_save = unwrap_fn(model)
    else:
        model_to_save = _default_unwrap_model(model)
    
    # Save all training state
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        **loss_dict,  # Include all loss histories
        'args': args,  # Save training configuration for compatibility checking
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Also save as latest checkpoint for easy resuming
    latest_path = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    print(f"Latest checkpoint saved: {latest_path}")
    
    # Save accumulated loss history for continuous training curves
    accumulated_losses = get_accumulated_losses(loss_dict, args, checkpoint_path)
    
    # Save loss history next to the checkpoint file
    checkpoint_loss_history = os.path.join(os.path.dirname(checkpoint_path), 'loss_history.json')
    loss_history = {
        **accumulated_losses,
        'last_updated': dt.datetime.now().isoformat(),
        'total_epochs': len(accumulated_losses.get('train_losses', [])),
        'checkpoint_file': os.path.basename(checkpoint_path)
    }
    with open(checkpoint_loss_history, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"Loss history saved with checkpoint: {checkpoint_loss_history}")


def load_checkpoint(
    checkpoint_path, 
    model, 
    optimizer, 
    scheduler, 
    args,
    unwrap_fn=None,
    validate_compatibility=True
):
    """
    Load a training checkpoint to resume training.
    
    This function is generic and works with any model architecture that follows
    the standard PyTorch training pattern.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (torch.nn.Module): Model to load state into
        optimizer (torch.optim.Optimizer): Optimizer to load state into
        scheduler: Learning rate scheduler to load state into
        args (argparse.Namespace): Current argument namespace for compatibility checking
        unwrap_fn (callable): Optional function to unwrap model from DataParallel/adapters
            before loading state_dict. If None, uses default unwrapping.
        validate_compatibility (bool): Whether to validate checkpoint compatibility
        
    Returns:
        tuple: (start_epoch, loss_dict) where loss_dict contains all loss histories
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # Validate compatibility if requested
    if validate_compatibility:
        saved_args = checkpoint['args']
        validate_checkpoint_compatibility(saved_args, args)
    
    # Unwrap model if needed
    if unwrap_fn is not None:
        model_to_load = unwrap_fn(model)
    else:
        model_to_load = _default_unwrap_model(model)
    
    # Load model state
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    print("Model state loaded successfully")
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Optimizer state loaded successfully")
    
    # Load scheduler state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print("Scheduler state loaded successfully")
    
    # Get training progress
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    
    # Extract all loss histories from checkpoint
    loss_dict = {}
    for key in checkpoint.keys():
        if 'loss' in key.lower() or key in ['train_losses', 'val_losses']:
            loss_dict[key] = checkpoint.get(key, [])
    
    # Backward compatibility: Ensure we have empty lists if losses are missing
    for loss_key in ['train_losses', 'val_losses']:
        if loss_key not in loss_dict:
            loss_dict[loss_key] = []
    
    print(f"Resuming training from epoch {start_epoch}")
    print(f"Loaded {len(loss_dict.get('train_losses', []))} training losses and " 
          f"{len(loss_dict.get('val_losses', []))} validation losses")
    
    return start_epoch, loss_dict


def validate_checkpoint_compatibility(saved_args, current_args, critical_params=None):
    """
    Validate that checkpoint is compatible with current training setup.
    
    Args:
        saved_args (argparse.Namespace): Arguments from saved checkpoint
        current_args (argparse.Namespace): Current training arguments
        critical_params (list): List of parameter names that must match. If None,
            uses a generic default set.
        
    Raises:
        ValueError: If incompatible configuration detected
    """
    # Default critical parameters if none provided
    if critical_params is None:
        critical_params = [
            'input_window_size', 'output_window_size', 'target_col_indices', 
            'n_target_cols'
        ]
    
    for param in critical_params:
        if hasattr(saved_args, param) and hasattr(current_args, param):
            saved_value = getattr(saved_args, param)
            current_value = getattr(current_args, param)
            if saved_value != current_value:
                raise ValueError(
                    f"Checkpoint incompatibility detected: "
                    f"saved {param}={saved_value}, current {param}={current_value}"
                )
    
    print("Checkpoint compatibility validated successfully")


def save_loss_history(loss_dict, args):
    """
    Save loss history to a persistent JSON file.
    
    Args:
        loss_dict (dict): Dictionary containing all loss histories
        args (argparse.Namespace): Argument namespace containing results directory
    """
    loss_history_path = os.path.join(args.results_dir, 'loss_history.json')
    
    # Create loss history dictionary with metadata
    loss_history = {
        **loss_dict,
        'last_updated': dt.datetime.now().isoformat(),
        'total_epochs': len(loss_dict.get('train_losses', []))
    }
    
    # Save to JSON file
    with open(loss_history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"Loss history saved to '{loss_history_path}' "
          f"(total epochs: {len(loss_dict.get('train_losses', []))})")


def load_loss_history(args, checkpoint_path=None):
    """
    Load loss history from persistent JSON file.
    
    When resuming training (checkpoint_path provided), tries to load loss history from:
    1. The checkpoint directory first (most accurate for resuming)
    2. Falls back to results directory if not found
    3. Returns empty history if neither exists
    
    Args:
        args (argparse.Namespace): Argument namespace containing results directory
        checkpoint_path (str): Optional path to checkpoint file being loaded
        
    Returns:
        dict: Dictionary containing loss histories and metadata
    """
    # First try to load from checkpoint directory if resuming
    if checkpoint_path is not None:
        checkpoint_loss_history = os.path.join(
            os.path.dirname(checkpoint_path), 'loss_history.json'
        )
        if os.path.exists(checkpoint_loss_history):
            try:
                with open(checkpoint_loss_history, 'r') as f:
                    history = json.load(f)
                print(f"Loaded loss history from checkpoint directory: {checkpoint_loss_history}")
                print(f"Total epochs in history: {history.get('total_epochs', 0)}")
                return history
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load checkpoint loss history from "
                      f"'{checkpoint_loss_history}': {e}")
    
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
            print(f"Warning: Could not load results loss history from "
                  f"'{results_loss_history}': {e}")
    
    # Return empty history if no valid history found
    print("No existing loss history found, starting fresh.")
    return {'train_losses': [], 'val_losses': [], 'total_epochs': 0}


def get_accumulated_losses(loss_dict, args, checkpoint_path=None):
    """
    Get accumulated losses from all training sessions.
    
    This function loads the persistent loss history and properly merges it with
    the current session's losses to maintain continuity across all training sessions.
    It uses delta-merging to prevent duplication of epochs.
    
    Args:
        loss_dict (dict): Current session's loss histories
        args (argparse.Namespace): Argument namespace containing results directory
        checkpoint_path (str): Optional path to checkpoint file being loaded when resuming
        
    Returns:
        dict: Dictionary of accumulated loss histories
    """
    # Load existing loss history, prioritizing checkpoint directory when resuming
    existing_history = load_loss_history(args, checkpoint_path)
    
    # Prepare accumulated losses dictionary
    accumulated = {}
    
    # Process each loss type in the current session
    for loss_key, current_losses in loss_dict.items():
        if not isinstance(current_losses, list):
            continue
            
        existing_losses = existing_history.get(loss_key, [])
        existing_count = len(existing_losses)
        current_count = len(current_losses)
        
        # If we have more epochs than what's saved, append only the new ones
        if current_count > existing_count:
            new_losses = current_losses[existing_count:]
            accumulated[loss_key] = existing_losses + new_losses
        else:
            # Current session has same or fewer epochs - use existing
            accumulated[loss_key] = existing_losses
        
        # Filter out None placeholders from old checkpoints (backward compatibility)
        if loss_key != 'train_losses' and loss_key != 'val_losses':
            accumulated[loss_key] = [x for x in accumulated[loss_key] if x is not None]
    
    return accumulated


def _default_unwrap_model(model):
    """
    Default function to unwrap model from DataParallel or other wrappers.
    
    Args:
        model (torch.nn.Module): Potentially wrapped model
        
    Returns:
        torch.nn.Module: Unwrapped model
    """
    m = model
    # Unwrap DataParallel
    if hasattr(m, 'module'):
        m = m.module
    # Unwrap custom adapter (e.g., GINODataParallelAdapter)
    if hasattr(m, 'inner'):
        m = m.inner
    return m
