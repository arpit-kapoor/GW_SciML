"""
Generic trainer module for neural operator models.

This module provides reusable training and evaluation loops that can work with
different model architectures, loss functions, and data loaders.
"""

import datetime as dt
import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint, load_checkpoint
from .visualization import plot_training_curves
from .parallel_utils import unwrap_dp, broadcast_static_inputs_for_dp


def train_model(
    train_loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    loss_fn,
    args,
    forward_fn=None,
    extract_core_fn=None
):
    """
    Generic training loop for neural operator models.
    
    This function provides a flexible training loop that can be customized via
    callback functions for model-specific forward passes and loss computation.
    
    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        model (torch.nn.Module): Model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        scheduler: Learning rate scheduler
        loss_fn (callable): Loss function that accepts (y_pred, y_true, weights)
            and returns (total_loss, *loss_components)
        args (argparse.Namespace): Training configuration arguments
        forward_fn (callable): Optional custom forward function. If None, uses default GINO forward.
            Should accept (model, batch, args) and return predictions.
        extract_core_fn (callable): Optional function to extract core points from predictions.
            Should accept (outputs, targets, batch) and return (core_outputs, core_targets, core_weights).
            If None, uses default extraction for patch-based training.
    
    Returns:
        torch.nn.Module: Trained model
    """
    # Use default forward and extraction functions if not provided
    if forward_fn is None:
        forward_fn = _default_gino_forward
    if extract_core_fn is None:
        extract_core_fn = _default_extract_core_points
    
    # Initialize training state
    start_epoch = 0
    loss_dict = {
        'train_losses': [],
        'train_global_losses': [],
        'train_conc_var_losses': [],
        'val_losses': [],
        'val_global_losses': [],
        'val_conc_var_losses': []
    }
    
    # Load checkpoint if resuming training
    if hasattr(args, 'resume_from') and args.resume_from is not None:
        from .parallel_utils import unwrap_model_for_state_dict
        start_epoch, loss_dict = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, args,
            unwrap_fn=unwrap_model_for_state_dict
        )
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) "
              f"Training Epoch {epoch+1} of {args.epochs}")
        print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train for one epoch
        epoch_losses = _train_one_epoch(
            train_loader, model, optimizer, loss_fn, args, 
            forward_fn, extract_core_fn
        )
        
        # Record training losses
        for key, value in epoch_losses.items():
            train_key = f'train_{key}' if not key.startswith('train_') else key
            if train_key in loss_dict:
                loss_dict[train_key].append(value)
        
        # Evaluate on validation set
        val_losses = evaluate_model(
            val_loader, model, loss_fn, args, forward_fn, extract_core_fn
        )
        
        # Record validation losses
        for key, value in val_losses.items():
            val_key = f'val_{key}' if not key.startswith('val_') else key
            if val_key in loss_dict:
                loss_dict[val_key].append(value)
        
        # Print epoch summary
        train_loss = epoch_losses.get('losses', epoch_losses.get('train_losses', 0))
        val_loss = val_losses.get('losses', val_losses.get('val_losses', 0))
        print(f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) "
              f"Epoch {epoch+1} - Training Loss: {train_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_checkpoint_every == 0:
            from .parallel_utils import unwrap_model_for_state_dict
            save_checkpoint(
                model, optimizer, scheduler, epoch, loss_dict, args,
                unwrap_fn=unwrap_model_for_state_dict
            )
        
        # Step the learning rate scheduler
        _step_scheduler(scheduler, args, epoch)
    
    # Save final checkpoint
    from .parallel_utils import unwrap_model_for_state_dict
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, loss_dict, args,
        filename='final_checkpoint.pth', unwrap_fn=unwrap_model_for_state_dict
    )
    
    # Plot and save training curves
    plot_training_curves(loss_dict, args)
    
    return model


def _train_one_epoch(
    train_loader, model, optimizer, loss_fn, args, forward_fn, extract_core_fn
):
    """Train for one epoch."""
    model.train()
    
    epoch_losses = {
        'losses': 0.0,
        'global_losses': 0.0,
        'conc_var_losses': 0.0
    }
    total_samples = 0
    
    for step_idx, batch in enumerate(train_loader, start=1):
        # Forward pass
        outputs = forward_fn(model, batch, args)
        
        # Extract core points and compute loss
        core_outputs, core_targets, core_weights = extract_core_fn(outputs, batch, args)
        loss_result = loss_fn(core_outputs, core_targets, core_weights)
        
        # Unpack loss (support different return formats)
        if isinstance(loss_result, tuple):
            loss = loss_result[0]
            loss_components = loss_result[1:] if len(loss_result) > 1 else ()
        else:
            loss = loss_result
            loss_components = ()
        
        # Accumulate losses
        batch_size = batch['x'].shape[0]
        epoch_losses['losses'] += loss.item() * batch_size
        
        if len(loss_components) >= 2:
            epoch_losses['global_losses'] += loss_components[0].item() * batch_size
            epoch_losses['conc_var_losses'] += loss_components[1].item() * batch_size
        
        total_samples += batch_size
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping if enabled
        if hasattr(args, 'grad_clip_norm') and args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        
        optimizer.step()
        
        # Log progress
        if step_idx % max(1, len(train_loader) // 10) == 0:
            print(f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) "
                  f"Step: {step_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
    
    # Calculate average losses
    for key in epoch_losses:
        epoch_losses[key] /= max(total_samples, 1)
    
    return epoch_losses


def evaluate_model(val_loader, model, loss_fn, args, forward_fn=None, extract_core_fn=None):
    """
    Evaluate model on validation set.
    
    Args:
        val_loader (DataLoader): Validation data loader
        model (torch.nn.Module): Model to evaluate
        loss_fn (callable): Loss function
        args (argparse.Namespace): Configuration arguments
        forward_fn (callable): Optional custom forward function
        extract_core_fn (callable): Optional function to extract core points
    
    Returns:
        dict: Dictionary of validation losses
    """
    # Use default functions if not provided
    if forward_fn is None:
        forward_fn = _default_gino_forward
    if extract_core_fn is None:
        extract_core_fn = _default_extract_core_points
    
    model.eval()
    
    val_losses = {
        'losses': 0.0,
        'global_losses': 0.0,
        'conc_var_losses': 0.0
    }
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Forward pass
            outputs = forward_fn(model, batch, args)
            
            # Extract core points and compute loss
            core_outputs, core_targets, core_weights = extract_core_fn(outputs, batch, args)
            loss_result = loss_fn(core_outputs, core_targets, core_weights)
            
            # Unpack loss
            if isinstance(loss_result, tuple):
                loss = loss_result[0]
                loss_components = loss_result[1:] if len(loss_result) > 1 else ()
            else:
                loss = loss_result
                loss_components = ()
            
            # Accumulate losses
            batch_size = batch['x'].shape[0]
            val_losses['losses'] += loss.item() * batch_size
            
            if len(loss_components) >= 2:
                val_losses['global_losses'] += loss_components[0].item() * batch_size
                val_losses['conc_var_losses'] += loss_components[1].item() * batch_size
            
            total_samples += batch_size
    
    # Calculate average losses
    for key in val_losses:
        val_losses[key] /= max(total_samples, 1)
    
    model.train()
    return val_losses


def _default_gino_forward(model, batch, args):
    """Default forward pass for GINO models with patch-based data."""
    # Move batch data to device
    point_coords = batch['point_coords'].to(args.device).float()
    latent_queries = batch['latent_queries'].to(args.device).float()
    x = batch['x'].to(args.device).float()
    
    batch_size = x.shape[0]
    
    # Broadcast static inputs for DataParallel compatibility
    input_geom_b, latent_queries_b, output_queries_b = broadcast_static_inputs_for_dp(
        point_coords, latent_queries, batch_size
    )
    
    # Unwrap model if in eval mode (avoid DP replication)
    if not model.training:
        model = unwrap_dp(model)
    
    # Forward pass
    outputs = model(
        input_geom=input_geom_b,
        latent_queries=latent_queries_b,
        x=x,
        output_queries=output_queries_b,
    )
    
    return outputs


def _default_extract_core_points(outputs, batch, args):
    """Default function to extract core points from predictions."""
    y = batch['y'].to(args.device).float()
    
    # Extract core points only (exclude ghost points)
    core_len = batch['core_len']
    core_outputs = outputs[:, :core_len]
    core_targets = y[:, :core_len]
    
    # Extract weights
    weights = batch['weights'].to(args.device).float()
    core_weights = weights[:core_len]
    
    return core_outputs, core_targets, core_weights


def _step_scheduler(scheduler, args, epoch):
    """Step the learning rate scheduler based on configuration."""
    if hasattr(args, 'scheduler_type') and args.scheduler_type == 'cosine':
        # Cosine scheduler steps every epoch
        scheduler.step()
        print(f"Learning rate updated to: {scheduler.get_last_lr()[0]:.6f}")
    else:
        # Exponential scheduler steps at specified intervals
        if hasattr(args, 'lr_scheduler_interval'):
            if (epoch + 1) % args.lr_scheduler_interval == 0:
                scheduler.step()
                print(f"Learning rate updated to: {scheduler.get_last_lr()[0]:.6f}")
