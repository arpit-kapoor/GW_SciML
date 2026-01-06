"""
Visualization utilities for neural operator training.

This module provides plotting and visualization functions that can be reused
across different model architectures and training scripts.
"""

import os
import matplotlib.pyplot as plt

from .checkpoint import get_accumulated_losses, save_loss_history


def plot_training_curves(loss_dict, args):
    """
    Plot training and validation loss curves and save to results directory.
    
    This function plots accumulated losses from all training sessions,
    providing a continuous view of training progress across resume operations.
    Handles variable numbers of loss types dynamically.
    
    Args:
        loss_dict (dict): Dictionary of loss histories, e.g.:
            {'train_losses': [...], 'val_losses': [...],
             'train_global_losses': [...], 'val_global_losses': [...]}
        args (argparse.Namespace): Argument namespace containing results directory
    """
    # Get accumulated losses from all training sessions
    checkpoint_path = getattr(args, 'resume_from', None)
    accumulated = get_accumulated_losses(loss_dict, args, checkpoint_path)
    
    # Extract main losses (required)
    accumulated_train = accumulated.get('train_losses', [])
    accumulated_val = accumulated.get('val_losses', [])
    
    if not accumulated_train or not accumulated_val:
        print("Warning: No loss data to plot")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(accumulated_train) + 1)
    
    # Plot main losses (solid lines)
    plt.plot(epochs, accumulated_train, 'b-', label='Training Loss', 
             linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, accumulated_val, 'r-', label='Validation Loss', 
             linewidth=2, marker='s', markersize=4)
    
    # Dynamically plot additional loss components if they exist
    additional_losses = {
        'train_global_losses': ('b--', 'Training Global Loss', '^'),
        'train_conc_var_losses': ('b-.', 'Training Conc Variance Loss', 'v'),
        'val_global_losses': ('r--', 'Validation Global Loss', 'd'),
        'val_conc_var_losses': ('r-.', 'Validation Conc Variance Loss', 'x'),
    }
    
    for loss_key, (style, label, marker) in additional_losses.items():
        if loss_key in accumulated and accumulated[loss_key]:
            loss_data = accumulated[loss_key]
            # Calculate offset for losses that start later than main losses
            offset = len(accumulated_train) - len(loss_data)
            loss_epochs = range(offset + 1, len(accumulated_train) + 1)
            plt.plot(loss_epochs, loss_data, style, label=label, 
                    linewidth=2, marker=marker, markersize=4)
    
    # Add visual indicators for resume points if this is a resumed session
    if hasattr(args, 'resume_from') and args.resume_from is not None:
        current_train = loss_dict.get('train_losses', [])
        if len(accumulated_train) > len(current_train):
            existing_count = len(accumulated_train) - len(current_train)
            if existing_count > 0:
                plt.axvline(x=existing_count + 0.5, color='gray', linestyle='--', 
                           alpha=0.7, linewidth=1)
                max_loss = max(max(accumulated_train), max(accumulated_val))
                plt.text(existing_count + 0.5, max_loss * 0.9, 
                        'Resume', rotation=90, ha='right', va='top', 
                        fontsize=10, alpha=0.7)
    
    plt.title('Training and Validation Loss Over Epochs (Accumulated)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics as text
    stats_text = _format_loss_statistics(accumulated)
    if stats_text:
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot to results directory
    plot_path = os.path.join(args.results_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to '{plot_path}' (total epochs: {len(accumulated_train)})")
    
    # Save the accumulated loss history for future sessions
    save_loss_history(accumulated, args)


def _format_loss_statistics(accumulated):
    """
    Format summary statistics for loss values.
    
    Args:
        accumulated (dict): Dictionary of accumulated loss histories
        
    Returns:
        str: Formatted statistics text
    """
    accumulated_train = accumulated.get('train_losses', [])
    accumulated_val = accumulated.get('val_losses', [])
    
    if not accumulated_train:
        return ""
    
    stats = [
        f'Total Epochs: {len(accumulated_train)}',
        f'Min Train Loss: {min(accumulated_train):.4f}',
        f'Min Val Loss: {min(accumulated_val):.4f}'
    ]
    
    # Add stats for other loss components if they exist
    additional_stats = {
        'train_global_losses': 'Min Train Global Loss',
        'val_global_losses': 'Min Val Global Loss',
        'train_conc_var_losses': 'Min Train Conc Variance Loss',
        'val_conc_var_losses': 'Min Val Conc Variance Loss',
    }
    
    for loss_key, label in additional_stats.items():
        if loss_key in accumulated and accumulated[loss_key]:
            min_val = min(accumulated[loss_key])
            stats.append(f'{label}: {min_val:.4f}')
    
    return '\n'.join(stats)
