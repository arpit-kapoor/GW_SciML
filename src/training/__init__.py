"""
Training utilities for neural operator models.

This package contains modular components for training, evaluation,
checkpointing, and visualization that can be reused across different
model architectures.
"""

from .config import (
    setup_training_arguments,
    configure_device,
    configure_target_col_indices,
)

from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_loss_history,
    load_loss_history,
    get_accumulated_losses,
)

from .visualization import plot_training_curves

from .parallel_utils import (
    DataParallelAdapter,
    unwrap_dp,
    unwrap_model_for_state_dict,
    broadcast_static_inputs_for_dp,
)

from .trainer import train_model, evaluate_model

__all__ = [
    # Config
    'setup_training_arguments',
    'configure_device',
    'configure_target_col_indices',
    # Checkpoint
    'save_checkpoint',
    'load_checkpoint',
    'save_loss_history',
    'load_loss_history',
    'get_accumulated_losses',
    # Visualization
    'plot_training_curves',
    # Parallel utils
    'GINODataParallelAdapter',
    'unwrap_dp',
    'unwrap_model_for_state_dict',
    'broadcast_static_inputs_for_dp',
    # Trainer
    'train_model',
    'evaluate_model',
]
