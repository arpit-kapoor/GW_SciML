"""
Inference utilities for neural operator models.

This package provides generic inference capabilities including:
- Model loading from checkpoints
- Prediction generation
- Results management
- Visualization utilities
"""

from .inference_utils import (
    setup_inference_arguments,
    load_checkpoint,
    create_model_from_checkpoint,
    generate_predictions,
)

from .results import (
    create_results_directory,
    save_predictions_to_disk,
    save_metadata,
    organize_and_save_results,
)

from .visualization import (
    create_scatter_comparison_plots,
    create_timeseries_comparison_plots,
    create_error_analysis_plots,
    create_3d_spatial_plots,
    create_video_from_images,
    create_all_visualizations,
    select_nodes_by_variance,
)

from .metrics import (
    compute_kge,
    denormalize_observations,
    compute_metrics,
    save_metrics,
)

__all__ = [
    # Inference utilities
    'setup_inference_arguments',
    'load_checkpoint',
    'create_model_from_checkpoint',
    'generate_predictions',
    
    # Results management
    'create_results_directory',
    'save_predictions_to_disk',
    'save_metadata',
    'organize_and_save_results',
    
    # Metrics
    'compute_kge',
    'denormalize_observations',
    'compute_metrics',
    'save_metrics',
    
    # Visualization
    'create_scatter_comparison_plots',
    'create_timeseries_comparison_plots',
    'create_error_analysis_plots',
    'select_nodes_by_variance',
    'create_3d_spatial_plots',
    'create_video_from_images',
    'create_all_visualizations',
]
