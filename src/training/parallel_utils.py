"""
DataParallel utilities for neural operator models.

This module provides helper functions and adapters for using PyTorch DataParallel
with models that have static (non-batched) inputs like point coordinates and query grids.
"""

import torch


class DataParallelAdapter(torch.nn.Module):
    """
    Thin wrapper to make neural operator models compatible with DataParallel.
    
    DataParallel expects to split all inputs along the batch dimension. For models like
    GINO and FNOInterpolate, we have both dynamic inputs (x) and static inputs 
    (point_coords, latent_queries) that should be shared across all replicas. This 
    adapter handles the fake batch dimension added to static inputs and strips it 
    before forwarding to the model.
    
    Works with: GINO, FNOInterpolate, and other coordinate-based neural operators.
    
    Usage:
        model = GINO(...)  # or FNOInterpolate(...)
        model = DataParallelAdapter(model)
        model = torch.nn.DataParallel(model)
    """
    def __init__(self, inner: torch.nn.Module):
        """
        Initialize the adapter.
        
        Args:
            inner (torch.nn.Module): The actual model to wrap (GINO, FNOInterpolate, etc.)
        """
        super().__init__()
        self.inner = inner  # the actual model

    def forward(self, input_geom, latent_queries, x, output_queries):
        """
        Forward pass that strips fake batch dimensions from static inputs.
        
        Note: Uses positional args (not keyword-only) for DataParallel compatibility.
        
        Args:
            input_geom (torch.Tensor): Input geometry [B, N_points, coord_dim] or [N_points, coord_dim]
            latent_queries (torch.Tensor): Latent queries [B, Qx, Qy, Qz, coord_dim] or [Qx, Qy, Qz, coord_dim]
            x (torch.Tensor): Input features [B, N_points, in_channels]
            output_queries (torch.Tensor): Output queries [B, N_points, coord_dim] or [N_points, coord_dim]
            
        Returns:
            torch.Tensor: Model predictions
        """
        # If these were broadcast with a fake batch dimension, drop it per-replica.
        if input_geom.dim() == 3:   # [B, N_points, coord_dim] -> [N_points, coord_dim]
            input_geom = input_geom[0]
        if output_queries.dim() == 3:  # [B, N_points, coord_dim] -> [N_points, coord_dim]
            output_queries = output_queries[0]
        if latent_queries.dim() == 5:  # [B, Qx, Qy, Qz, coord_dim] -> [Qx, Qy, Qz, coord_dim]
            latent_queries = latent_queries[0]
        
        # Forward into the real model.
        return self.inner(
            input_geom=input_geom,
            latent_queries=latent_queries,
            x=x,
            output_queries=output_queries,
        )


def unwrap_dp(model):
    """
    Unwrap model from DataParallel wrapper.
    
    Args:
        model (torch.nn.Module): Potentially wrapped model
        
    Returns:
        torch.nn.Module: Unwrapped model
    """
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def unwrap_model_for_state_dict(model: torch.nn.Module) -> torch.nn.Module:
    """
    Unwrap model from all wrappers (DataParallel and custom adapters) for state dict operations.
    
    This ensures we always save/load the underlying model regardless of wrapping.
    
    Args:
        model (torch.nn.Module): Potentially wrapped model
        
    Returns:
        torch.nn.Module: Fully unwrapped base model
    """
    m = model
    if hasattr(m, 'module'):  # DataParallel wrapper
        m = m.module
    if hasattr(m, 'inner'):   # Custom adapter (like GINODataParallelAdapter)
        m = m.inner
    return m


def broadcast_static_inputs_for_dp(point_coords, latent_queries, batch_size, coord_dim=None):
    """
    Create fake batch dimensions for static inputs to enable DataParallel scattering.
    
    DataParallel expects all inputs to have a batch dimension. This function adds
    a fake batch dimension to static inputs (coordinates and queries) using
    inexpensive expand views (no memory copy).
    
    Args:
        point_coords (torch.Tensor): Point coordinates [N_points, coord_dim]
        latent_queries (torch.Tensor): Latent query grid [Qx, Qy, Qz, coord_dim] or similar
        batch_size (int): Size of the batch dimension to create
        coord_dim (int): Optional coordinate dimensionality for validation
        
    Returns:
        tuple: (input_geom_b, latent_queries_b, output_queries_b) with fake batch dimensions
    """
    # Validate coordinate dimensions if provided
    if coord_dim is not None:
        assert point_coords.shape[-1] == coord_dim, \
            f"Point coords have dim {point_coords.shape[-1]}, expected {coord_dim}"
        assert latent_queries.shape[-1] == coord_dim, \
            f"Latent queries have dim {latent_queries.shape[-1]}, expected {coord_dim}"
    
    # Add batch dimension using expand (creates view, no memory copy)
    input_geom_b = point_coords.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N_points, coord_dim]
    output_queries_b = input_geom_b  # same as input_geom
    
    # For latent queries, preserve all grid dimensions
    latent_queries_b = latent_queries.unsqueeze(0).expand(
        (batch_size,) + latent_queries.shape
    )  # [B, Qx, Qy, Qz, coord_dim] or similar
    
    return input_geom_b, latent_queries_b, output_queries_b
