from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from .conv import SpectralConv
from .mlp import MLP


class FactorizedSpaceTimeSpectralConv(nn.Module):
    """
    Factorized 4D Spectral Convolution using Space-Time splitting.
    Splits a 4D convolution (X, Y, Z, T) into a 3D spatial convolution (X, Y, Z)
    and a parallel 1D temporal convolution (T).
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        max_n_modes=None,
        bias=True,
        n_layers=1,
        init_std="auto",
        fft_norm="backward",
        rank=0.5
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Split modes
        spatial_modes = n_modes[:3]
        temporal_modes = [n_modes[3]]
        
        spatial_max_modes = max_n_modes[:3] if max_n_modes is not None else None
        temporal_max_modes = [max_n_modes[3]] if max_n_modes is not None else None
        
        self.spatial_conv = SpectralConv(
            in_channels, out_channels, spatial_modes, max_n_modes=spatial_max_modes,
            bias=bias, n_layers=n_layers, init_std=init_std, fft_norm=fft_norm, rank=rank
        )
        self.temporal_conv = SpectralConv(
            in_channels, out_channels, temporal_modes, max_n_modes=temporal_max_modes,
            bias=bias, n_layers=n_layers, init_std=init_std, fft_norm=fft_norm, rank=rank
        )

    def forward(self, x, indices=0, output_shape=None):
        """
        x: (batch, channels, x, y, z, t)
        """
        batch_size, channels, dx, dy, dz, dt = x.shape
        
        # Spatial branch: fold T into Batch -> (batch * t, channels, x, y, z)
        x_space = x.permute(0, 5, 1, 2, 3, 4).contiguous().view(batch_size * dt, channels, dx, dy, dz)
        out_shape_space = output_shape[:3] if output_shape is not None else None
        out_space = self.spatial_conv(x_space, indices=indices, output_shape=out_shape_space)
        out_dx, out_dy, out_dz = out_space.shape[2:]
        out_space = out_space.view(batch_size, dt, self.out_channels, out_dx, out_dy, out_dz)
        out_space = out_space.permute(0, 2, 3, 4, 5, 1) # (batch, channels, x, y, z, t)
        
        # Temporal branch: fold X, Y, Z into Batch -> (batch * x * y * z, channels, t)
        x_time = x.permute(0, 2, 3, 4, 1, 5).contiguous().view(batch_size * dx * dy * dz, channels, dt)
        out_shape_time = [output_shape[3]] if output_shape is not None else None
        out_time = self.temporal_conv(x_time, indices=indices, output_shape=out_shape_time)
        out_dt = out_time.shape[2]
        out_time = out_time.view(batch_size, dx, dy, dz, self.out_channels, out_dt)
        out_time = out_time.permute(0, 4, 1, 2, 3, 5) # (batch, channels, x, y, z, t)
        
        return out_space + out_time

    def __getitem__(self, indices):
        return SubFactorizedConv(self, indices)


class SubFactorizedConv(nn.Module):
    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices

    def forward(self, x, **kwargs):
        return self.main_conv.forward(x, self.indices, **kwargs)


class FNOBlocks(nn.Module):
    """
    FNOBlocks performs the sequence of Fourier layers with skip connections and spectral convolutions.
    This class is designed to be used as a building block within the FNO model, encapsulating the
    skip connection and spectral convolution logic.
    """

    def __init__(
        self,
        n_layers,
        n_modes,
        hidden_channels,
        skip_fno_bias=False,
        fft_norm="forward",
        rank=0.5,
        max_n_modes=None,
        non_linearity=F.gelu,
    ):
        """
        Parameters
        ----------
        n_layers : int
            Number of Fourier layers.
        n_modes : tuple
            Number of Fourier modes to use in each dimension.
        hidden_channels : int
            Number of hidden channels in the model.
        skip_fno_bias : bool, optional
            Whether to use bias in the skip connection layers.
        fft_norm : str, optional
            Normalization mode for FFT.
        rank : float, optional
            Rank for low-rank spectral convolution.
        max_n_modes : tuple or None, optional
            Maximum number of modes in each dimension.
        non_linearity : callable, optional
            Non-linearity to use after each Fourier layer except the last.
        """
        super().__init__()
        self.n_layers = n_layers
        self.non_linearity = non_linearity
        self.n_dim = len(n_modes)

        # Create skip connection layers (1x1 convs or linear layers)
        if self.n_dim == 4:
            # PyTorch doesn't have Conv4d, so we use Conv3d and fold Time into Batch during forward
            skip_conv_class = nn.Conv3d
        else:
            skip_conv_class = getattr(nn, f"Conv{self.n_dim}d")

        self.fno_skips = nn.ModuleList(
            [
                skip_conv_class(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    bias=skip_fno_bias,
                )
                for _ in range(n_layers)
            ]
        )

        # Create spectral convolution layers
        if self.n_dim == 4:
            conv_class = FactorizedSpaceTimeSpectralConv
        else:
            conv_class = SpectralConv

        self.convs = nn.ModuleList([
            conv_class(
                hidden_channels,
                hidden_channels,
                n_modes=n_modes,
                fft_norm=fft_norm,
                rank=rank,
                max_n_modes=max_n_modes
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, output_shape=None):
        """
        Forward pass through the FNOBlocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        output_shape : {tuple, list of tuples, None}, optional
            Optionally specify the output shape for odd-shaped inputs.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the FNOBlocks.
        """
        for layer_idx in range(self.n_layers):
            # Compute skip connection for this layer
            if self.n_dim == 4:
                # Fold Time into Batch for Conv3d skip
                b, c, dx, dy, dz, dt = x.shape
                x_skip = x.permute(0, 5, 1, 2, 3, 4).contiguous().view(b * dt, c, dx, dy, dz)
                x_skip_fno = self.fno_skips[layer_idx](x_skip)
                x_skip_fno = x_skip_fno.view(b, dt, self.fno_skips[layer_idx].out_channels, dx, dy, dz).permute(0, 2, 3, 4, 5, 1).contiguous()
            else:
                x_skip_fno = self.fno_skips[layer_idx](x)
            # Apply spectral convolution for this layer
            if isinstance(output_shape, list):
                out_shape = output_shape[layer_idx] if layer_idx < len(output_shape) else None
            else:
                out_shape = output_shape
            x_fno = self.convs[layer_idx](x, output_shape=out_shape)
            # Add skip connection
            x = x_fno + x_skip_fno
            # Apply non-linearity after all but the last layer
            if layer_idx < (self.n_layers - 1):
                x = self.non_linearity(x)
        return x


class FNO(nn.Module):
    """
    Fourier Neural Operator (FNO) model.

    Consists of:
        - A lifting layer (MLP or linear, depending on lifting_channels)
        - A sequence of n Fourier integral operator layers (FNOBlocks)
        - A projection layer (MLP)
    """

    def __init__(self, 
                 n_modes,
                 hidden_channels,
                 in_channels=3,
                 out_channels=1,
                 lifting_channels=256,
                 projection_channels=256,
                 n_layers=4,
                 non_linearity=F.gelu,
                 skip_fno_bias=False,
                 fft_norm="forward",
                 rank=0.5,
                 max_n_modes=None):
        """
        Initialize the FNO model.

        Parameters
        ----------
        n_modes : tuple
            Number of Fourier modes to use in each dimension.
        hidden_channels : int
            Number of hidden channels in the model.
        in_channels : int, optional
            Number of input channels. Default is 3.
        out_channels : int, optional
            Number of output channels. Default is 1.
        lifting_channels : int, optional
            Number of channels in the hidden layer of the lifting MLP. If 0 or None, uses a linear layer.
        projection_channels : int, optional
            Number of channels in the hidden layer of the projection MLP.
        n_layers : int, optional
            Number of Fourier layers.
        non_linearity : callable, optional
            Non-linearity to use after each Fourier layer except the last.
        skip_fno_bias : bool, optional
            Whether to use bias in the skip connection convolutions.
        fft_norm : str, optional
            Normalization mode for FFT.
        rank : float, optional
            Rank for low-rank spectral convolution.
        max_n_modes : tuple or None, optional
            Maximum number of modes to use in each dimension.
        """
        super(FNO, self).__init__()
        self.n_dim = len(n_modes)

        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity

        # Lifting layer: if lifting_channels is set, use a 2-layer MLP; otherwise, use a single linear layer.
        if self.lifting_channels:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
            )
        else:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
            )
        
        # Projection layer: always a 2-layer MLP with specified hidden size and non-linearity.
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

        # Use FNOBlocks for the sequence of Fourier layers with skip connections
        self.fno_blocks = FNOBlocks(
            n_layers=n_layers,
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            skip_fno_bias=skip_fno_bias,
            fft_norm=fft_norm,
            rank=rank,
            max_n_modes=max_n_modes,
            non_linearity=non_linearity,
        )

    def forward(self, x, output_shape=None, **kwargs):
        """
        FNO's forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        output_shape : {tuple, list of tuples, None}, optional
            Optionally specify the output shape for odd-shaped inputs.
            - If None, do not specify an output shape.
            - If tuple, specifies the output shape of the **last** FNO Block.
            - If list of tuples, specifies the exact output shape of each FNO Block.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the FNO model.
        """

        # The following code is commented out for potential future use:
        # It allows for flexible output shape specification for each FNO block.
        # if output_shape is None:
        #     output_shape = [None]*self.n_layers
        # elif isinstance(output_shape, tuple):
        #     output_shape = [None]*(self.n_layers - 1) + [output_shape]
        
        x = self.lifting(x)
        x = self.fno_blocks(x, output_shape=output_shape)
        x = self.projection(x)
        return x

    @property
    def n_modes(self):
        """Get the number of Fourier modes in each dimension."""
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        """Set the number of Fourier modes in each dimension."""
        self._n_modes = n_modes


class FNOInterpolate(nn.Module):
    """
    FNO with Interpolation-based encoding/decoding for 4D Space-Time grids.
    
    Maps between irregular point clouds (with 4D space-time coordinates) and
    regular (Nx, Ny, Nz, Nt) grids using nearest-neighbor / grid_sample
    interpolation. The architecture consists of:
    - 4D interpolation encoding: maps input point cloud to regular 4D grid
    - FNO processing: factorized space-time spectral convolution on regular grid
    - 4D interpolation decoding: maps from regular grid to output query points
    
    Input features per point-time pair: [obs, current_forcings, future_forcings]
    """

    def __init__(
        self,
        # Grid configuration
        latent_query_dims=(16, 16, 8, 10),
        coord_dim=3,
        
        # Input/Output channels
        in_channels=10,
        out_channels=2,
        
        # Latent features
        latent_feature_channels=None,
        
        # FNO configuration
        fno_n_layers=4,
        fno_n_modes=(8, 8, 6, 8),
        fno_hidden_channels=128,
        fno_skip_fno_bias=False,
        fno_fft_norm="forward",
        fno_rank=1.0,
        fno_max_n_modes=None,
        fno_non_linearity=F.gelu,
        
        # Lifting/Projection
        lifting_channels=128,
        projection_channel_ratio=4,
        
        # Interpolation settings
        align_corners=True,
        padding_mode='border',
        interpolation_mode='bilinear',
    ):
        """
        Parameters
        ----------
        latent_query_dims : tuple
            Size of the regular grid in latent space, e.g. (16, 16, 8, 10) for 3D+Time.
            Must have exactly coord_dim + 1 elements (spatial dims + temporal dim).
        coord_dim : int
            Number of spatial coordinate dimensions (2 or 3). Time is treated as
            an additional grid dimension but uses spatial coord_dim for grid_sample.
        in_channels : int
            Number of input feature channels per point-time pair
            (e.g., 2 obs + 4 cur_forcings + 4 fut_forcings = 10)
        out_channels : int
            Number of output feature channels (e.g., 2 for mass_concentration + head)
        latent_feature_channels : int, optional
            Number of additional latent feature channels to concatenate
        fno_n_layers : int
            Number of FNO layers
        fno_n_modes : tuple
            Number of Fourier modes per dimension (spatial + temporal)
        fno_hidden_channels : int
            Hidden channels in FNO
        lifting_channels : int
            Hidden channels in lifting MLP
        projection_channel_ratio : int
            Ratio to multiply fno_hidden_channels for projection MLP hidden channels
        align_corners : bool
            Whether to align corners in grid_sample
        padding_mode : str
            Padding mode for grid_sample ('zeros', 'border', 'reflection')
        interpolation_mode : str
            Interpolation mode for grid_sample ('bilinear' or 'nearest').
            Note: PyTorch internally performs trilinear interpolation when 5D tensors are passed with 'bilinear'.
        """
        super(FNOInterpolate, self).__init__()
        
        self.coord_dim = coord_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_feature_channels = latent_feature_channels
        self.fno_hidden_channels = fno_hidden_channels
        self.projection_channel_ratio = projection_channel_ratio
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        
        # Grid dimensions: spatial + temporal
        assert len(latent_query_dims) == coord_dim + 1, \
            f"latent_query_dims must have {coord_dim + 1} elements (spatial + time), got {len(latent_query_dims)}"
        self.latent_query_dims = latent_query_dims
        self.spatial_query_dims = latent_query_dims[:coord_dim]
        self.n_temporal_steps = latent_query_dims[coord_dim]
        
        # Set interpolation mode for grid_sample ('bilinear' or 'nearest')
        # PyTorch grid_sample expects 'bilinear' (which handles trilinear internally for 5D inputs) or 'nearest'
        self.interpolation_mode = 'nearest' if interpolation_mode == 'nearest' else 'bilinear'
        
        # Calculate FNO input channels
        self.fno_in_channels = in_channels
        if latent_feature_channels is not None:
            self.fno_in_channels += latent_feature_channels
        
        # Lifting layer: operates on 4D grid (B, C, X, Y, Z, T) via Conv1d fallback
        self.lifting = MLP(
            in_channels=self.fno_in_channels,
            out_channels=fno_hidden_channels,
            hidden_channels=lifting_channels,
            n_layers=2,
            n_dim=coord_dim + 1,  # 4D for space-time
        )
        
        # FNO blocks: factorized space-time spectral convolution
        self.fno_blocks = FNOBlocks(
            n_layers=fno_n_layers,
            n_modes=fno_n_modes,
            hidden_channels=fno_hidden_channels,
            skip_fno_bias=fno_skip_fno_bias,
            fft_norm=fno_fft_norm,
            rank=fno_rank,
            max_n_modes=fno_max_n_modes,
            non_linearity=fno_non_linearity,
        )
        
        # Projection layer: pointwise (1D) on sampled features
        projection_channels = projection_channel_ratio * fno_hidden_channels
        self.projection = MLP(
            in_channels=fno_hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=fno_non_linearity,
        )

    def _normalize_coords(self, coords, ref_coords):
        """
        Normalize spatial coordinates to [-1, 1] range for grid_sample.
        
        Parameters
        ----------
        coords : torch.Tensor
            Coordinates to normalize, shape (..., coord_dim)
        ref_coords : torch.Tensor
            Reference coordinates defining the bounding box, shape (N, coord_dim)
        
        Returns
        -------
        torch.Tensor
            Normalized coordinates in [-1, 1]
        """
        min_coords = ref_coords.min(dim=0, keepdim=True)[0]
        max_coords = ref_coords.max(dim=0, keepdim=True)[0]
        normalized = 2 * (coords - min_coords) / (max_coords - min_coords + 1e-8) - 1
        return normalized

    def _interpolate_to_grid(self, points_4d, features, spatial_ref_coords, latent_queries):
        """
        Interpolate point cloud features onto a regular 4D grid using
        per-time-slice nearest-neighbor spatial interpolation.
        
        Parameters
        ----------
        points_4d : torch.Tensor
            Point coordinates with time, shape (N_pts × T, 4) where last dim is (x,y,z,t_norm)
        features : torch.Tensor
            Point features, shape (batch, N_pts × T, in_channels)
        spatial_ref_coords : torch.Tensor
            Spatial reference coordinates for normalization, shape (N_pts, coord_dim)
        latent_queries : torch.Tensor
            Grid coordinates, shape (Nx, Ny, Nz, Nt, 4)
        
        Returns
        -------
        torch.Tensor
            Gridded features, shape (batch, in_channels, Nx, Ny, Nz, Nt)
        """
        batch_size = features.shape[0]
        n_total_points = points_4d.shape[0]
        Nx, Ny, Nz = self.spatial_query_dims
        Nt = self.n_temporal_steps
        
        # Extract spatial coordinates from points (first coord_dim dimensions)
        point_spatial = points_4d[:, :self.coord_dim]  # (N_pts × T, 3)
        point_time = points_4d[:, self.coord_dim]       # (N_pts × T,)
        
        # Extract temporal grid positions from latent_queries
        # latent_queries has shape (Nx, Ny, Nz, Nt, 4), temporal values at [:, :, :, :, 3]
        grid_time_values = latent_queries[0, 0, 0, :, self.coord_dim]  # (Nt,)
        
        # Extract spatial grid coordinates (same for all time slices)
        spatial_grid_coords = latent_queries[:, :, :, 0, :self.coord_dim]  # (Nx, Ny, Nz, 3)
        spatial_grid_flat = spatial_grid_coords.reshape(-1, self.coord_dim)  # (Nx*Ny*Nz, 3)
        n_spatial_grid = spatial_grid_flat.shape[0]
        
        # Points per time step: N_pts = n_total_points / Nt
        # (All time slices have same spatial points, tiled in the collate function)
        n_pts_per_t = n_total_points // Nt
        
        # Allocate output grid: (batch, C, Nx, Ny, Nz, Nt)
        C = features.shape[-1]
        grid_features = torch.zeros(
            batch_size, C, Nx * Ny * Nz, Nt,
            device=features.device, dtype=features.dtype
        )
        
        # For each time slice, find nearest spatial neighbors and assign features
        for t_idx in range(Nt):
            # Points for this time slice: indices [t_idx * n_pts_per_t : (t_idx+1) * n_pts_per_t]
            # But since points are laid out as (pt0_t0, pt1_t0, ..., ptN_t0, pt0_t1, ..., ptN_t1, ...)
            # we need: points[t_idx::Nt] if interleaved, or points[t_idx*n_pts_per_t:(t_idx+1)*n_pts_per_t] if blocked
            # The collate function uses blocked layout: reshape(n_pts * T_in, C_in) from (n_pts, T_in, C_in)
            # So points are: [pt0_t0, pt0_t1, ..., pt0_tT, pt1_t0, pt1_t1, ..., pt1_tT, ...]
            # → We need every Nt-th point starting at t_idx
            slice_mask = torch.arange(n_total_points, device=features.device)
            # Points are (N_pts, T, ...) → reshaped to (N_pts*T, ...) in C-order
            # So point i at time t is at index i * Nt + t (if reshape from (N_pts, T) to (N_pts*T))
            # Wait: collate reshapes (N_pts, T_in, C_in) → (N_pts*T_in, C_in)
            # So index = pt_idx * T + t_idx
            t_point_indices = torch.arange(t_idx, n_total_points, Nt, device=features.device)
            
            # Spatial coords for this time slice
            t_spatial = point_spatial[t_point_indices]  # (n_pts_per_t, 3)
            
            # Find nearest spatial neighbor for each grid point
            # distances: (n_spatial_grid, n_pts_per_t)
            distances = torch.cdist(spatial_grid_flat.unsqueeze(0), t_spatial.unsqueeze(0)).squeeze(0)
            nearest_idx = distances.argmin(dim=1)  # (n_spatial_grid,)
            
            # Gather features for this time slice from all batches
            t_features = features[:, t_point_indices, :]  # (batch, n_pts_per_t, C)
            grid_features[:, :, :, t_idx] = t_features[:, nearest_idx, :].permute(0, 2, 1)  # (batch, C, n_spatial_grid)
        
        # Reshape to (batch, C, Nx, Ny, Nz, Nt)
        grid_features = grid_features.reshape(batch_size, C, Nx, Ny, Nz, Nt)
        
        return grid_features

    def _interpolate_from_grid(self, grid_features, output_queries_4d, spatial_ref_coords):
        """
        Interpolate from regular 4D grid to output query points.
        Uses fold-T-into-batch approach: for each time slice, perform 3D grid_sample.
        
        Parameters
        ----------
        grid_features : torch.Tensor
            Features on regular grid, shape (batch, channels, Nx, Ny, Nz, Nt)
        output_queries_4d : torch.Tensor
            Output query coordinates, shape (N_pts × T_out, 4)
        spatial_ref_coords : torch.Tensor
            Spatial reference coordinates for normalization, shape (N_pts, coord_dim)
        
        Returns
        -------
        torch.Tensor
            Interpolated features at query points, shape (batch, N_pts × T_out, channels)
        """
        batch_size, channels, Nx, Ny, Nz, Nt = grid_features.shape
        n_total_queries = output_queries_4d.shape[0]
        n_pts_per_t = n_total_queries // Nt
        
        # Normalize spatial coordinates of queries to [-1, 1]
        query_spatial = output_queries_4d[:, :self.coord_dim]  # (N_pts × T_out, 3)
        normalized_spatial = self._normalize_coords(query_spatial, spatial_ref_coords)
        
        # For each time slice, do 3D grid_sample
        all_sampled = []
        for t_idx in range(Nt):
            # Extract grid slice for this time step: (batch, channels, Nx, Ny, Nz)
            grid_slice = grid_features[:, :, :, :, :, t_idx]
            
            # Get query points for this time step
            # Layout: (N_pts, T, ...) → (N_pts*T, ...) means point i at time t = index i*T + t
            t_query_indices = torch.arange(t_idx, n_total_queries, Nt, device=grid_features.device)
            t_normalized = normalized_spatial[t_query_indices]  # (n_pts_per_t, 3)
            
            # Prepare for grid_sample: (batch, n_pts_per_t, 1, 1, 3)
            sample_grid = t_normalized.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, n_pts, 1, 1, 3)
            sample_grid = sample_grid.expand(batch_size, -1, -1, -1, -1)
            
            # grid_sample: input (B, C, D, H, W), grid (B, D_out, H_out, W_out, 3)
            sampled = F.grid_sample(
                grid_slice,
                sample_grid,
                mode=self.interpolation_mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners
            )  # (batch, channels, n_pts_per_t, 1, 1)
            
            sampled = sampled.squeeze(-1).squeeze(-1)  # (batch, channels, n_pts_per_t)
            all_sampled.append(sampled)
        
        # Stack and interleave time slices: need to match the (N_pts*T, ...) layout
        # all_sampled[t] has shape (batch, channels, n_pts_per_t)
        # We need output (batch, N_pts*T_out, channels) with layout pt0_t0, pt0_t1, ..., pt0_tT, pt1_t0, ...
        stacked = torch.stack(all_sampled, dim=3)  # (batch, channels, n_pts_per_t, Nt)
        # Reshape to (batch, channels, n_pts_per_t * Nt) with interleaved time
        result = stacked.reshape(batch_size, channels, n_pts_per_t * Nt)
        result = result.permute(0, 2, 1)  # (batch, N_pts × T_out, channels)
        
        return result

    def forward(self, input_geom, latent_queries, output_queries, x=None, **kwargs):
        """
        Forward pass through FNOInterpolate.
        
        Parameters
        ----------
        input_geom : torch.Tensor
            Input 4D coordinates (space + time), shape (N_pts × T_in, 4) or (1, N_pts × T_in, 4)
        latent_queries : torch.Tensor
            Regular 4D grid coordinates, shape (Nx, Ny, Nz, Nt, 4) or (1, Nx, Ny, Nz, Nt, 4)
        output_queries : torch.Tensor
            Output 4D query coordinates, shape (N_pts × T_out, 4) or (1, N_pts × T_out, 4)
        x : torch.Tensor, optional
            Input features, shape (batch, N_pts × T_in, in_channels)
        
        Returns
        -------
        torch.Tensor
            Output features, shape (batch, N_pts × T_out, out_channels)
        """
        if x is None:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        
        # Squeeze batch dimensions if present
        if input_geom.ndim == 3:
            input_geom = input_geom.squeeze(0)
        if output_queries.ndim == 3:
            output_queries = output_queries.squeeze(0)
        if latent_queries.ndim == len(self.latent_query_dims) + 2:
            latent_queries = latent_queries.squeeze(0)
        
        # Extract spatial reference coordinates (first coord_dim dims of input points)
        # Use unique spatial coords for normalization reference
        spatial_ref = input_geom[:, :self.coord_dim]
        
        # 1. Interpolate input features to 4D grid
        grid_features = self._interpolate_to_grid(
            input_geom, x, spatial_ref, latent_queries
        )  # (batch, C_in, Nx, Ny, Nz, Nt)
        
        # 2. Lifting: (batch, C_in, Nx, Ny, Nz, Nt) → (batch, hidden, Nx, Ny, Nz, Nt)
        grid_features = self.lifting(grid_features)
        
        # 3. FNO blocks: factorized space-time spectral conv (already handles 4D)
        grid_features = self.fno_blocks(grid_features)
        
        # 4. Interpolate from grid to output query points
        output_features = self._interpolate_from_grid(
            grid_features, output_queries, spatial_ref
        )  # (batch, N_pts × T_out, hidden)
        
        # 5. Projection: (batch, N_pts × T_out, hidden) → (batch, N_pts × T_out, out_channels)
        output_features = output_features.permute(0, 2, 1)  # (batch, hidden, N_pts × T_out)
        output_features = self.projection(output_features)    # (batch, out_channels, N_pts × T_out)
        output_features = output_features.permute(0, 2, 1)   # (batch, N_pts × T_out, out_channels)
        
        return output_features