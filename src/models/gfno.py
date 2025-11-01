"""
GFNO (Graph Neural Operator with Fourier Neural Operator) Model

This module implements a hybrid architecture combining:
- GNO (Graph Neural Operator) for encoding boundary conditions on irregular geometries
- FNO (Fourier Neural Operator) for processing in latent space on regular grids

The model is designed for spatio-temporal problems where boundary conditions are specified
on irregular point clouds, but the solution needs to be computed on a regular grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neuralop.gno import GNOBlock
from .neuralop.fno import FNOBlocks
from .neuralop.channel_mlp import ChannelMLP


class GFNO(nn.Module):
    """
    Graph-Fourier Neural Operator (GFNO) for learning operators on irregular domains.
    
    Architecture:
    1. Input GNO: Encodes boundary conditions from irregular points to latent grid
    2. Latent FNO: Processes features in latent space using Fourier transforms
    3. Projection: Maps latent features to output channels
    
    The model can optionally incorporate additional latent features at the query points.
    """

    def __init__(
        self,
        # Input GNO parameters
        gno_coord_dim=3,
        gno_radius=0.03,
        gno_pos_embed_type='transformer',
        gno_pos_embed_channels=32,
        gno_pos_embed_max_positions=10000,
        gno_channel_mlp_layers=[64, 64, 64],
        gno_channel_mlp_non_linearity=F.gelu,
        gno_out_channels=3,
        # Latent features
        latent_feature_channels=None,
        # FNO parameters
        fno_n_layers=4,
        fno_n_modes=(16, 16, 16),
        fno_hidden_channels=128,
        fno_skip_fno_bias=False,
        fno_fft_norm="forward",
        fno_rank=1.0,
        fno_max_n_modes=None,
        fno_non_linearity=F.gelu,
        # Lifting parameters
        lifting_channels=128,
        # Projection parameters
        projection_channel_ratio=4,
        out_channels=1,
        # Neighbor search settings
        use_open3d_neighbor_search=None,
    ):
        """
        Initialize the GFNO model.
        
        Args:
            gno_coord_dim (int): Coordinate dimension for GNO (e.g., 2 for 2D, 3 for 3D)
            gno_radius (float): Radius for neighbor search in GNO
            gno_pos_embed_type (str): Type of positional embedding ('transformer', 'nerf', etc.)
            gno_pos_embed_channels (int): Number of channels for positional embedding
            gno_pos_embed_max_positions (int): Maximum positions for transformer embeddings
            gno_channel_mlp_layers (list): Hidden layer sizes for GNO channel MLP
            gno_channel_mlp_non_linearity (callable): Activation function for GNO MLP
            gno_out_channels (int): Output channels from GNO
            latent_feature_channels (int, optional): Number of additional latent feature channels
            fno_n_layers (int): Number of FNO layers
            fno_n_modes (tuple): Number of Fourier modes per dimension
            fno_hidden_channels (int): Hidden channels in FNO
            fno_skip_fno_bias (bool): Whether to skip bias in FNO layers
            fno_fft_norm (str): FFT normalization mode ('forward', 'backward', 'ortho')
            fno_rank (float): Rank for low-rank factorization (1.0 = full rank)
            fno_max_n_modes (int, optional): Maximum number of modes
            fno_non_linearity (callable): Activation function for FNO
            lifting_channels (int): Channels in lifting layer
            projection_channel_ratio (int): Channel ratio for projection layer
            out_channels (int): Number of output channels
            use_open3d_neighbor_search (bool, optional): Use Open3D for neighbor search (auto-detect if None)
        """
        super(GFNO, self).__init__()

        # Determine whether to use open3d neighbor search based on coordinate dimension
        if use_open3d_neighbor_search is None:
            # Only use open3d for 3D coordinates
            use_open3d_neighbor_search = (gno_coord_dim == 3)

        # Store configuration
        self.gno_coord_dim = gno_coord_dim
        self.gno_radius = gno_radius
        self.gno_pos_embed_type = gno_pos_embed_type
        self.gno_pos_embed_channels = gno_pos_embed_channels
        self.gno_pos_embed_max_positions = gno_pos_embed_max_positions
        self.gno_channel_mlp_layers = gno_channel_mlp_layers
        self.gno_channel_mlp_non_linearity = gno_channel_mlp_non_linearity
        self.gno_out_channels = gno_out_channels
        self.latent_feature_channels = latent_feature_channels

        # Initialize GNO block for encoding input boundary conditions
        self.gno = GNOBlock(
            in_channels=0,  # Will be inferred from input
            out_channels=gno_out_channels,
            coord_dim=gno_coord_dim,
            radius=gno_radius,
            pos_embedding_type=gno_pos_embed_type,
            pos_embedding_channels=gno_pos_embed_channels,
            pos_embedding_max_positions=gno_pos_embed_max_positions,
            reduction='mean',
            weighting_fn=None,
            channel_mlp_layers=gno_channel_mlp_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type='linear',
            use_open3d_neighbor_search=use_open3d_neighbor_search,
            use_torch_scatter_reduce=False
        )
        
        # Store additional attributes for forward pass
        self.fno_hidden_channels = fno_hidden_channels
        self.in_coord_dim_reverse_order = list(range(2, gno_coord_dim + 2))  # For permute operation
        self.adain_pos_embed = None  # Placeholder for adaptive instance norm embedding
        self.fno_norm = None  # Placeholder for FNO normalization
        self.out_gno_tanh = None  # Placeholder for output GNO tanh activation

        # Determine FNO input channels (GNO output + optional latent features)
        if latent_feature_channels is not None:
            self.fno_in_channels = gno_out_channels + latent_feature_channels
        else:
            self.fno_in_channels = gno_out_channels

        # Initialize FNO blocks for latent space processing
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

        # Initialize lifting layer (maps input features to FNO hidden dimension)
        self.lifting_channels = lifting_channels
        self.lifting = ChannelMLP(
            in_channels=self.fno_in_channels,
            hidden_channels=self.lifting_channels,
            out_channels=fno_hidden_channels,
            n_layers=2
        )

        # Initialize projection layer (maps FNO output to desired output channels)
        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = projection_channel_ratio * fno_hidden_channels
        self.out_channels = out_channels
        self.projection = ChannelMLP(
            in_channels=fno_hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            non_linearity=fno_non_linearity
        )

    def latent_embedding(self, in_p, ada_in=None):
        """
        Process features in latent space using FNO.
        
        Args:
            in_p (torch.Tensor): Input features with shape (batch, n_1, ..., n_k, channels)
            ada_in (torch.Tensor, optional): Adaptive instance normalization parameters
            
        Returns:
            torch.Tensor: Latent embeddings with shape (batch, channels, n_1, ..., n_k)
        """
        # Permute from (batch, n_1, ..., n_k, channels) to (batch, channels, n_1, ..., n_k)
        in_p = in_p.permute(0, len(in_p.shape)-1, *list(range(1, len(in_p.shape)-1)))
        
        # Update adaptive instance normalization embedding if provided
        if ada_in is not None:
            if ada_in.ndim == 2:
                ada_in = ada_in.squeeze(0)
            if self.adain_pos_embed is not None:
                ada_in_embed = self.adain_pos_embed(ada_in.unsqueeze(0)).squeeze(0)
            else:
                ada_in_embed = ada_in
            if self.fno_norm == "ada_in":
                self.fno_blocks.set_ada_in_embeddings(ada_in_embed)

        # Apply lifting and FNO blocks
        in_p = self.lifting(in_p)
        in_p = self.fno_blocks(in_p)

        return in_p

    def forward(self, input_geom, latent_queries, x=None, latent_features=None, ada_in=None, **kwargs):
        """
        Forward pass of the GFNO model.
        
        Args:
            input_geom (torch.Tensor): Input geometry (boundary condition coordinates)
                Shape: (batch, n_input_points, coord_dim) or (n_input_points, coord_dim)
            latent_queries (torch.Tensor): Query points for latent space
                Shape: (batch, n_1, ..., n_k, coord_dim) or (n_1, ..., n_k, coord_dim)
            x (torch.Tensor, optional): Input features at boundary conditions
                Shape: (batch, n_input_points, in_channels)
            latent_features (torch.Tensor, optional): Additional features at query points
                Shape: (batch, n_1, ..., n_k, latent_feature_channels)
            ada_in (torch.Tensor, optional): Adaptive instance norm parameters
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Output predictions
                Shape: (batch, n_1, ..., n_k, out_channels)
        """
        # Determine batch size
        if x is None:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        
        # Validate latent features if provided
        if latent_features is not None:
            assert self.latent_feature_channels is not None, \
                "if passing latent features, latent_feature_channels must be set."
            assert latent_features.shape[-1] == self.latent_feature_channels, \
                f"Expected {self.latent_feature_channels} latent feature channels, got {latent_features.shape[-1]}"

            # Validate dimensionality
            assert latent_features.ndim == self.gno_coord_dim + 2, \
                f"Latent features must be of shape (batch, n_gridpts_1, ...n_gridpts_n, channels), got {latent_features.shape}"
            
            # Broadcast if needed
            if latent_features.shape[0] != batch_size:
                if latent_features.shape[0] == 1:
                    latent_features = latent_features.repeat(batch_size, *[1]*(latent_features.ndim-1))

        # Process GNO encoding
        if (input_geom.shape[0] == 1 and input_geom.ndim == 3) or input_geom.ndim == 2:
            # Single sample or unbatched input
            input_geom = input_geom.squeeze(0) 
            latent_queries = latent_queries.squeeze(0)
            
            # Reshape to grid
            grid_shape = latent_queries.shape[1:-1]

            # Pass through input GNOBlock 
            in_p = self.gno(
                y=input_geom,
                x=latent_queries.view((-1, latent_queries.shape[-1])),
                f_y=x
            )
        elif input_geom.shape[0] == batch_size:
            
            # Reshape to grid
            grid_shape = latent_queries.shape[1:-1]
            
            # Batched input - process each sample separately
            in_p_list = []
            for b in range(batch_size):
                in_p_b = self.gno(
                    y=input_geom[b],
                    x=latent_queries[b].view((-1, latent_queries.shape[-1])),
                    f_y=x[b] if x is not None else None
                )
                in_p_list.append(in_p_b)
            in_p = torch.stack(in_p_list, dim=0)
        else:
            raise ValueError(f"input_geom batch size {input_geom.shape[0]} does not match x batch size {batch_size}")
        
        # Disregard positional encoding dim
        in_p = in_p.view((batch_size, *grid_shape, -1))
        
        # Concatenate with latent features if provided
        if latent_features is not None:
            in_p = torch.cat((in_p, latent_features), dim=-1)
        
        # Apply FNO in latent space
        latent_embed = self.latent_embedding(in_p=in_p, ada_in=ada_in)

        # Process latent embeddings
        # Shape: (batch, channels, n_1, n_2, ..., n_k)
        batch_size = latent_embed.shape[0]
        
        # Permute to (batch, n_1, n_2, ...n_k, channels)
        latent_embed = latent_embed.permute(0, *self.in_coord_dim_reverse_order, 1)
        
        # Optional tanh activation
        if self.out_gno_tanh in ['latent_embed', 'both']:
            latent_embed = torch.tanh(latent_embed)

        # Project to output channels
        # Permute back to (batch, channels, n_1, n_2, n_3)
        latent_embed = latent_embed.permute(0, 4, 1, 2, 3)
        out = self.projection(latent_embed)
        # Permute to (batch, n_1, n_2, n_3, out_channels)
        out = out.permute(0, *self.in_coord_dim_reverse_order, 1)
        
        return out
