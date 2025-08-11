from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gno import GNOBlock
from .fno import FNOBlocks
from .channel_mlp import ChannelMLP


class GINO(nn.Module):
    """Geometry-Informed Neural Operator (GINO) model.
    """

    def __init__(
        self,
        # Input GNO
        in_gno_coord_dim=3,
        in_gno_radius=0.03,
        gno_pos_embed_type='transformer',
        in_gno_pos_embed_channels=32,
        in_gno_pos_embed_max_positions=10000,
        in_gno_channel_mlp_layers=[64, 64, 64],
        in_gno_channel_mlp_non_linearity=F.gelu,
        in_gno_out_channels=3,
        # Latent features
        latent_feature_channels=None,
        # FNO
        fno_n_layers=4,
        fno_n_modes=(16, 16, 16),
        fno_hidden_channels=128,
        fno_skip_fno_bias=False,
        fno_fft_norm="forward",
        fno_rank=1.0,
        fno_max_n_modes=None,
        fno_non_linearity=F.gelu,
        # Lifting
        lifting_channels=128,
        # Output GNO
        out_gno_coord_dim=3,
        out_gno_radius=0.03,
        out_gno_pos_embed_type='transformer',
        out_gno_pos_embed_channels=32,
        out_gno_pos_embed_max_positions=10000,
        out_gno_channel_mlp_layers=[64, 64, 64],
        out_gno_channel_mlp_non_linearity=F.gelu,
        # Projection
        projection_channel_ratio=4,
        out_channels=1,
        # Neighbor search settings
        use_open3d_neighbor_search=None,
    ):
        super(GINO, self).__init__()

        # Determine whether to use open3d neighbor search based on coordinate dimension
        if use_open3d_neighbor_search is None:
            # Only use open3d for 3D coordinates
            use_open3d_neighbor_search = (in_gno_coord_dim == 3 and out_gno_coord_dim == 3)

        self.in_gno_coord_dim = in_gno_coord_dim
        self.in_gno_radius = in_gno_radius
        self.gno_pos_embed_type = gno_pos_embed_type
        self.in_gno_pos_embed_channels = in_gno_pos_embed_channels
        self.in_gno_pos_embed_max_positions = in_gno_pos_embed_max_positions
        self.in_gno_channel_mlp_layers = in_gno_channel_mlp_layers
        self.in_gno_channel_mlp_non_linearity = in_gno_channel_mlp_non_linearity
        self.in_gno_out_channels = in_gno_out_channels
        self.latent_feature_channels = latent_feature_channels

        self.in_gno = GNOBlock(
            in_channels=0,
            out_channels=in_gno_out_channels,
            coord_dim=in_gno_coord_dim,
            radius=in_gno_radius,
            pos_embedding_type=gno_pos_embed_type,
            pos_embedding_channels=in_gno_pos_embed_channels,
            pos_embedding_max_positions=in_gno_pos_embed_max_positions,
            reduction='mean',
            weighting_fn=None,
            channel_mlp_layers=in_gno_channel_mlp_layers,
            channel_mlp_non_linearity=in_gno_channel_mlp_non_linearity,
            transform_type='linear',
            use_open3d_neighbor_search=use_open3d_neighbor_search,
            use_torch_scatter_reduce=False
        )
        
        # Store additional attributes needed for forward pass
        self.fno_hidden_channels = fno_hidden_channels
        self.in_coord_dim_reverse_order = list(range(2, in_gno_coord_dim + 2))  # For permute operation
        self.adain_pos_embed = None  # Placeholder for adaptive instance norm embedding
        self.fno_norm = None  # Placeholder for FNO normalization
        self.out_gno_tanh = None  # Placeholder for output GNO tanh activation

        if latent_feature_channels is not None:
            self.fno_in_channels = in_gno_out_channels + latent_feature_channels
        else:
            self.fno_in_channels = in_gno_out_channels

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

        self.lifting_channels = lifting_channels
        self.lifting = ChannelMLP(
            in_channels=self.fno_in_channels,
            hidden_channels=self.lifting_channels,
            out_channels=fno_hidden_channels,
            n_layers=2
        )

        self.out_gno_coord_dim = out_gno_coord_dim
        self.out_gno_radius = out_gno_radius
        self.out_gno_pos_embed_type = out_gno_pos_embed_type
        self.out_gno_pos_embed_channels = out_gno_pos_embed_channels
        self.out_gno_pos_embed_max_positions = out_gno_pos_embed_max_positions
        self.out_gno_channel_mlp_layers = out_gno_channel_mlp_layers
        self.out_gno_channel_mlp_non_linearity = out_gno_channel_mlp_non_linearity

        self.out_gno = GNOBlock(
            in_channels=fno_hidden_channels,
            out_channels=fno_hidden_channels,
            coord_dim=out_gno_coord_dim,
            radius=out_gno_radius,
            pos_embedding_type=out_gno_pos_embed_type,
            pos_embedding_channels=out_gno_pos_embed_channels,
            pos_embedding_max_positions=out_gno_pos_embed_max_positions,
            reduction='sum',
            weighting_fn=None,
            channel_mlp_layers=out_gno_channel_mlp_layers,
            channel_mlp_non_linearity=out_gno_channel_mlp_non_linearity,
            transform_type='linear',
            use_open3d_neighbor_search=use_open3d_neighbor_search,
            use_torch_scatter_reduce=False
        )
        
        # Store reference to out_gno for forward pass
        self.gno_out = self.out_gno

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = projection_channel_ratio * fno_hidden_channels
        self.out_channels = out_channels
        self.projection = ChannelMLP(
            in_channels=fno_hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=fno_non_linearity
        )

    def latent_embedding(self, in_p, ada_in=None):

        # in_p : (batch, n_1 , ... , n_k, in_channels + k)
        # ada_in : (fno_ada_in_dim, )

        # permute (b, n_1, ..., n_k, c) -> (b,c, n_1,...n_k)
        in_p = in_p.permute(0, len(in_p.shape)-1, *list(range(1,len(in_p.shape)-1)))
        #Update Ada IN embedding    
        if ada_in is not None:
            if ada_in.ndim == 2:
                ada_in = ada_in.squeeze(0)
            if self.adain_pos_embed is not None:
                ada_in_embed = self.adain_pos_embed(ada_in.unsqueeze(0)).squeeze(0)
            else:
                ada_in_embed = ada_in
            if self.fno_norm == "ada_in":
                self.fno_blocks.set_ada_in_embeddings(ada_in_embed)

        #Apply FNO blocks
        in_p = self.lifting(in_p)

        # for idx in range(self.fno_blocks.n_layers):
        in_p = self.fno_blocks(in_p)

        return in_p 

    def forward(self, input_geom, latent_queries, output_queries, x=None, latent_features=None, ada_in=None, **kwargs):

        if x is None:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        
        if latent_features is not None:
            assert self.latent_feature_channels is not None,\
                  "if passing latent features, latent_feature_channels must be set."
            assert latent_features.shape[-1] == self.latent_feature_channels

            # batch, n_gridpts_1, .... n_gridpts_n, gno_coord_dim
            assert latent_features.ndim == self.in_gno_coord_dim + 2,\
                f"Latent features must be of shape (batch, n_gridpts_1, ...n_gridpts_n, gno_coord_dim), got {latent_features.shape}"
            # latent features must have the same shape (except channels) as latent_queries 
            if latent_features.shape[0] != batch_size:
                if latent_features.shape[0] == 1:
                    latent_features = latent_features.repeat(batch_size, *[1]*(latent_features.ndim-1))

        input_geom = input_geom.squeeze(0) 
        latent_queries = latent_queries.squeeze(0)

        # Pass through input GNOBlock 
        in_p = self.in_gno(y=input_geom,
                           x=latent_queries.view((-1, latent_queries.shape[-1])),
                           f_y=x)
        
        grid_shape = latent_queries.shape[:-1] # disregard positional encoding dim
        
        # shape (batch_size, grid1, ...gridn, -1)
        in_p = in_p.view((batch_size, *grid_shape, -1))
        
        if latent_features is not None:
            in_p = torch.cat((in_p, latent_features), dim=-1)
        # take apply fno in latent space
        latent_embed = self.latent_embedding(in_p=in_p, 
                                             ada_in=ada_in)

        # Integrate latent space to output queries
        #latent_embed shape (b, c, n_1, n_2, ..., n_k)
        batch_size = latent_embed.shape[0]
        # permute to (b, n_1, n_2, ...n_k, c)
        # then reshape to (b, n_1 * n_2 * ...n_k, out_channels)
        latent_embed = latent_embed.permute(0, *self.in_coord_dim_reverse_order, 1).reshape(batch_size, -1, self.fno_hidden_channels)
        
        if self.out_gno_tanh in ['latent_embed', 'both']:
            latent_embed = torch.tanh(latent_embed)
        

        # integrate over the latent space
        # if output queries is a dict, query the output gno separately 
        # with each tensor of query points
        if isinstance(output_queries, dict):
            out = {}
            for key, out_p in output_queries.items():
                out_p = out_p.squeeze(0)

                sub_output = self.gno_out(y=latent_queries.reshape((-1, latent_queries.shape[-1])), 
                    x=out_p,
                    f_y=latent_embed,)
                sub_output = sub_output.permute(0, 2, 1)

                # Project pointwise to out channels
                #(b, n_in, out_channels)
                sub_output = self.projection(sub_output).permute(0, 2, 1)  

                out[key] = sub_output
        else:
            output_queries = output_queries.squeeze(0)

            # latent queries is of shape (d_1 x d_2 x... d_n x n), reshape to n_out x n
            out = self.gno_out(y=latent_queries.reshape((-1, latent_queries.shape[-1])), 
                        x=output_queries,
                        f_y=latent_embed,)
            out = out.permute(0, 2, 1)

            # Project pointwise to out channels
            #(b, n_in, out_channels)
            out = self.projection(out).permute(0, 2, 1)  
        
        return out


if __name__ == "__main__":
    import torch

    # Dummy input data for demonstration
    # These shapes should match the expected input shapes in the docstring
    input_geom = torch.rand(1, 10, 2)  # e.g., 2D coordinates, 10 input points
    latent_queries = torch.rand(1, 16, 16, 2)  # e.g., 16x16 grid in 2D
    output_queries = torch.rand(1, 20, 2)  # 20 output query points
    x = torch.rand(1, 10, 3)  # batch=1, 10 input points, 3 input channels
    latent_features = torch.rand(1, 16, 16, 8)  # batch=1, 16x16 grid, 8 latent channels

    # Instantiate your GINO model here
    model = GINO(
        in_gno_coord_dim=2,
        in_gno_out_channels=3,
        out_channels=1,
        lifting_channels=128,
        latent_feature_channels=8,
    )
    print("input_geom shape:", input_geom.shape)
    print("latent_queries shape:", latent_queries.shape)
    print("output_queries shape:", output_queries.shape)
    print("x shape:", x.shape)
    print("latent_features shape:", latent_features.shape)

    output = model(input_geom, latent_queries, output_queries, x=x, latent_features=latent_features)
    print("Output:", output)
