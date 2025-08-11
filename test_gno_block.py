import torch
import numpy as np
from src.models.neuralop.gno import GNOBlock

def test_gno_block_forward():
    # Create a regular grid y of shape (40, 40, 40, 3)
    grid_size = 40
    x_grid = np.stack(np.meshgrid(
        np.linspace(0, 1, grid_size),
        np.linspace(0, 1, grid_size),
        np.linspace(0, 1, grid_size),
        indexing='ij'
    ), axis=-1)  # shape: (40, 40, 40, 3)
    x_grid = torch.tensor(x_grid, dtype=torch.float32)


    # Create 100 random query points x in [0, 1]^3
    y = torch.rand(100, 3, dtype=torch.float32)

    # print shape of x_grid and y
    print(x_grid.shape, y.shape)

    # Instantiate GNOBlock
    gno = GNOBlock(
        in_channels=0,
        out_channels=3,
        coord_dim=3,
        radius=0.03,
        pos_embedding_type='transformer',
        pos_embedding_channels=32,
        pos_embedding_max_positions=10000,
        reduction='mean',
        weighting_fn=None,
        channel_mlp_layers=[64, 64, 64],
        channel_mlp_non_linearity=torch.nn.GELU(),
        transform_type='linear',
        use_open3d_neighbor_search=True,
        use_torch_scatter_reduce=False
    )

    # print the data type of gnoblock torch parameters
    for name, param in gno.named_parameters():
        print(f"{name}: {param.dtype}")
    print(f"x_grid: {x_grid.dtype}")
    print(f"y: {y.dtype}")

    # Forward pass
    out = gno(y=y, x=x_grid.view((-1, 3)))
    print('Output shape:', out.shape)
    assert out.shape[0] == x_grid.view((-1, 3)).shape[0], f"Expected output shape[0]={x_grid.view((-1, 3)).shape[0]}, got {out.shape[0]}"
    
if __name__ == "__main__":
    test_gno_block_forward() 