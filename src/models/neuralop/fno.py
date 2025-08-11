from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from .conv import SpectralConv
from .mlp import MLP


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
        self.fno_skips = nn.ModuleList(
            [
                getattr(nn, f"Conv{self.n_dim}d")(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    bias=skip_fno_bias,
                )
                for _ in range(n_layers)
            ]
        )

        # Create spectral convolution layers
        self.convs = nn.ModuleList([
            SpectralConv(
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