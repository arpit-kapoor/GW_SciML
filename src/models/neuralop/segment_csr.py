from typing import Literal
import importlib

import torch

def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    reduction: Literal["mean", "sum"],
    use_scatter=True,
):
    """segment_csr reduces all entries of a CSR-formatted
    matrix by summing or averaging over neighbors.

    Used to reduce features over neighborhoods
    in neuralop.layers.IntegralTransform

    If use_scatter is set to False or torch_scatter is not
    properly built, segment_csr falls back to a vectorised PyTorch
    implementation (a single index_add over neighborhood ids). It matches the
    original per-neighborhood loop up to floating-point summation order; on
    CUDA it is only bitwise reproducible with
    ``torch.use_deterministic_algorithms(True)``.

    Parameters
    ----------
    src : torch.Tensor
        tensor of features for each point
    indptr : torch.Tensor
        splits representing start and end indices
        of each neighborhood in src
    reduce : Literal['mean', 'sum'], optional
        how to reduce a neighborhood. if mean,
        reduce by taking the average of all neighbors.
        Otherwise take the sum.
    use_scatter : bool, optional
        whether to use ``torch-scatter.segment_csr``. If False, uses the native PyTorch reduction.
        By default True

        .. warning:: 

            ``torch-scatter`` is an optional dependency that conflicts with the newest versions of PyTorch,
            so you must handle the conflict explicitly in your environment. See :ref:`torch_scatter_dependency` 
            for more information. 
    """
    if reduction not in ["mean", "sum"]:
        raise ValueError("reduce must be one of 'mean', 'sum'")

    if (
        importlib.util.find_spec("torch_scatter") is not None
        and use_scatter
    ):
        """only import torch_scatter when cuda is available"""
        import torch_scatter.segment_csr as scatter_segment_csr

        return scatter_segment_csr(src, indptr, reduce=reduction)

    else:
        if use_scatter:
            print("Warning: use_scatter is True but torch_scatter is not properly built. \
                  Defaulting to naive PyTorch implementation")
        # if batched, shape [b, n_reps, channels]
        # otherwise shape [n_reps, channels]
        batched = src.ndim == 3
        point_dim = 1 if batched else 0
        # neighborhoods are the same for every batch element, so the first row of indptr describes them all
        splits = indptr[0] if batched else indptr
        n_nbrs = splits[1:] - splits[:-1]
        n_out = n_nbrs.shape[0]

        # Vectorised reduction: tag every entry of src with the neighborhood it belongs to and
        # accumulate them in one index_add_, instead of looping over neighborhoods in Python
        segment_ids = torch.repeat_interleave(torch.arange(n_out, device=src.device), n_nbrs)

        # if batched, shape [b, n_out, channels]
        # otherwise shape [n_out, channels]
        output_shape = list(src.shape)
        output_shape[point_dim] = n_out
        out = torch.zeros(output_shape, device=src.device, dtype=src.dtype)
        out = out.index_add(point_dim, segment_ids, src)

        if reduction == "mean":
            # empty neighborhoods stay zero
            counts = n_nbrs.clamp(min=1).to(src.dtype)
            out = out / (counts.view(1, -1, 1) if batched else counts.view(-1, 1))
        return out
