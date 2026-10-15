"""
The vectorised segment_csr must give the same forward values and gradients as the original
per-neighborhood Python loop, for sum and mean, batched and unbatched, with empty neighborhoods,
and inside a full GINO forward/backward pass.
"""
import pytest
import torch
from torch import einsum

import src.models.neuralop.integral_transform as integral_transform
from src.models.neuralop.gino import GINO
from src.models.neuralop.segment_csr import segment_csr

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def loop_segment_csr(src, indptr, reduction, use_scatter=False):
    """The original native implementation, kept as the reference."""
    if src.ndim == 3:
        batched, point_dim = True, 1
    else:
        batched, point_dim = False, 0
    output_shape = list(src.shape)
    n_out = indptr.shape[point_dim] - 1
    output_shape[point_dim] = n_out
    out = torch.zeros(output_shape, device=src.device)
    for i in range(n_out):
        if batched:
            from_idx = (slice(None), slice(indptr[0, i], indptr[0, i + 1]))
            ein_str = 'bio->bo'
            start = indptr[0, i]
            n_nbrs = indptr[0, i + 1] - start
            to_idx = (slice(None), i)
        else:
            from_idx = slice(indptr[i], indptr[i + 1])
            ein_str = 'io->o'
            start = indptr[i]
            n_nbrs = indptr[i + 1] - start
            to_idx = i
        src_from = src[from_idx]
        if n_nbrs > 0:
            to_reduce = einsum(ein_str, src_from)
            if reduction == "mean":
                to_reduce /= n_nbrs
            out[to_idx] += to_reduce
    return out


def random_csr(n_out, max_nbrs, device, generator):
    counts = torch.randint(0, max_nbrs + 1, (n_out,), generator=generator)
    counts[::7] = 0                                          # guarantee empty neighborhoods
    return torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]).to(device)


def forward_backward(fn, src, indptr, reduction, weight):
    src = src.clone().requires_grad_(True)
    out = fn(src, indptr, reduction=reduction, use_scatter=False)
    (out * weight).sum().backward()
    return out.detach(), src.grad


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('reduction', ['sum', 'mean'])
def test_matches_loop(device, batched, reduction):
    g = torch.Generator().manual_seed(0)
    indptr = random_csr(257, 12, device, g)
    n_edges = int(indptr[-1])
    src = torch.randn((3, n_edges, 16) if batched else (n_edges, 16), generator=g).to(device)
    if batched:
        indptr = indptr.unsqueeze(0).repeat(3, 1)              # as IntegralTransform passes it
    out_shape = (3, 257, 16) if batched else (257, 16)
    weight = torch.randn(out_shape, generator=g).to(device)   # non-uniform upstream gradient

    ref_out, ref_grad = forward_backward(loop_segment_csr, src, indptr, reduction, weight)
    new_out, new_grad = forward_backward(segment_csr, src, indptr, reduction, weight)

    assert new_out.shape == ref_out.shape and new_out.dtype == ref_out.dtype
    torch.testing.assert_close(new_out, ref_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(new_grad, ref_grad, rtol=1e-5, atol=1e-6)
    empty = (indptr[0] if batched else indptr).diff() == 0
    assert empty.any() and (new_out[:, empty] if batched else new_out[empty]).abs().max() == 0


@pytest.mark.parametrize('device', DEVICES)
def test_gino_forward_backward_matches_loop(device, monkeypatch):
    torch.manual_seed(0)
    model = GINO(in_gno_radius=0.24, out_gno_radius=0.24, in_gno_out_channels=30,
                 in_gno_channel_mlp_layers=[32, 64, 32], fno_n_layers=2, fno_n_modes=(4, 4, 4),
                 fno_hidden_channels=32, lifting_channels=16, out_gno_channel_mlp_layers=[32, 64, 32],
                 projection_channel_ratio=2, out_channels=10, use_open3d_neighbor_search=False).to(device)
    points = torch.rand(600, 3, device=device) * torch.tensor([1.5, 1.5, 2.5], device=device)
    grid = torch.stack(torch.meshgrid(*[torch.linspace(0, e, n, device=device)
                                        for e, n in zip((1.5, 1.5, 2.5), (8, 8, 8))], indexing='ij'), dim=-1)
    x = torch.randn(4, 600, 30, device=device)
    y = torch.randn(4, 600, 10, device=device)

    def run():
        model.zero_grad(set_to_none=True)
        out = model(input_geom=points, latent_queries=grid, x=x, output_queries=points)
        ((out - y) ** 2).mean().backward()
        return out.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    monkeypatch.setattr(integral_transform, 'segment_csr', loop_segment_csr)
    ref_out, ref_grads = run()
    monkeypatch.setattr(integral_transform, 'segment_csr', segment_csr)
    new_out, new_grads = run()

    torch.testing.assert_close(new_out, ref_out, rtol=1e-5, atol=1e-6)
    assert ref_grads.keys() == new_grads.keys()
    for name in ref_grads:
        torch.testing.assert_close(new_grads[name], ref_grads[name], rtol=1e-4, atol=1e-6, msg=name)
