"""Per-sample length (padding) support.

Ground truth throughout: a Python loop of per-sample SLICED calls
(batch=1 each, no padding anywhere), which exercises none of the length
machinery. The batched call with lens_x/lens_y must match it, and padding
frames must receive exactly-zero gradients.
"""
import math

import pytest
import torch

from softdtw_cuda import SoftDTW

GAMMA = 1.0


def _ref_sliced(x, y, lens_x, lens_y, gamma, normalize):
    """Per-sample sliced reference. Divergence computed manually from three
    non-normalized calls so lens_x[b] != lens_y[b] is allowed."""
    sdtw = SoftDTW(gamma=gamma, normalize=False, fused=False)
    outs = []
    for b in range(x.shape[0]):
        xb = x[b : b + 1, : int(lens_x[b])]
        yb = y[b : b + 1, : int(lens_y[b])]
        o = sdtw(xb, yb)
        if normalize:
            o = o - 0.5 * (sdtw(xb, xb) + sdtw(yb, yb))
        outs.append(o)
    return torch.cat(outs, dim=0)


def _make_inputs(device, dtype=torch.float64, B=5, N=48, D=7, requires_grad=False, equal_lens=False):
    g = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(B, N, D, generator=g, dtype=dtype).to(device)
    y = torch.randn(B, N, D, generator=g, dtype=dtype).to(device)
    # padding filled with large garbage so accidental participation is loud
    lens_x = torch.tensor([N, 1, 17, 33, N // 2], dtype=torch.int64)[:B]
    lens_y = lens_x.clone() if equal_lens else torch.tensor([N, 5, 9, N, N // 3], dtype=torch.int64)[:B]
    for b in range(B):
        x[b, lens_x[b] :] = 1000.0 + b
        y[b, lens_y[b] :] = -2000.0 - b
    x.requires_grad_(requires_grad)
    return x, y, lens_x.to(device), lens_y.to(device)


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_forward_matches_sliced(device, fused, normalize):
    if fused and device == "cpu":
        pytest.skip("fused requires CUDA")
    equal_lens = normalize  # normalize path pairs (x,y) per sample; keep lens general otherwise
    x, y, lx, ly = _make_inputs(device, equal_lens=equal_lens)
    sdtw = SoftDTW(gamma=GAMMA, normalize=normalize, fused=fused)
    out = sdtw(x, y, lens_x=lx, lens_y=ly)
    ref = _ref_sliced(x, y, lx, ly, GAMMA, normalize)
    assert torch.allclose(out, ref, atol=1e-8, rtol=1e-8), (out - ref).abs().max().item()


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_grads_match_sliced_and_padding_grad_is_zero(device, fused, normalize):
    if fused and device == "cpu":
        pytest.skip("fused requires CUDA")
    equal_lens = normalize
    x, y, lx, ly = _make_inputs(device, requires_grad=True, equal_lens=equal_lens)
    sdtw = SoftDTW(gamma=GAMMA, normalize=normalize, fused=fused)
    sdtw(x, y, lens_x=lx, lens_y=ly).sum().backward()
    grad_batched = x.grad.detach().clone()
    x.grad = None

    _ref_sliced(x, y, lx, ly, GAMMA, normalize).sum().backward()
    grad_ref = x.grad.detach().clone()
    x.grad = None

    for b in range(x.shape[0]):
        n = int(lx[b])
        assert torch.allclose(grad_batched[b, :n], grad_ref[b, :n], atol=1e-7, rtol=1e-6), (
            b,
            (grad_batched[b, :n] - grad_ref[b, :n]).abs().max().item(),
        )
        # padding frames: exactly zero, not merely small
        assert (grad_batched[b, n:] == 0).all(), (b, grad_batched[b, n:].abs().max().item())


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("fused", [False, True])
def test_full_lens_matches_none(device, fused):
    if fused and device == "cpu":
        pytest.skip("fused requires CUDA")
    torch.manual_seed(7)
    B, N, M, D = 3, 20, 24, 5
    x = torch.randn(B, N, D, dtype=torch.float64, device=device)
    y = torch.randn(B, M, D, dtype=torch.float64, device=device)
    sdtw = SoftDTW(gamma=GAMMA, fused=fused)
    out_none = sdtw(x, y)
    out_full = sdtw(
        x,
        y,
        lens_x=torch.full((B,), N, dtype=torch.int64, device=device),
        lens_y=torch.full((B,), M, dtype=torch.int64, device=device),
    )
    assert torch.allclose(out_none, out_full, atol=0, rtol=0)


@pytest.mark.parametrize("fused", [False, True])
def test_long_seq_tiled_path_with_lens(fused):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    # N > 1024 forces the tiled anti-diagonal path
    B, N, D = 2, 1100, 4
    g = torch.Generator(device="cpu").manual_seed(99)
    x = torch.randn(B, N, D, generator=g, dtype=torch.float64).cuda()
    y = torch.randn(B, N, D, generator=g, dtype=torch.float64).cuda()
    lx = torch.tensor([1050, 300], dtype=torch.int64)
    ly = torch.tensor([1080, 200], dtype=torch.int64)
    x[0, 1050:] = 500.0
    y[1, 200:] = -500.0
    sdtw = SoftDTW(gamma=GAMMA, fused=fused)
    out = sdtw(x, y, lens_x=lx.cuda(), lens_y=ly.cuda())
    ref = _ref_sliced(x, y, lx, ly, GAMMA, normalize=False)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-8), (out - ref).abs().max().item()


def test_gradcheck_with_lens_cpu():
    torch.manual_seed(3)
    B, N, D = 2, 8, 3
    x = torch.randn(B, N, D, dtype=torch.float64, requires_grad=True)
    y = torch.randn(B, N, D, dtype=torch.float64)
    lx = torch.tensor([6, 8])
    ly = torch.tensor([8, 5])
    sdtw = SoftDTW(gamma=GAMMA, fused=False)
    assert torch.autograd.gradcheck(
        lambda xx: sdtw(xx, y, lens_x=lx, lens_y=ly).sum(), (x,), eps=1e-6, atol=1e-5
    )


def test_lens_validation():
    x = torch.randn(2, 10, 3)
    y = torch.randn(2, 10, 3)
    sdtw = SoftDTW(gamma=GAMMA, fused=False)
    with pytest.raises(ValueError):
        sdtw(x, y, lens_x=torch.tensor([11, 5]))  # > N
    with pytest.raises(ValueError):
        sdtw(x, y, lens_x=torch.tensor([0, 5]))  # < 1
    with pytest.raises(ValueError):
        sdtw(x, y, lens_x=torch.tensor([5]))  # wrong shape
    with pytest.raises(TypeError):
        sdtw(x, y, lens_x=torch.tensor([5.0, 5.0]))  # float dtype
