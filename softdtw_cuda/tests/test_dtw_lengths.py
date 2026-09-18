"""Per-sample length (padding) support for the exact hard-DTW path.

Mirrors test_lengths.py's approach for SoftDTW: ground truth throughout is a
Python loop of per-sample SLICED calls (batch=1 each, no padding anywhere),
which exercises none of the length machinery. The batched call with
lens_x/lens_y must match it exactly (hard-DTW has no smoothing to blur small
differences). DTW is forward-only (no autograd graph), so unlike
test_lengths.py there is no gradient/padding-gradient coverage here.
"""
from __future__ import annotations

import torch

try:
    import pytest
except ImportError:                       # allow running as a plain script (no pytest)
    class _Mark:
        def __getattr__(self, _):
            return lambda *a, **k: (lambda f: f)
    class _Pytest:
        mark = _Mark()
    pytest = _Pytest()

from softdtw_cuda import DTW


def _ref_sliced(x, y, lens_x, lens_y, fused, bandwidth):
    """Per-sample sliced reference: unpadded (batch=1) DTW calls."""
    dtw = DTW(dist="sqeuclidean", fused=fused, bandwidth=bandwidth)
    outs = []
    for b in range(x.shape[0]):
        xb = x[b : b + 1, : int(lens_x[b])]
        yb = y[b : b + 1, : int(lens_y[b])]
        outs.append(dtw(xb, yb))
    return torch.cat(outs, dim=0)


def _make_inputs(device, dtype=torch.float64, B=5, N=48, M=None, D=7,
                  equal_lens=False):
    if M is None:
        M = N
    g = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(B, N, D, generator=g, dtype=dtype).to(device)
    y = torch.randn(B, M, D, generator=g, dtype=dtype).to(device)
    lens_x = torch.tensor([N, 1, max(1, N // 3), max(1, N - 2), N // 2], dtype=torch.int64)[:B]
    if equal_lens:
        lens_x = torch.minimum(lens_x, torch.tensor(M, dtype=torch.int64))
        lens_y = lens_x.clone()
    else:
        lens_y = torch.tensor([M, min(M, 5), max(1, M // 4), M, max(1, M // 3)], dtype=torch.int64)[:B]
    for b in range(B):
        x[b, lens_x[b]:] = 1000.0 + b
        y[b, lens_y[b]:] = -2000.0 - b
    return x, y, lens_x.to(device), lens_y.to(device)


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("bandwidth", [None, 3.0])
def test_forward_matches_sliced(device, fused, bandwidth):
    if fused and device == "cpu":
        pytest.skip("fused requires CUDA")
    x, y, lx, ly = _make_inputs(device)
    dtw = DTW(dist="sqeuclidean", fused=fused, bandwidth=bandwidth)
    out = dtw(x, y, lens_x=lx, lens_y=ly)
    ref = _ref_sliced(x, y, lx, ly, fused, bandwidth)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("fused", [False, True])
def test_nm_swap_with_unequal_lens(device, fused):
    """Padded N > M (forces the fused kernel's buffer-axis swap) combined
    with per-sample lengths that differ, including one sample whose length
    equals the padded dim and another much shorter -- exercises the swap
    decision and the per-sample terminal-cell gather together."""
    if fused and device == "cpu":
        pytest.skip("fused requires CUDA")
    x, y, lx, ly = _make_inputs(device, B=3, N=16, M=4, D=5)
    lx = torch.tensor([16, 5, 9], dtype=torch.int64, device=device)
    ly = torch.tensor([4, 4, 3], dtype=torch.int64, device=device)
    for b in range(3):
        x[b, lx[b]:] = 1000.0 + b
        y[b, ly[b]:] = -2000.0 - b
    dtw = DTW(dist="sqeuclidean", fused=fused)
    out = dtw(x, y, lens_x=lx, lens_y=ly)
    ref = _ref_sliced(x, y, lx, ly, fused, None)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_unfused_cpu_agree_with_lens():
    x, y, lx, ly = _make_inputs("cpu", dtype=torch.float32, B=4, N=20, M=24, D=6)
    cpu_out = DTW(dist="sqeuclidean", fused=False)(x, y, lens_x=lx, lens_y=ly)

    xg, yg, lxg, lyg = x.cuda(), y.cuda(), lx.cuda(), ly.cuda()
    fused_out = DTW(dist="sqeuclidean", fused=True)(xg, yg, lens_x=lxg, lens_y=lyg).cpu()
    unfused_out = DTW(dist="sqeuclidean", fused=False)(xg, yg, lens_x=lxg, lens_y=lyg).cpu()

    torch.testing.assert_close(fused_out, cpu_out, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(unfused_out, cpu_out, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("fused", [False, True])
def test_full_lens_matches_none(device, fused):
    if fused and device == "cpu":
        pytest.skip("fused requires CUDA")
    torch.manual_seed(7)
    B, N, M, D = 3, 20, 24, 5
    x = torch.randn(B, N, D, dtype=torch.float64, device=device)
    y = torch.randn(B, M, D, dtype=torch.float64, device=device)
    dtw = DTW(dist="sqeuclidean", fused=fused)
    out_none = dtw(x, y)
    out_full = dtw(
        x, y,
        lens_x=torch.full((B,), N, dtype=torch.int64, device=device),
        lens_y=torch.full((B,), M, dtype=torch.int64, device=device),
    )
    torch.testing.assert_close(out_none, out_full, atol=0, rtol=0)


@pytest.mark.parametrize("fused", [False, True])
def test_long_seq_with_lens(fused):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    B, N, D = 2, 1500, 4
    g = torch.Generator(device="cpu").manual_seed(99)
    x = torch.randn(B, N, D, generator=g, dtype=torch.float32).cuda()
    y = torch.randn(B, N, D, generator=g, dtype=torch.float32).cuda()
    lx = torch.tensor([1050, 300], dtype=torch.int64)
    ly = torch.tensor([1080, 200], dtype=torch.int64)
    x[0, 1050:] = 500.0
    y[1, 200:] = -500.0
    dtw = DTW(dist="sqeuclidean", fused=fused)
    out = dtw(x, y, lens_x=lx.cuda(), lens_y=ly.cuda())
    ref = _ref_sliced(x, y, lx, ly, fused, None)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-3)


def test_lens_validation():
    x = torch.randn(2, 10, 3)
    y = torch.randn(2, 10, 3)
    dtw = DTW(dist="sqeuclidean", fused=False)
    with pytest.raises(ValueError):
        dtw(x, y, lens_x=torch.tensor([11, 5]))  # > N
    with pytest.raises(ValueError):
        dtw(x, y, lens_x=torch.tensor([0, 5]))  # < 1
    with pytest.raises(ValueError):
        dtw(x, y, lens_x=torch.tensor([5]))  # wrong shape
    with pytest.raises(TypeError):
        dtw(x, y, lens_x=torch.tensor([5.0, 5.0]))  # float dtype


if __name__ == "__main__":
    x, y, lx, ly = _make_inputs("cpu")
    dtw = DTW(dist="sqeuclidean", fused=False)
    out = dtw(x, y, lens_x=lx, lens_y=ly)
    ref = _ref_sliced(x, y, lx, ly, False, None)
    ok = torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
    print(f"[{'PASS' if ok else 'FAIL'}] CPU forward matches sliced, maxdiff "
          f"{(out - ref).abs().max().item():.2e}")
    if torch.cuda.is_available():
        print("Run under pytest for full CUDA coverage (fused/unfused/swap/long-seq).")
