from __future__ import annotations

import math
import numpy as np
import torch
from numba_cuda_mlir import cuda
from numba import jit, prange

from .kernels import softdtw_forward_kernel, softdtw_forward_diag_cuda
from .kernels import softdtw_backward_log_cuda, softdtw_backward_log_diag_cuda
from .kernels import softdtw_forward_diag_sqeuclid_cuda
from .kernels import softdtw_backward_log_diag_sqeuclid_cuda

# GLOBALS
TPB_LONG = 256


# HELPERS
def _diag_bounds(p: int, N: int, M: int) -> tuple[int, int]:
    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    return i_min, i_max



def _threads_and_passes(N: int, M: int) -> tuple[int, int]:
    tpb = max(N, M)
    n_passes = 2 * tpb - 1
    return tpb, n_passes


def _resolve_lens(
    lens: torch.Tensor | None,
    B: int,
    maxlen: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Normalize an optional per-sample length tensor to (B,) int32 on
    `device`. None means every sample uses the full padded length."""
    if lens is None:
        return torch.full((B,), maxlen, dtype=torch.int32, device=device)
    if not torch.is_tensor(lens):
        raise TypeError(f"{name} must be a torch.Tensor or None, got {type(lens)}")
    if lens.dim() != 1 or lens.numel() != B:
        raise ValueError(f"{name} must have shape ({B},), got {tuple(lens.shape)}")
    if lens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"{name} must be an integer tensor, got {lens.dtype}")
    lens = lens.to(device=device, dtype=torch.int32).contiguous()
    lo = int(lens.min().item())
    hi = int(lens.max().item())
    if lo < 1 or hi > maxlen:
        raise ValueError(f"{name} values must be in [1, {maxlen}], got range [{lo}, {hi}]")
    return lens


def _per_sample_out(R: torch.Tensor, lens_x: torch.Tensor, lens_y: torch.Tensor) -> torch.Tensor:
    """out[b] = R[b, LX[b], LY[b]] -- the final DP cell of each sample in
    padded coordinates (unpadded (n-1, m-1) maps to padded (n, m))."""
    B = R.shape[0]
    bidx = torch.arange(B, device=R.device)
    return R[bidx, lens_x.long(), lens_y.long()].contiguous()


def _region_masks(
    B: int,
    Np2: int,
    Mp2: int,
    lens_x: torch.Tensor,
    lens_y: torch.Tensor,
    device: torch.device,
):
    """Boolean masks over padded (B, N+2, M+2) coordinates.

    reach[b, i, j] is True inside sample b's reachable DP region
    [0..n] x [0..m] (padded coords). Everything outside must read as
    -inf in R_work during the backward pass: with per-sample lengths the
    cells between a sample's true end and the padded buffer end hold
    +inf left over from the forward init, and a raw +inf neighbor read
    would produce -inf + inf = NaN in log space.

    interior[b, i, j] is True on the cells holding real distances
    ([1..n] x [1..m] in padded coords); used to zero D_pad outside.
    """
    ii = torch.arange(Np2, device=device).view(1, -1, 1)
    jj = torch.arange(Mp2, device=device).view(1, 1, -1)
    lx = lens_x.long().view(-1, 1, 1)
    ly = lens_y.long().view(-1, 1, 1)
    reach = (ii <= lx) & (jj <= ly)
    interior = (ii >= 1) & (ii <= lx) & (jj >= 1) & (jj <= ly)
    return reach, interior


def _prepare_backward_boundaries(
    R: torch.Tensor,
    lens_x: torch.Tensor,
    lens_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample analog of the classic global boundary setup
    (R[:, :, -1] = -inf; R[:, -1, :] = -inf; R[:, -1, -1] = R[:, -2, -2];
    logE[:, -1, -1] = 0), placed at each sample's own (n+1, m+1) corner.

    Returns (R_work, logE) ready for the backward kernels.
    """
    B, Np2, Mp2 = R.shape
    dev = R.device
    reach, _ = _region_masks(B, Np2, Mp2, lens_x, lens_y, dev)

    neg_inf = torch.tensor(-math.inf, device=dev, dtype=R.dtype)
    R_work = torch.where(reach, R, neg_inf)

    bidx = torch.arange(B, device=dev)
    lx = lens_x.long()
    ly = lens_y.long()
    R_work[bidx, lx + 1, ly + 1] = R[bidx, lx, ly]

    logE = torch.full((B, Np2, Mp2), -math.inf, device=dev, dtype=R.dtype)
    logE[bidx, lx + 1, ly + 1] = 0.0
    return R_work, logE


# MAIN - on-the-fly D
def softdtw_forward_cuda_fused_sqeuclid(
    X: torch.Tensor,
    Y: torch.Tensor,
    gamma: float,
    bandwidth: float,
    lens_x: torch.Tensor | None = None,
    lens_y: torch.Tensor | None = None,
):
    """
    Fused SoftDTW forward for squared-euclidean distance that does NOT materialize D (B,N,M).

    X: (B,N,D), Y: (B,M,D) CUDA tensors
    lens_x/lens_y: optional (B,) int tensors of per-sample true lengths
    Returns: (out: (B,), R: (B,N+2,M+2))
    """
    if not (X.is_cuda and Y.is_cuda):
        raise ValueError("Expected CUDA tensors X and Y")
    if X.dim() != 3 or Y.dim() != 3:
        raise ValueError(f"Expected X,Y as (B,N,D)/(B,M,D). Got {tuple(X.shape)} and {tuple(Y.shape)}")
    if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
        raise ValueError(f"Batch/features mismatch: {tuple(X.shape)} vs {tuple(Y.shape)}")

    # Detach before passing to numba
    X_ = X.detach().contiguous()
    Y_ = Y.detach().contiguous()

    B, N, D = X_.shape
    M = Y_.shape[1]

    LX = _resolve_lens(lens_x, B, N, X_.device, "lens_x")
    LY = _resolve_lens(lens_y, B, M, X_.device, "lens_y")

    # Allocate DP table
    R = torch.full((B, N + 2, M + 2), math.inf, device=X_.device, dtype=X_.dtype)
    R[:, 0, 0] = 0.0

    X_ca = cuda.as_cuda_array(X_)
    Y_ca = cuda.as_cuda_array(Y_)
    R_ca = cuda.as_cuda_array(R)
    LX_ca = cuda.as_cuda_array(LX)
    LY_ca = cuda.as_cuda_array(LY)

    inv_bw = float(bandwidth)  # can be -1.0 to disable

    # Anti-diagonals over unpadded (i,j): p = i + j. Iterate to the padded
    # bound; samples whose own n+m-1 is smaller no-op via in-kernel bounds.
    for p in range(N + M - 1):
        i_min = max(0, p - (M - 1))
        i_max = min(N - 1, p)
        if i_max < i_min:
            continue
        diag_len = i_max - i_min + 1
        grid_x = (diag_len + TPB_LONG - 1) // TPB_LONG

        # grid=(grid_x, B), so batch = blockIdx.y in kernel
        softdtw_forward_diag_sqeuclid_cuda[(grid_x, B), TPB_LONG](
            X_ca,
            Y_ca,
            R_ca,
            float(gamma),
            inv_bw,
            LX_ca,
            LY_ca,
            D,
            p,
        )

    out = _per_sample_out(R, LX, LY)
    return out, R


def softdtw_backward_cuda_fused_sqeuclid(
    X: torch.Tensor,
    Y: torch.Tensor,
    R: torch.Tensor,
    gamma: float,
    bandwidth: float,
    lens_x: torch.Tensor | None = None,
    lens_y: torch.Tensor | None = None,
):
    """
    Fused SoftDTW backward (log-space) for squared-euclidean distance that does NOT materialize D_pad.

    Inputs:
      X: (B,N,D) CUDA
      Y: (B,M,D) CUDA
      R: (B,N+2,M+2) CUDA (from forward)
    Returns:
      E: (B,N,M) CUDA  (E = d SoftDTW / d D  in linear space, via exp(logE));
      exactly 0 outside each sample's (LX[b], LY[b]) region.
    """
    if not (X.is_cuda and Y.is_cuda and R.is_cuda):
        raise ValueError("Expected CUDA tensors X, Y, R")
    if X.dim() != 3 or Y.dim() != 3:
        raise ValueError(f"Expected X,Y as (B,N,D)/(B,M,D). Got {tuple(X.shape)} and {tuple(Y.shape)}")
    if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
        raise ValueError(f"Batch/features mismatch: {tuple(X.shape)} vs {tuple(Y.shape)}")

    # Detach before passing to numba
    X_ = X.detach().contiguous()
    Y_ = Y.detach().contiguous()

    B, N, D = X_.shape
    M = Y_.shape[1]

    if R.shape != (B, N + 2, M + 2):
        raise ValueError(f"Expected R shape {(B, N+2, M+2)}, got {tuple(R.shape)}")

    LX = _resolve_lens(lens_x, B, N, X_.device, "lens_x")
    LY = _resolve_lens(lens_y, B, M, X_.device, "lens_y")

    R_ = R.contiguous()

    # ---------- per-sample boundary conditions for R and logE ----------
    R_work, logE = _prepare_backward_boundaries(R_, LX, LY)

    X_ca = cuda.as_cuda_array(X_)
    Y_ca = cuda.as_cuda_array(Y_)
    Rw_ca = cuda.as_cuda_array(R_work)
    logE_ca = cuda.as_cuda_array(logE)
    LX_ca = cuda.as_cuda_array(LX)
    LY_ca = cuda.as_cuda_array(LY)

    inv_gamma = float(1.0 / gamma)
    bw = float(bandwidth)

    # Reverse anti-diagonals over unpadded indices p = i + j, starting from (N-1)+(M-1)-1 = N+M-2 down to 0
    for p in range(N + M - 2, -1, -1):
        i_min = max(0, p - (M - 1))
        i_max = min(N - 1, p)
        if i_max < i_min:
            continue
        diag_len = i_max - i_min + 1
        grid_x = (diag_len + TPB_LONG - 1) // TPB_LONG

        softdtw_backward_log_diag_sqeuclid_cuda[(grid_x, B), TPB_LONG](
            X_ca,
            Y_ca,
            Rw_ca,
            logE_ca,
            inv_gamma,
            bw,
            LX_ca,
            LY_ca,
            D,
            p,
        )

    # Clear the per-sample seed before cropping: for a short sample the
    # (n+1, m+1) virtual corner falls INSIDE the [1..N]x[1..M] crop and its
    # exp(0)=1 would leak a spurious gradient onto padding frames. It is
    # only needed while the recurrence runs.
    bidx = torch.arange(B, device=logE.device)
    logE[bidx, LX.long() + 1, LY.long() + 1] = -math.inf

    # crop + exp (cells outside each sample's region stayed -inf -> exp = 0)
    E = torch.exp(logE[:, 1:N + 1, 1:M + 1]).contiguous()
    return E



# MAIN - Full D Matrix
def softdtw_forward_cuda(
    D: torch.Tensor,
    gamma: float,
    bandwidth: float,
    lens_x: torch.Tensor | None = None,
    lens_y: torch.Tensor | None = None,
):
    if not D.is_cuda:
        raise ValueError("Expected CUDA tensor D")

    D_ = D.detach().contiguous()
    B, N, M = D_.shape
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    LX = _resolve_lens(lens_x, B, N, D_.device, "lens_x")
    LY = _resolve_lens(lens_y, B, M, D_.device, "lens_y")

    # Allocate DP table
    R = torch.full((B, N + 2, M + 2), math.inf, device=D_.device, dtype=D_.dtype)
    R[:, 0, 0] = 0.0

    LX_ca = cuda.as_cuda_array(LX)
    LY_ca = cuda.as_cuda_array(LY)

    # --- Fast path: one block per batch element ---
    tpb, n_passes = _threads_and_passes(N, M)
    USE_FAST_PATH = (tpb <= 1024)

    if USE_FAST_PATH:
        softdtw_forward_kernel[B, tpb](
            cuda.as_cuda_array(D_),
            float(gamma),
            float(bandwidth),
            LX_ca,
            LY_ca,
            n_passes,
            cuda.as_cuda_array(R),
        )
        out = _per_sample_out(R, LX, LY)
        return out, R

    # --- Long sequence path: tiled anti-diagonal launches ---


    D_ca = cuda.as_cuda_array(D_)
    R_ca = cuda.as_cuda_array(R)

    # Iterate anti-diagonals in unpadded (i,j) coords over D (shape N x M)
    for p in range(N + M - 1):
        i_min, i_max = _diag_bounds(p, N, M)
        if i_max < i_min:
            continue
        diag_len = i_max - i_min + 1
        grid_x = (diag_len + TPB_LONG - 1) // TPB_LONG

        # grid=(grid_x, B) so batch index is blockIdx.y inside kernel
        softdtw_forward_diag_cuda[(grid_x, B), TPB_LONG](
            D_ca,
            R_ca,
            float(gamma),
            float(bandwidth),
            LX_ca,
            LY_ca,
            p,
        )

    out = _per_sample_out(R, LX, LY)
    return out, R


def softdtw_backward_cuda_log(
    D: torch.Tensor,
    R: torch.Tensor,
    gamma: float,
    bandwidth: float,
    lens_x: torch.Tensor | None = None,
    lens_y: torch.Tensor | None = None,
):
    if not D.is_cuda:
        raise ValueError("Expected CUDA tensor D")

    D_ = D.detach().contiguous()
    B, N, M = D_.shape
    R = R.contiguous()

    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    LX = _resolve_lens(lens_x, B, N, D_.device, "lens_x")
    LY = _resolve_lens(lens_y, B, M, D_.device, "lens_y")

    # ---------- pad D, zeroed outside each sample's region ----------
    # The recurrence at a sample's last valid cell reads D_pad at its
    # (n+1, m+1) seed corner and expects 0 there; with per-sample lengths
    # that cell would otherwise hold a real distance computed from padding.
    D_pad = torch.zeros((B, N + 2, M + 2), device=D_.device, dtype=D_.dtype)
    D_pad[:, 1:N + 1, 1:M + 1] = D_
    _, interior = _region_masks(B, N + 2, M + 2, LX, LY, D_.device)
    D_pad = D_pad * interior.to(D_pad.dtype)

    # ---------- per-sample boundary conditions for R and logE ----------
    R_work, logE = _prepare_backward_boundaries(R, LX, LY)

    LX_ca = cuda.as_cuda_array(LX)
    LY_ca = cuda.as_cuda_array(LY)

    # ---------- choose fast vs tiled ----------
    tpb, n_passes = _threads_and_passes(N, M)
    USE_FAST_PATH = (tpb <= 1024)

    if USE_FAST_PATH:
        # fast path: diagonal backward kernel (single block per batch)
        softdtw_backward_log_cuda[B, tpb](
            cuda.as_cuda_array(D_pad),
            cuda.as_cuda_array(R_work),
            float(1.0 / gamma),
            float(bandwidth),
            LX_ca,
            LY_ca,
            n_passes,
            cuda.as_cuda_array(logE),
        )
    else:
        # tiled path: launch one kernel per anti-diagonal in reverse order

        Dp_ca = cuda.as_cuda_array(D_pad)
        Rw_ca = cuda.as_cuda_array(R_work)
        logE_ca = cuda.as_cuda_array(logE)

        inv_gamma = float(1.0 / gamma)
        bw = float(bandwidth)
        if bw <= 0:
            bw = -1.0

        # unpadded indices (i,j) are 0..N-1, 0..M-1, diagonals p = i+j
        for p in range(N + M - 2, -1, -1):
            i_min, i_max = _diag_bounds(p, N, M)
            if i_max < i_min:
                continue
            diag_len = i_max - i_min + 1
            grid_x = (diag_len + TPB_LONG - 1) // TPB_LONG

            softdtw_backward_log_diag_cuda[(grid_x, B), TPB_LONG](
                Dp_ca,
                Rw_ca,
                logE_ca,
                inv_gamma,
                bw,
                LX_ca,
                LY_ca,
                p,
            )

    # Clear the per-sample seed before cropping (see the fused backward's
    # comment: a short sample's virtual corner falls inside the crop).
    bidx = torch.arange(B, device=logE.device)
    logE[bidx, LX.long() + 1, LY.long() + 1] = -math.inf

    # crop + exp (cells outside each sample's region stayed -inf -> exp = 0)
    E = torch.exp(logE[:, 1:N + 1, 1:M + 1]).contiguous()
    return E




# ---- CPU reference (optional but useful for tests) ----

@jit(nopython=True, parallel=True)
def _softdtw_forward_cpu_np(D: np.ndarray, gamma: float, bandwidth: float, LX: np.ndarray, LY: np.ndarray):
    B, N, M = D.shape
    R = np.ones((B, N + 2, M + 2), dtype=D.dtype) * np.inf
    R[:, 0, 0] = 0.0
    for b in prange(B):
        n = LX[b]
        m = LY[b]
        for j in range(1, m + 1):
            for i in range(1, n + 1):
                if 0 < bandwidth < abs(i - j):
                    continue
                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R


@jit(nopython=True, parallel=True)
def _softdtw_backward_cpu_np(D_: np.ndarray, R: np.ndarray, gamma: float, bandwidth: float, LX: np.ndarray, LY: np.ndarray):
    B, N, M = D_.shape
    D = np.zeros((B, N + 2, M + 2), dtype=D_.dtype)
    E = np.zeros((B, N + 2, M + 2), dtype=D_.dtype)

    for b in prange(B):
        n = LX[b]
        m = LY[b]

        # pad D inside this sample's region only (rest stays 0)
        for i in range(n):
            for j in range(m):
                D[b, i + 1, j + 1] = D_[b, i, j]

        # per-sample boundary conditions at the (n+1, m+1) virtual corner
        for jj in range(M + 2):
            R[b, n + 1, jj] = -np.inf
        for ii in range(N + 2):
            R[b, ii, m + 1] = -np.inf
        R[b, n + 1, m + 1] = R[b, n, m]
        E[b, n + 1, m + 1] = 1.0

        for j in range(m, 0, -1):
            for i in range(n, 0, -1):
                if np.isinf(R[b, i, j]):
                    R[b, i, j] = -np.inf
                if 0 < bandwidth < abs(i - j):
                    continue
                a0 = (R[b, i + 1, j] - R[b, i, j] - D[b, i + 1, j]) / gamma
                b0 = (R[b, i, j + 1] - R[b, i, j] - D[b, i, j + 1]) / gamma
                c0 = (R[b, i + 1, j + 1] - R[b, i, j] - D[b, i + 1, j + 1]) / gamma
                a = np.exp(a0); bb = np.exp(b0); c = np.exp(c0)
                E[b, i, j] = E[b, i + 1, j] * a + E[b, i, j + 1] * bb + E[b, i + 1, j + 1] * c

        # clear the seed: for a short sample the (n+1, m+1) virtual corner
        # falls inside the [1..N]x[1..M] crop and its 1.0 would leak a
        # spurious gradient onto padding frames
        E[b, n + 1, m + 1] = 0.0

    return E[:, 1:N + 1, 1:M + 1]


def softdtw_forward_cpu(
    D: torch.Tensor,
    gamma: float,
    bandwidth: float,
    lens_x: torch.Tensor | None = None,
    lens_y: torch.Tensor | None = None,
):
    B, N, M = D.shape
    LX = _resolve_lens(lens_x, B, N, D.device, "lens_x")
    LY = _resolve_lens(lens_y, B, M, D.device, "lens_y")
    D_np = D.detach().cpu().numpy()
    LX_np = LX.cpu().numpy().astype(np.int64)
    LY_np = LY.cpu().numpy().astype(np.int64)
    R_np = _softdtw_forward_cpu_np(D_np, float(gamma), float(bandwidth), LX_np, LY_np)
    R = torch.from_numpy(R_np).to(D.device).type_as(D)
    out = _per_sample_out(R, LX, LY)
    return out, R


def softdtw_backward_cpu(
    D: torch.Tensor,
    R: torch.Tensor,
    gamma: float,
    bandwidth: float,
    lens_x: torch.Tensor | None = None,
    lens_y: torch.Tensor | None = None,
):
    B, N, M = D.shape
    LX = _resolve_lens(lens_x, B, N, D.device, "lens_x")
    LY = _resolve_lens(lens_y, B, M, D.device, "lens_y")
    D_np = D.detach().cpu().numpy()
    R_np = R.detach().cpu().numpy().copy()  # .copy() prevents in-place mutation of saved autograd tensor
    LX_np = LX.cpu().numpy().astype(np.int64)
    LY_np = LY.cpu().numpy().astype(np.int64)
    E_np = _softdtw_backward_cpu_np(D_np, R_np, float(gamma), float(bandwidth), LX_np, LY_np)
    return torch.from_numpy(E_np).to(D.device).type_as(D).contiguous()
