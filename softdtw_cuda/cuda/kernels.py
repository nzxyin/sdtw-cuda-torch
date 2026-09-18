from __future__ import annotations

import math
from numba_cuda_mlir import cuda

# All kernels take per-sample length arrays LX (B,) and LY (B,) of int32:
# sample b's true (unpadded) lengths are n = LX[b] <= N and m = LY[b] <= M,
# where N/M are the padded buffer dims. Cells outside [0, n) x [0, m) are
# never read or written by the DP recurrence; callers pass full-length
# arrays (LX[b] = N, LY[b] = M) to recover the classic fixed-length
# behavior. The launcher is responsible for the per-sample boundary
# conditions the backward kernels rely on (R_work masked to -inf outside
# each sample's region with the seed corner at (n+1, m+1), D_pad zeroed
# outside, logE seeded at (n+1, m+1)).


@cuda.jit
def softdtw_forward_diag_sqeuclid_cuda(X, Y, R, gamma, bandwidth, LX, LY, D, p):
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    N = LX[b]
    M = LY[b]

    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if diag_len <= 0 or t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    if bandwidth > 0 and abs(i - j) > bandwidth:
        return

    # cost = ||X[b,i,:] - Y[b,j,:]||^2
    cost = 0.0
    for k in range(D):
        diff = X[b, i, k] - Y[b, j, k]
        cost += diff * diff

    inv_gamma = 1.0 / gamma

    r0 = -R[b, ip - 1, jp - 1] * inv_gamma
    r1 = -R[b, ip - 1, jp]     * inv_gamma
    r2 = -R[b, ip,     jp - 1] * inv_gamma

    rmax = r0
    if r1 > rmax: rmax = r1
    if r2 > rmax: rmax = r2

    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
    softmin = -gamma * (math.log(rsum) + rmax)

    R[b, ip, jp] = cost + softmin


@cuda.jit
def dtw_forward_diag_stream_sqeuclid_cuda(X, Y, prev2, prev1, curr, bandwidth, N, M, LX, LY, out, D, p):
    """Fused *hard* DTW forward, streaming: one anti-diagonal per launch over
    three rotating 1-D buffers -> O(min(N, M)) memory (no (N, M) DP table).

    Cells on anti-diagonal ``p = i + j`` depend only on diagonals ``p-1`` and
    ``p-2``, so ``prev1``/``prev2`` hold those two and ``curr`` receives this
    one; the host rotates the three buffers after each launch. Buffers are
    indexed by absolute ``i`` and have length ``N`` (the caller ensures
    ``N <= M`` via ``DTW(x, y) == DTW(y, x)``). The aggregation is a true
    ``min`` and there is no gamma. Forward-only, so there is no backward kernel.

    Per-sample lengths (LX/LY, whole-batch padded N/M still bound the sweep
    and buffer indexing): each rotating buffer slot is reused every 3
    diagonals, so a short sample's terminal cell can't simply be read back
    from the buffers once the sweep moves past it -- it's captured into
    ``out`` (B,) the instant it's computed instead. ``out`` must be
    pre-filled with +inf by the host: a terminal cell pruned by the
    bandwidth is never written here, and +inf is the correct "unreachable"
    answer for it.
    """
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    # Out-of-band cells must still write +inf: their slot is reused every 3
    # diagonals and may be read (by index) from a later in-band diagonal.
    if bandwidth > 0 and abs(i - j) > bandwidth:
        curr[b, i] = math.inf
        return

    # cost = ||X[b,i,:] - Y[b,j,:]||^2  (on the fly; no (N,M) cost tensor)
    cost = 0.0
    for k in range(D):
        diff = X[b, i, k] - Y[b, j, k]
        cost += diff * diff

    # min over the three predecessors; the (0,0) corner is seeded with 0. Reads
    # are guarded by predecessor validity, so only written slots are ever read.
    if i == 0 and j == 0:
        m = 0.0
    else:
        m = math.inf
        if i - 1 >= 0 and j - 1 >= 0:      # (i-1, j-1) on diagonal p-2
            v = prev2[b, i - 1]
            if v < m: m = v
        if i - 1 >= 0:                      # (i-1, j)   on diagonal p-1
            v = prev1[b, i - 1]
            if v < m: m = v
        if j - 1 >= 0:                      # (i, j-1)   on diagonal p-1
            v = prev1[b, i]
            if v < m: m = v

    result = cost + m
    curr[b, i] = result
    if i == LX[b] - 1 and j == LY[b] - 1:
        out[b] = result


@cuda.jit
def softdtw_backward_log_diag_sqeuclid_cuda(X, Y, R, logE, inv_gamma, bandwidth, LX, LY, D, p):
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    N = LX[b]
    M = LY[b]

    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if diag_len <= 0 or t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    # Demote unreachable / out-of-band cells (R == +inf) to -inf *before* the
    # bandwidth prune, so that in-band neighbours reading them contribute
    # exp(-inf)=0 instead of producing (-inf)+(+inf)=NaN in the log-space sum.
    if math.isinf(R[b, ip, jp]):
        R[b, ip, jp] = -math.inf

    if bandwidth > 0 and abs(i - j) > bandwidth:
        return

    Rij = R[b, ip, jp]

    # On-the-fly transition costs. The per-sample bounds (i + 1 < N with
    # N = LX[b], not the padded dim) are correctness-critical: at the last
    # valid cell (n-1, m-1) the diagonal transition targets the virtual
    # seed corner (n+1, m+1), whose cost must be 0 -- computing a real
    # distance against padding frames there would corrupt the gradient.

    # cost_down: (i+1, j)
    cost_down = 0.0
    if i + 1 < N:
        for k in range(D):
            diff = X[b, i + 1, k] - Y[b, j, k]
            cost_down += diff * diff

    # cost_right: (i, j+1)
    cost_right = 0.0
    if j + 1 < M:
        for k in range(D):
            diff = X[b, i, k] - Y[b, j + 1, k]
            cost_right += diff * diff

    # cost_diag: (i+1, j+1)
    cost_diag = 0.0
    if (i + 1 < N) and (j + 1 < M):
        for k in range(D):
            diff = X[b, i + 1, k] - Y[b, j + 1, k]
            cost_diag += diff * diff

    la = (R[b, ip + 1, jp]     - Rij - cost_down)  * inv_gamma
    lb = (R[b, ip,     jp + 1] - Rij - cost_right) * inv_gamma
    lc = (R[b, ip + 1, jp + 1] - Rij - cost_diag)  * inv_gamma

    t1 = logE[b, ip + 1, jp]     + la
    t2 = logE[b, ip,     jp + 1] + lb
    t3 = logE[b, ip + 1, jp + 1] + lc

    m = t1
    if t2 > m: m = t2
    if t3 > m: m = t3

    if m == -math.inf:
        logE[b, ip, jp] = -math.inf
    else:
        logE[b, ip, jp] = m + math.log(math.exp(t1 - m) + math.exp(t2 - m) + math.exp(t3 - m))



@cuda.jit
def softdtw_forward_kernel(D, gamma, bandwidth, LX, LY, n_passes, R):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    max_i = LX[b]
    max_j = LY[b]

    I = tid
    inv_gamma = 1.0 / gamma

    # n_passes is sized for the largest sample in the batch; passes beyond
    # this sample's own 2*max(n,m)-1 are no-ops (the I+J==p guard never
    # fires for them), and the loop count stays uniform per block so
    # syncthreads is safe.
    for p in range(n_passes):
        J = max(0, min(p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        if I + J == p and (I < max_i and J < max_j):
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
        cuda.syncthreads()

@cuda.jit
def softdtw_forward_diag_cuda(D, R, gamma, bandwidth, LX, LY, p):
    b = cuda.blockIdx.y  # batch in Y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    N = LX[b]
    M = LY[b]

    # diagonal bounds in unpadded coordinates
    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)

    diag_len = i_max - i_min + 1
    if diag_len <= 0 or t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    # bandwidth pruning (in padded coords uses ip/jp, but difference same)
    if bandwidth > 0 and abs(ip - jp) > bandwidth:
        return

    inv_gamma = 1.0 / gamma

    r0 = -R[b, ip - 1, jp - 1] * inv_gamma
    r1 = -R[b, ip - 1, jp]     * inv_gamma
    r2 = -R[b, ip,     jp - 1] * inv_gamma

    rmax = r0
    if r1 > rmax: rmax = r1
    if r2 > rmax: rmax = r2

    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
    softmin = -gamma * (math.log(rsum) + rmax)

    R[b, ip, jp] = D[b, i, j] + softmin


@cuda.jit
def dtw_forward_diag_cuda(D, R, bandwidth, N, M, p):
    """D-based forward for *hard* DTW (min recurrence, no gamma). Forward-only."""
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)

    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    if bandwidth > 0 and abs(ip - jp) > bandwidth:
        return

    r0 = R[b, ip - 1, jp - 1]
    r1 = R[b, ip - 1, jp]
    r2 = R[b, ip,     jp - 1]

    m = r0
    if r1 < m: m = r1
    if r2 < m: m = r2

    R[b, ip, jp] = D[b, i, j] + m



@cuda.jit(device=True, inline=True)
def _logsumexp3(a, b, c):
    m = a
    if b > m: m = b
    if c > m: m = c
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.exp(a - m) + math.exp(b - m) + math.exp(c - m))


@cuda.jit
def softdtw_backward_log_cuda(D, R, inv_gamma, bandwidth, LX, LY, n_passes, logE):
    """
    D: (B, N+2, M+2) padded, zeroed outside each sample's valid region
    R: (B, N+2, M+2) padded (with per-sample boundary conditions already set)
    logE: (B, N+2, M+2) padded, initialized to -inf with per-sample seed
          logE[b, LX[b]+1, LY[b]+1] = 0
    """
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    max_i = LX[k]
    max_j = LY[k]

    I = tid

    for p in range(n_passes):
        rev_p = n_passes - p - 1
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        if I + J == rev_p and (I < max_i and J < max_j):

            # Demote unreachable / out-of-band cells (R == +inf) to -inf *before*
            # the bandwidth prune, so in-band neighbours reading them contribute
            # exp(-inf)=0 rather than (-inf)+(+inf)=NaN in the log-space sum.
            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # pruning
            if not (abs(i - j) > bandwidth > 0):

                Rij = R[k, i, j]

                # log transition weights (no exp here!)
                la = (R[k, i + 1, j]     - Rij - D[k, i + 1, j])     * inv_gamma
                lb = (R[k, i, j + 1]     - Rij - D[k, i, j + 1])     * inv_gamma
                lc = (R[k, i + 1, j + 1] - Rij - D[k, i + 1, j + 1]) * inv_gamma

                t1 = logE[k, i + 1, j]     + la
                t2 = logE[k, i, j + 1]     + lb
                t3 = logE[k, i + 1, j + 1] + lc

                logE[k, i, j] = _logsumexp3(t1, t2, t3)

        cuda.syncthreads()

@cuda.jit
def softdtw_backward_log_diag_cuda(Dp, R, logE, inv_gamma, bandwidth, LX, LY, p):
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    N = LX[b]
    M = LY[b]

    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if diag_len <= 0 or t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    # Demote unreachable / out-of-band cells (R == +inf) to -inf *before* the
    # bandwidth prune (see softdtw_backward_log_cuda) to avoid (-inf)+(+inf)=NaN
    # in the log-space accumulation of in-band neighbours.
    if math.isinf(R[b, ip, jp]):
        R[b, ip, jp] = -math.inf

    # pruning
    if bandwidth > 0 and abs(i - j) > bandwidth:
        return

    Rij = R[b, ip, jp]

    la = (R[b, ip + 1, jp]     - Rij - Dp[b, ip + 1, jp])     * inv_gamma
    lb = (R[b, ip, jp + 1]     - Rij - Dp[b, ip, jp + 1])     * inv_gamma
    lc = (R[b, ip + 1, jp + 1] - Rij - Dp[b, ip + 1, jp + 1]) * inv_gamma

    t1 = logE[b, ip + 1, jp]     + la
    t2 = logE[b, ip, jp + 1]     + lb
    t3 = logE[b, ip + 1, jp + 1] + lc

    logE[b, ip, jp] = _logsumexp3(t1, t2, t3)
