# SoftDTW-CUDA (PyTorch + Numba)

> This is a fork of [BGU-CS-VIL/sdtw-cuda-torch](https://github.com/BGU-CS-VIL/sdtw-cuda-torch)
> maintained by [@nzxyin](https://github.com/nzxyin), adding **variable-length padded batch
> support** (`lens_x`/`lens_y`) for real-world training data such as spectrograms — see
> [Variable-Length Sequences](#variable-length-sequences-spectrograms-asrtts) below — plus an
> updated [compatibility matrix](#compatibility-matrix) for current Numba/Python/PyTorch/CUDA
> versions.

A **GPU-accelerated, memory-efficient, and numerically stable** implementation of
**Soft Dynamic Time Warping (SoftDTW)** for PyTorch.

This package is designed primarily as a **loss function for training neural networks**, with additional support for **time series averaging** (barycenters). Strong emphasis on:

* 🔥 **GPU memory efficiency**
* 📏 **Long sequence support** (lengths > 1024)
* 🧮 **Numerical stability** (log-space backward)
* ⚡ **Optional fused distance computation** (no `(B,N,M)` tensor)
* 🧩 **Variable-length padded batches** (per-sample `lens_x`/`lens_y`, exact-zero padding gradients)
* 📊 **Time series averaging** (SoftDTW barycenters)

---

## Why This Implementation?

Compared to the popular CUDA implementation by [Maghoumi et al.](https://github.com/mblondel/soft-dtw), this repo fixes critical limitations for real training workloads:

### Feature Comparison

| Feature | Maghoumi CUDA | This Repo |
|---|---|---|
| CUDA forward | ✅ | ✅ |
| CUDA backward | ⚠️ numerically unstable | ✅ log-space stable |
| Max sequence length | ❌ ≤ 1024 | ✅ unbounded (tiled) |
| Memory-efficient fused mode | ❌ | ✅ |
| Variable-length padded batches | ❌ (fixed length only) | ✅ per-sample `lens_x`/`lens_y` |

### Key Benchmark (B=32, N=512, D=64)

| | Maghoumi | Ours (Unfused) | Ours (Fused) |
|---|---|---|---|
| **Peak Memory** | 8,256 MB | 257 MB | 161 MB |
| **Runtime** | 2,791 ms | **42 ms** | 430 ms |
| **vs. Maghoumi memory** | — | 96.9% less | 98.0% less |
| **vs. Maghoumi speed** | — | **67× faster** | 6.5× faster |

### When to Use Each Mode

| Scenario | Mode | Reason |
|---|---|---|
| Large D, big batches | Fused | ~98% memory savings |
| Speed-critical / inference | Unfused | 10–67× faster than Fused |
| N > 1024 | Both modes | Both use tiled anti-diagonal execution; fused saves more memory |
| Small D (D=1–4) | Unfused | Fused savings are small (~30%) |

### Limitations

* Fused mode requires **CUDA** and **squared Euclidean distance only**
* Fused is 10–25× slower in runtime than unfused (memory/compute trade-off)
* CPU implementation is for testing only, not performance

> Full benchmark tables and analysis: [bench/README.md](bench/README.md)

---

## Installation

### Requirements

* Python ≥ 3.10
* NVIDIA GPU with CUDA Toolkit 13.x — compute capability ≥ 7.5 (Turing or newer; CUDA 13
  dropped Maxwell/Pascal/Volta support), driver ≥ 580 (≥ 595.45.04 for CUDA 13.2+ on Linux)
* PyTorch with CUDA support (see below)
* Numba ≥ 0.60, **but not ≥ 0.66 as of this writing** — this package's kernels use
  `numba.cuda`, and Numba ≥ 0.66 is confirmed broken for this repo via every path tried so far
  (in-tree target and `numba-cuda 0.30.4` both fail on real GPU test runs, for three different
  reasons — see the Compatibility Matrix below and [#2](https://github.com/nzxyin/sdtw-cuda-torch/issues/2)).
  `numba-cuda-mlir` is a confirmed-working alternative but requires a small code change this fork
  hasn't made yet (also detailed below)

### Compatibility Matrix

Every row below was actually run through this repo's own 45-item pytest suite on a real GPU
(L40S/Ada) via `sbatch` — not inferred from upstream release notes. Each combination changes
exactly one thing from the validated baseline (Numba 0.65.1 / Python 3.13.9 / PyTorch
2.13.0+cu130) so a failure can be attributed to a single cause.

| Combination | Result | Detail |
|---|---|---|
| Baseline: Numba 0.65.1, Python 3.13.9, PyTorch 2.13.0+cu130, CUDA 13.0 | ✅ **40/40 passed** | 2026-08-31, real L40S GPUs (SLURM jobs 10273367/10273744/10273441) |
| Python → **3.14** (latest stable) | ✅ **40 passed, 5 skipped** | 2026-09-02, job 10291991 |
| PyTorch → **2.14.0+cu130** (released 2026-09-02) | ✅ **40 passed, 5 skipped** | job 10291991 |
| PyTorch → **2.14.0+cu132** (CUDA 13.2 wheel) | ✅ **40 passed, 5 skipped** | job 10291991 |
| Numba → **0.67.0** (current latest), in-tree CUDA target | ❌ **19 failed, 21 passed, 5 skipped** | job 10291991 — root cause 1 below |
| Everything above at once (Numba 0.67.0 + Python 3.14 + PyTorch 2.14.0+cu132) | ❌ **19 failed, 21 passed, 5 skipped** — identical failure set to the Numba-alone row | job 10291991 — confirms Numba 0.67.0's in-tree target is the sole cause, not Python or PyTorch |
| Numba 0.67.0 + `numba-cuda==0.30.4` (explicit out-of-tree package) | ❌ **19 failed** (different error) | job 10292057 — root cause 2 below |
| Numba 0.67.0 + `numba-cuda==0.30.4` + `numpy<2.5` | ❌ **19 failed** (yet another different error) | job 10292066 — root cause 3 below |
| Numba 0.67.0 (core, CPU path only) + **`numba-cuda-mlir==0.5.1`** (CUDA target) | ✅ **40 passed, 5 skipped** | job 10292092 — see "Working path for Numba 0.67.0" below |

**Three independent, confirmed root causes make Numba ≥ 0.66's usual CUDA paths fail today** —
each isolated by actually changing one variable and rerunning the suite, not inferred:

1. **In-tree target (no `numba-cuda` installed):** every failure is `TypeError: Signature
   mismatch: 2 argument types given, but function takes 1 arguments`, and only CUDA-path tests
   fail — CPU-path tests all pass. This repo's kernels (`softdtw_cuda/cuda/kernels.py`) call
   two-argument `max()`/`min()` extensively (e.g. `max(0, p - (M - 1))`, `min(N - 1, p)`), which
   is exactly the pattern broken by [numba/numba#10753](https://github.com/numba/numba/issues/10753)
   — an **open upstream regression** introduced in Numba 0.66.0 (PR #10543 refactored `max`/`min`
   into `*args` overloads), affecting only the in-tree CUDA target.
2. **`numba-cuda==0.30.4` installed (the fix for #1):** avoids the max/min bug, but every test
   now fails with `AttributeError: module 'numpy' has no attribute 'row_stack'`, raised from
   inside `numba-cuda` itself at CUDA-target registry-load time. NumPy fully removed `row_stack`
   in 2.5 (which `uv` resolves by default alongside Numba 0.67.0); `numba-cuda` 0.30.4 hasn't
   caught up — tracked as [NVIDIA/numba-cuda#907](https://github.com/NVIDIA/numba-cuda/issues/907)
   (open since 2026-07-03, with an unmerged one-line patch).
3. **`numba-cuda==0.30.4` + `numpy<2.5` (routing around #2):** gets past module load into actual
   kernel compilation, then fails with `RuntimeError: Missing libdevice file` — `numba-cuda`'s own
   CUDA-toolkit pathfinder can't locate `libdevice.10.bc` in an environment that only has PyTorch's
   bundled NVRTC wheel. Not yet resolved (an explicit `cuda-nvcc`/libdevice-providing package might
   fix it, but wasn't tested — see [#2](https://github.com/nzxyin/sdtw-cuda-torch/issues/2)).

**As of today, there is no known way to get Numba 0.67.0 working via either the in-tree target or
`numba-cuda` for this repo.** Do not upgrade past Numba 0.65.x on either path.

**Working path for Numba 0.67.0: `numba-cuda-mlir`, empirically verified.**
[`numba-cuda-mlir`](https://github.com/NVIDIA/numba-cuda-mlir) is NVIDIA's actively-developed
successor to `numba-cuda` (commits as recent as today, PyPI releases every 2-3 weeks) — an
independent MLIR-based compiler with **no dependency on `numba`/`numba-cuda` at all**, so it
sidesteps all three bugs above. Verified by patching a scratch copy of this repo (changing just
`from numba import cuda` to `from numba_cuda_mlir import cuda` in `kernels.py` and `launcher.py`
— `launcher.py`'s unrelated `from numba import jit, prange` CPU path stays as plain `numba`) and
running the real test suite: **40 passed, 5 skipped**, with Numba 0.67.0 (core), `numba-cuda-mlir`
0.5.1, NumPy 2.5.2, and PyTorch 2.13.0+cu130 (job 10292092) — the only combination in this whole
investigation that gets Numba 0.67.0 working end-to-end. Not yet adopted by this repo — doing so
would mean:
* Bumping this repo's Python floor from `>=3.10` to `>=3.11` (`numba-cuda-mlir`'s own requirement)
* Adding `numba-cuda-mlir` as an explicit dependency (keeping `numba` core for the CPU-only path)
* A more verbose `NumbaPerformanceWarning` on small-grid kernel launches than plain `numba.cuda`
  emits at the same sizes (cosmetic, not a correctness issue)

Full details and reproduction commands: [#2](https://github.com/nzxyin/sdtw-cuda-torch/issues/2).

> ⚠️ Compatibility beyond the validated combinations above is not guaranteed. PyPI PyTorch
> wheels newer than `cu130` (e.g. `cu132`) may also be less mature than `cu130`.

### Step 1 — Install PyTorch with CUDA

PyTorch must be installed **before** this package, with the correct CUDA variant for your system. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the right command. Example for CUDA 13.0 (the combination validated for this fork):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```


### Step 2 — Install this package

```bash
git clone https://github.com/nzxyin/sdtw-cuda-torch
pip install -e sdtw-cuda-torch
```

---

## Usage

### Basic (Unfused)

```python
from softdtw_cuda import SoftDTW

loss_fn = SoftDTW(gamma=1.0)

x = torch.randn(B, N, D, device="cuda", requires_grad=True)
y = torch.randn(B, M, D, device="cuda", requires_grad=True)

loss = loss_fn(x, y).mean()
loss.backward()
```

* Explicit distance computation
* More flexible
* Higher memory usage

---

### Fused Mode (Recommended for Training)

```python
loss_fn = SoftDTW(
    gamma=1.0,
    dist="sqeuclidean",
    fused=True
)

loss = loss_fn(x, y).mean()
loss.backward()
```

**Fused mode**

* No distance tensor
* Much lower GPU memory
* Best choice for large `N`, `D`

---

### Variable-Length Batches (Padding Support)

Real batches rarely share one sequence length. Pass per-sample lengths and
padding frames never enter the alignment — the DP recurrence stops at each
sample's own true length, per-sample results are read from each sample's own
final DP cell, and **padding frames receive exactly-zero gradients**:

```python
loss_fn = SoftDTW(gamma=1.0)

x = torch.randn(B, N, D, device="cuda", requires_grad=True)  # padded
y = torch.randn(B, M, D, device="cuda")                      # padded
lens_x = torch.tensor([...])  # (B,) true lengths, 1 <= lens_x[b] <= N
lens_y = torch.tensor([...])  # (B,) true lengths, 1 <= lens_y[b] <= M

loss = loss_fn(x, y, lens_x=lens_x, lens_y=lens_y).mean()
loss.backward()
```

* Works in every mode: fused / unfused, CUDA / CPU, `normalize=True` / `False`
* Equivalent to (but much faster than) a Python loop of per-sample sliced
  batch-1 calls — batch parallelism is preserved on the GPU
* With `normalize=True`, the padded dims must still match (`N == M`), but
  per-sample `lens_x[b]`/`lens_y[b]` may differ
* Omitting the lengths keeps the classic fixed-length behavior

---
# Applications
## Forecasting

![Forecasting](https://github.com/nzxyin/sdtw-cuda-torch/blob/main/examples/forecasting_results.png)
Train a simple forecaster using SoftDTW as the loss function:

```python
import torch
from softdtw_cuda import SoftDTW

model = MyForecaster().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = SoftDTW(gamma=1.0, fused=True)

for x_batch, y_batch in dataloader:
    y_pred = model(x_batch.cuda())           # (B, N, D)
    loss = loss_fn(y_pred, y_batch.cuda()).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

See [examples/forecasting_example.py](examples/forecasting_example.py) for a complete working example with sine wave data.



## Variable-Length Sequences (Spectrograms, ASR/TTS)

Speech and audio batches almost never share one frame count — spectrograms,
mel-features, and other frame-rate time series have a different true length
per utterance and get zero-padded to the batch's longest sample. Pass
`lens_x`/`lens_y` so the padding never enters the alignment or the gradient,
which keeps the batched call numerically identical to looping over unpadded
per-sample calls:

```python
from softdtw_cuda import SoftDTW

model = MySpectrogramModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = SoftDTW(gamma=0.5, fused=True)

for specs, targets, lengths in dataloader:  # specs/targets: (B, T_max, n_mels), padded
    pred = model(specs.cuda())              # (B, T_max, n_mels)
    loss = loss_fn(pred, targets.cuda(), lens_x=lengths.cuda(), lens_y=lengths.cuda()).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

See [examples/variable_length_spectrogram_example.py](examples/variable_length_spectrogram_example.py) for a complete working example that trains a denoiser on synthetic variable-length spectrograms and verifies the padded-batch loss exactly matches a per-sample loop.



## Time Series Barycenters (Averaging)
![SoftDTW Barycenter](https://github.com/nzxyin/sdtw-cuda-torch/blob/main/examples/softdtw_barycenter_example.png)

Compute a DTW-space average (barycenter) for a batch of sequences:

```python
from softdtw_cuda import softdtw_barycenter

sequences = torch.randn(10, 100, 3, device="cuda")  # 10 sequences

barycenter = softdtw_barycenter(
    sequences,
    gamma=1.0,
    max_iter=100,
    lr=0.1,
)

print(barycenter.shape)  # (100, 3)
```

**Key options:**

* `gamma`: Regularization strength (higher = smoother)
* `max_iter`: Optimization iterations
* `lr`: Adam learning rate (0.1 default)
* `fused`: Auto-select fused mode (memory vs speed trade-off)
* `early_stopping=True`: Detects convergence, saves ~30-50% iterations

See [BARYCENTERS.md](softdtw_cuda/BARYCENTERS.md) for detailed docs and [examples/barycenter_example.py](examples/barycenter_example.py) for visualization.


---

## Normalization

Supports the common normalized variant:

$$\mathrm{SoftDTW\_norm}(x,y) = \mathrm{SoftDTW}(x,y) - \tfrac{1}{2}\bigl(\mathrm{SoftDTW}(x,x) + \mathrm{SoftDTW}(y,y)\bigr)$$

Enable with:

```python
SoftDTW(normalize=True)
```

⚠️ **Current constraint:** normalization requires equal sequence lengths
`x.shape == y.shape == (B, N, D)`

---

## Notes

* SoftDTW **may return negative values** (expected)
* Squared Euclidean distances are always ≥ 0
* Negativity arises from the soft-min aggregation

---

## Tests

```bash
pytest -v
```

| Test file | What it covers |
|---|---|
| `test_softdtw_small.py` | CPU and CUDA forward/backward, gradient correctness |
| `test_softdtw_long.py` | Sequences longer than 1024 (tiled kernel) |
| `test_softdtw_log_backward.py` | Log-space backward numerical stability |
| `test_fused_sqeuclid.py` | Fused vs unfused equivalence for squared Euclidean |
| `test_sqeuclidean.py` | Distance computation correctness |
| `test_validation.py` | Input validation: gamma, device, empty sequences, shape mismatches |
| `test_lengths.py` | Variable-length padded batches (`lens_x`/`lens_y`): batched-vs-per-sample equivalence, exact-zero padding gradients, tiled-path lengths, gradcheck |

---

## Benchmarking

Full benchmark suite available in `bench/` directory. Key results:

**SoftDTW Loss Function:**
* Memory efficiency: 92-98% reduction vs. Maghoumi et al.
* Supports arbitrary sequence lengths (no 1024 limit)
* Numerically stable via log-space backward pass

**Barycenter Optimization:**
* Early stopping typically saves 30-50% of iterations
* Cosine annealing + gradient clipping ensures stability
* Supports both fused and unfused modes

Run benchmarks with:
```bash
python bench/bench_memory.py
python examples/barycenter_example.py --compare
```

---

## Acknowledgments

**SoftDTW Loss:**
> Cuturi & Blondel,
> *Soft-DTW: a Differentiable Loss Function for Time-Series*, ICML 2017

**Barycenter Implementation:**
> Based on [tslearn](https://github.com/tslearn-team/tslearn) implementation, originally from Cuturi & Blondel (ICML 2017)

**Prior PyTorch/CUDA implementations this work builds on:**
* [Sleepwalking/pytorch-softdtw](https://github.com/Sleepwalking/pytorch-softdtw) — PyTorch GPU implementation
* [Maghoumi/pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda) — CUDA implementation (motivation for memory and stability improvements)
* [keonlee9420/Soft-DTW-Loss](https://github.com/keonlee9420/Soft-DTW-Loss) — additional PyTorch reference implementation

---

## License

MIT

