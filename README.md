# SoftDTW-CUDA (PyTorch + Numba)

[![Tests](https://github.com/nzxyin/sdtw-cuda-torch/actions/workflows/test.yml/badge.svg)](https://github.com/nzxyin/sdtw-cuda-torch/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

> This is a fork of [BGU-CS-VIL/sdtw-cuda-torch](https://github.com/BGU-CS-VIL/sdtw-cuda-torch)
> maintained by [@nzxyin](https://github.com/nzxyin). Changes from upstream: **variable-length
> padded batch support** (`lens_x`/`lens_y`) for real-world training data such as spectrograms
> (see [Variable-Length Sequences](#variable-length-sequences-spectrograms-asrtts)), and a CUDA
> backend migration from `numba.cuda`/`numba-cuda` to
> [`numba-cuda-mlir`](https://github.com/NVIDIA/numba-cuda-mlir), because plain `numba.cuda`
> breaks on Numba 0.66+ for this repo's kernels. Validated on Python 3.11 through 3.14, see the
> [Compatibility Matrix](#compatibility-matrix).

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
| **vs. Maghoumi memory** | N/A | 96.9% less | 98.0% less |
| **vs. Maghoumi speed** | N/A | **67× faster** | 6.5× faster |

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

* Python 3.11+
* NVIDIA GPU, Compute Capability 7.0+ (Volta or newer), with a matching driver
* PyTorch with CUDA support (see below)
* Numba 0.60+ for the CPU fallback path only. CUDA kernels use
  [`numba-cuda-mlir`](https://github.com/NVIDIA/numba-cuda-mlir) instead of `numba.cuda`; see
  its [own requirements](https://github.com/NVIDIA/numba-cuda-mlir#installation-requirements)
  for full driver and toolkit details, and the Compatibility Matrix below for why.

### Compatibility Matrix

This fork's CUDA kernels use [`numba-cuda-mlir`](https://github.com/NVIDIA/numba-cuda-mlir)
instead of `numba.cuda`/`numba-cuda` (see "Why `numba-cuda-mlir`?" below). Plain `numba` is
still a dependency, used only for the CPU fallback path. Every row below was run through this
repo's own 45-item pytest suite on a real GPU, not inferred from release notes or vendor claims.

Each Python version is paired with the current latest Numba and PyTorch release, confirmed via
each project's own wheel index to actually have a build for that Python version:

| Python | Numba | PyTorch | numba-cuda-mlir | Result |
|---|---|---|---|---|
| **3.11** | 0.67.0 | 2.14.0+cu130 | 0.5.1 | 40 passed, 5 skipped |
| **3.12** | 0.67.0 | 2.14.0+cu130 | 0.5.1 | 40 passed, 5 skipped |
| **3.13** | 0.67.0 | 2.14.0+cu130 | 0.5.1 | 40 passed, 5 skipped |
| **3.14** | 0.67.0 | 2.14.0+cu130 | 0.5.1 | 40 passed, 5 skipped |

Python 3.10 is no longer supported (`numba-cuda-mlir` requires 3.11+). PyTorch 2.14.0's `cu132`
build (CUDA 13.2) also passes the full suite.

Also validated, from before the `numba-cuda-mlir` migration: Numba 0.65.1 with the old
`numba.cuda` in-tree target, Python 3.13, PyTorch 2.13.0+cu130, CUDA 13.0: 40/40 passed.

Compatibility beyond the combinations above is not guaranteed.

### Why `numba-cuda-mlir`?

This fork originally used `numba.cuda` directly. Testing Numba 0.67.0 (the latest release)
surfaced three separate upstream bugs that break this repo's kernels on Numba 0.66+:

1. The in-tree CUDA target hits [numba/numba#10753](https://github.com/numba/numba/issues/10753),
   an open regression in two-argument `max()`/`min()`, which this repo's kernels use throughout.
2. Installing `numba-cuda` (the usual fix) instead hits
   [NVIDIA/numba-cuda#907](https://github.com/NVIDIA/numba-cuda/issues/907): it still
   references `np.row_stack`, removed in NumPy 2.5.
3. Working around that with `numpy<2.5` then fails on a missing `libdevice` file, which
   `numba-cuda`'s pathfinder cannot locate in this setup.

`numba-cuda-mlir` avoids all three: it does not depend on `numba` or `numba-cuda` at all. The
migration was two import lines (`from numba import cuda` to `from numba_cuda_mlir import
cuda`), a Python floor bump to 3.11, and a new dependency. Full details on
[#2](https://github.com/nzxyin/sdtw-cuda-torch/issues/2).

### Step 1: Install PyTorch with CUDA

PyTorch must be installed **before** this package, with the correct CUDA variant for your system. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the right command. Example for CUDA 13.0 (the combination validated for this fork):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```


### Step 2: Install this package with the matching CUDA extra

This package's CUDA kernels depend on
[`numba-cuda-mlir`](https://github.com/NVIDIA/numba-cuda-mlir), which needs to know whether to
pull CUDA 12.x or 13.x toolkit components. Pick the extra matching the CUDA variant you
installed PyTorch with in Step 1:

```bash
git clone https://github.com/nzxyin/sdtw-cuda-torch
pip install -e "sdtw-cuda-torch[cu13]"   # or "[cu12]" for CUDA 12.x
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
padding frames never enter the alignment: the DP recurrence stops at each
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
  batch-1 calls; batch parallelism is preserved on the GPU
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

Speech and audio batches almost never share one frame count. Spectrograms,
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
* [Sleepwalking/pytorch-softdtw](https://github.com/Sleepwalking/pytorch-softdtw): PyTorch GPU implementation
* [Maghoumi/pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda): CUDA implementation (motivation for memory and stability improvements)
* [keonlee9420/Soft-DTW-Loss](https://github.com/keonlee9420/Soft-DTW-Loss): additional PyTorch reference implementation

---

## License

MIT

