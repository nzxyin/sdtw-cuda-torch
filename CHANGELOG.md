# Changelog

All notable changes to this project are documented here.

## [Unreleased]

## [0.3.0] - 2026-09-17

### Added
- Merged upstream `BGU-CS-VIL/sdtw-cuda-torch`'s Phase-1 public update (commit `b0dcc94`,
  PR #1) — see issue #5:
  - Exact hard-DTW: `DTW` module + functional `dtw()`, forward-only (no autograd graph),
    with a three-buffer anti-diagonal streaming kernel (`dtw_forward_diag_stream_sqeuclid_cuda`)
    giving `O(B·min(N,M))` peak memory, plus a D-based tiled path (`dtw_forward_cuda`) for
    non-squared-Euclidean distances.
  - Classic DBA barycenter (`dtw_barycenter`, Petitjean et al. 2011) alongside the existing
    differentiable `softdtw_barycenter`.
  - New benchmarks (`bench/bench_dtw_sdtw.py`, `bench/bench_barycenter_analysis.py`) and tests
    (`test_dtw.py`, `test_dba.py`, `test_gradcheck.py`, `test_reference_agreement.py`).
  - README reworked with upstream's memory-first framing, anti-diagonal streaming figure, and
    mode-selection guide, while keeping this fork's own install/compatibility-matrix/
    numba-cuda-mlir sections (upstream's generic install instructions and its stale
    `CUDA toolkit <= 12.6` constraint — the limitation this fork's numba-cuda-mlir migration
    exists to escape, see `0.2.0` below — were not carried over). Also corrected a README
    benchmark claim: an old `67×` fused-mode runtime speedup figure was upstream's own since-
    retracted near-OOM measurement artifact; the merged table reports memory only, per upstream's
    revision.
  - `requirements.txt`/`environment.yml` (both new from upstream) were adjusted to depend on
    `numba-cuda-mlir` rather than plain `numba` for the CUDA path, consistent with `pyproject.toml`
    and issue #2.
  - The exact-DTW/DBA path does **not** yet support this fork's `lens_x`/`lens_y` variable-length
    batching (upstream's `DTW.forward` takes no length args, and `dtw_forward_cuda_fused_sqeuclid`
    swaps X/Y by whole-tensor `N`/`M` in a way per-sample lengths would break) — tracked as a
    separate fast-follow, see issue #6.
  - The Sakoe-Chiba NaN-gradient fix upstream shipped in this same commit was **not** re-ported
    here: it's the same fix already ported by hand below (`c6d3f70`), confirmed via an
    almost-identical regression test on both sides.

### Fixed
- CUDA backward kernels (`softdtw_backward_log_diag_sqeuclid_cuda`, `softdtw_backward_log_cuda`,
  `softdtw_backward_log_diag_cuda`) could produce NaN gradients when a Sakoe-Chiba `bandwidth`
  was set. A band-pruned cell's `R` value stayed at its `+inf` forward-pass init instead of being
  demoted to `-inf`, so an unpruned neighbor reading it raw produced `-inf + inf = NaN` in the
  log-space backward sum. Fixed by demoting `R` to `-inf` in place *before* the bandwidth prune
  check in each kernel, matching the pattern the CPU reference already used. Confirmed present in
  upstream `BGU-CS-VIL/sdtw-cuda-torch` too (fixed there in commit `b0dcc941`, after this fork's
  divergence point); this change ports that fix. Added
  `softdtw_cuda/tests/test_cuda_bandwidth_regression.py` as a regression test. See issue #1.

## [0.2.0] - 2026-09-02
### Changed
- Migrated the CUDA backend from `numba`/`numba-cuda` to `numba-cuda-mlir`, fixing
  incompatibility with Numba >= 0.66 and modern PyTorch/Python versions.
- Documented fork status, the Numba/Python/PyTorch/CUDA compatibility matrix, and variable-length
  padding usage in the README.
- Bumped version to 0.2.0.

### Added
- Per-sample length support (`lens_x`/`lens_y`) for padded variable-length batches, for training
  on spectrogram/ASR-TTS style data with per-sample sequence lengths.

## [0.1.0] - 2026-02-19
- Forked from `BGU-CS-VIL/sdtw-cuda-torch` at its initial release.
