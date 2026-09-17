# Changelog

All notable changes to this project are documented here.

## [Unreleased]

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
