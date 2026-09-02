#!/usr/bin/env python3
"""
Training on variable-length spectrograms with padded batches.

Real spectrogram batches (ASR, TTS, speech/audio time series in general)
rarely share one frame count: utterances have different durations, so a
batch is built by padding every sample up to the longest one in it. This
example trains a small per-frame denoiser on synthetic variable-length
spectrograms using SoftDTW(lens_x=..., lens_y=...), so padding frames never
enter the alignment or leak gradient -- numerically identical to looping
over unpadded per-sample calls, but batched on the GPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from softdtw_cuda import SoftDTW


def generate_variable_length_spectrograms(
    num_sequences: int = 128,
    n_mels: int = 20,
    min_frames: int = 40,
    max_frames: int = 120,
    noise_std: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic 'spectrograms': smooth per-mel-bin signals of random length.

    Returns padded clean targets (B, max_frames, n_mels), padded noisy
    inputs of the same shape, and the true per-sample frame counts (B,).
    Padding frames (beyond each sample's length) are left as zero.
    """
    lengths = torch.randint(min_frames, max_frames + 1, (num_sequences,))
    clean = torch.zeros(num_sequences, max_frames, n_mels)
    noisy = torch.zeros(num_sequences, max_frames, n_mels)

    for b in range(num_sequences):
        T = lengths[b].item()
        t = torch.linspace(0, 4 * torch.pi, T).unsqueeze(1)  # (T, 1)
        freqs = torch.rand(n_mels) * 2 + 0.5  # (n_mels,)
        phases = torch.rand(n_mels) * 2 * torch.pi
        signal = torch.sin(t * freqs + phases)  # (T, n_mels)
        clean[b, :T] = signal
        noisy[b, :T] = signal + noise_std * torch.randn(T, n_mels)

    return clean, noisy, lengths


class FrameDenoiser(nn.Module):
    """Small per-frame MLP: denoises each spectrogram frame independently."""

    def __init__(self, n_mels: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_mels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_mels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_mels) -> (B, T, n_mels), applied frame-wise."""
        return self.net(x)


def main():
    print("Variable-Length Spectrogram Training with SoftDTW")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    n_mels = 20
    clean, noisy, lengths = generate_variable_length_spectrograms(
        num_sequences=128, n_mels=n_mels, min_frames=40, max_frames=120,
    )
    print(
        f"Batch: {tuple(clean.shape)}, "
        f"lengths range [{lengths.min().item()}, {lengths.max().item()}]\n"
    )

    clean, noisy, lengths = clean.to(device), noisy.to(device), lengths.to(device)

    model = FrameDenoiser(n_mels=n_mels).to(device)
    loss_fn = SoftDTW(gamma=0.5, fused=(device == "cuda"))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training...")
    for epoch in range(200):
        model.train()
        pred = model(noisy)
        # lens_x=lens_y=lengths: padding frames never enter the DP recurrence
        # and receive exactly-zero gradient, so this padded-batch call is
        # numerically identical to a Python loop over per-sample unpadded
        # calls -- just parallel across the batch on the GPU.
        loss = loss_fn(pred, clean, lens_x=lengths, lens_y=lengths).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1:3d} | SoftDTW loss: {loss.item():.4f}")

    print("\nVerifying padded-batch == per-sample-loop equivalence...")
    model.eval()
    with torch.no_grad():
        pred = model(noisy)
        batched = loss_fn(pred, clean, lens_x=lengths, lens_y=lengths)

        per_sample = []
        for b in range(clean.shape[0]):
            T = lengths[b].item()
            per_sample.append(loss_fn(pred[b : b + 1, :T], clean[b : b + 1, :T]).squeeze(0))
        per_sample = torch.stack(per_sample)

    max_err = (batched - per_sample).abs().max().item()
    print(f"Max |batched - per-sample| = {max_err:.2e} (should be ~0)")


if __name__ == "__main__":
    main()
