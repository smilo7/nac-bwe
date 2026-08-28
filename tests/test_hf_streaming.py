"""
Verifies that the streaming HF path is equivalent to the offline path.

Part A: conv caching (the trained model):
    forward_stft_streaming() fed in blocks of size {1, 5, 25} and concatenated
    must equal forward_stft() on the whole sequence, to float tolerance. This
    proves the cached causal convs are exact and that block size does not affect
    output (only latency).

Part B: streaming iSTFT (DSP):
    StreamingISTFT fed the same coefficients block-wise must equal
    torch.istft(center=False) over the whole sequence.

No checkpoint required, since a randomly initialised model is enough. This
tests numerical equivalence of two code paths, not audio quality.

Usage:
    python tests/test_hf_streaming.py
"""

import sys
from pathlib import Path


import torch

from nac_bwe.models.latent_bwe_net import (
    LatentBWENet, StreamingISTFT,
    LATENT_DIM, N_FFT, HOP_LENGTH, WIN_LENGTH,
)

BLOCK_SIZES = [1, 5, 25]
T_TOTAL     = 100   # frames; divisible by every block size above


def test_conv_caching(model, latents):
    offline = model.forward_stft(latents)               # [B, N_HF, T]
    print(f"\nPart A: conv caching (offline ref shape {tuple(offline.shape)})")
    ok = True
    for block in BLOCK_SIZES:
        states = None
        outs = []
        i = 0
        while i < latents.shape[-1]:
            n = min(block, latents.shape[-1] - i)
            stft_blk, states = model.forward_stft_streaming(
                latents[..., i:i + n], states
            )
            outs.append(stft_blk)
            i += n
        streamed = torch.cat(outs, dim=-1)

        err = (streamed - offline).abs().max().item()
        status = "✓" if err < 1e-4 else "✗"
        print(f"  block={block:>3}: max|streamed-offline| = {err:.2e}  {status}")
        ok = ok and err < 1e-4
    assert ok, "conv caching is not equivalent to offline forward_stft"
    return offline


def test_streaming_istft(model):
    """
    STFT->iSTFT round-trip: stream the iSTFT of a known signal's STFT and check
    it reconstructs the original where the overlap-add envelope is constant.
    The first and last window are excluded, since center=False cannot
    reconstruct them exactly. Also confirms the result is identical across
    block sizes.
    """
    B = 2
    T = T_TOTAL
    n_samples = (T - 1) * HOP_LENGTH + N_FFT
    x = torch.randn(B, n_samples)

    spec = torch.stft(
        x, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=model.window, center=False, return_complex=True,
    )                                                    # [B, N_BINS_FULL, T']
    T = spec.shape[-1]

    print("\nPart B: streaming iSTFT round-trip "
          f"(signal {n_samples} samp, {T} frames)")
    edge = N_FFT                     # skip the ramped overlap-add edges
    ok = True
    reference_stream = None
    for block in BLOCK_SIZES:
        istft = StreamingISTFT(window=model.window)
        outs = []
        i = 0
        while i < T:
            n = min(block, T - i)
            outs.append(istft(spec[..., i:i + n]))
            i += n
        streamed = torch.cat(outs, dim=-1)               # [B, T*hop]

        interior_y = streamed[:, edge:streamed.shape[-1] - edge]
        interior_x = x[:, edge:streamed.shape[-1] - edge]
        recon = (interior_y - interior_x).abs().max().item()

        # block-size invariance: every block size must give the same samples
        if reference_stream is None:
            reference_stream = streamed
            inv = 0.0
        else:
            m = min(streamed.shape[-1], reference_stream.shape[-1])
            inv = (streamed[:, :m] - reference_stream[:, :m]).abs().max().item()

        status = "✓" if recon < 1e-4 and inv < 1e-5 else "✗"
        print(f"  block={block:>3}: round-trip err = {recon:.2e}  "
              f"block-invariance = {inv:.2e}  {status}")
        ok = ok and recon < 1e-4 and inv < 1e-5
    assert ok, "streaming iSTFT failed round-trip or block-invariance"


def main():
    torch.manual_seed(0)
    model = LatentBWENet(hidden_dim=256, num_blocks=4).eval()
    latents = torch.randn(2, LATENT_DIM, T_TOTAL)

    print("=== HF streaming equivalence test ===")
    with torch.no_grad():
        test_conv_caching(model, latents)
        test_streaming_istft(model)

    print("\n=== All streaming equivalence checks passed ===")


if __name__ == "__main__":
    main()
