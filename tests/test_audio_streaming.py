"""
Verifies the audio model (AudioBWENet, center=False) streams exactly.

Unlike the latent model, the audio model has an analysis STFT on the input, so
streaming must carry STFT overlap state across blocks in addition to the conv
caches. This test feeds 48 kHz audio in blocks of several sizes and checks:

Part A: HF STFT equivalence:
    forward_stft_streaming() over blocks (analysis STFT state + conv caches),
    concatenated, equals the offline analysis+backbone (_features_from_48k ->
    _backbone) on the whole signal. Also checks block-size invariance.

Part B: end-to-end waveform:
    streaming HF STFT -> StreamingISTFT, concatenated, matches the offline
    forward synthesis (causal_istft) where the overlap-add envelope is
    constant.

Resampling 24->48 kHz is external to the model's streaming path (handled by a
streaming resampler, as the LF path is), so the test operates on 48 kHz blocks.

No checkpoint needed, this tests numerical equivalence of two code paths.

Usage:
    python tests/test_audio_streaming.py
"""

import sys
from pathlib import Path


import torch

from nac_bwe.models.latent_bwe_net import StreamingISTFT, HF_BIN_START, N_FFT, HOP_LENGTH, WIN_LENGTH
from nac_bwe.models.audio_bwe_net import AudioBWENet

BLOCK_FRAMES = [1, 5, 25]   # block sizes in *frames*; samples = frames * hop
T_TOTAL      = 100          # total frames; divisible by every block size above


def _offline_hf(model, x48):
    feats = model._features_from_48k(x48)
    real, imag = model._backbone(feats)
    return torch.complex(real, imag)


def test_analysis_streaming(model, x48):
    offline = _offline_hf(model, x48)                    # [B, N_HF, T]
    print(f"\nPart A: analysis+conv streaming (offline ref {tuple(offline.shape)})")
    ok = True
    for bf in BLOCK_FRAMES:
        block = bf * HOP_LENGTH
        states = None
        outs = []
        i = 0
        while i < x48.shape[-1]:
            n = min(block, x48.shape[-1] - i)
            hf_blk, states = model.forward_stft_streaming(x48[:, i:i + n], states)
            if hf_blk.shape[-1] > 0:
                outs.append(hf_blk)
            i += n
        streamed = torch.cat(outs, dim=-1)
        m = min(streamed.shape[-1], offline.shape[-1])
        err = (streamed[..., :m] - offline[..., :m]).abs().max().item()
        frames_match = streamed.shape[-1] == offline.shape[-1]
        status = "✓" if err < 1e-4 and frames_match else "✗"
        print(f"  block={bf:>3} frames: max|streamed-offline| = {err:.2e} "
              f"| frames {streamed.shape[-1]}=={offline.shape[-1]} {status}")
        ok = ok and err < 1e-4 and frames_match
    assert ok, "audio analysis streaming is not equivalent to offline forward_stft"
    return offline


def _full_spec(hf):
    B, _, T = hf.shape
    lf = torch.zeros(B, HF_BIN_START, T, dtype=hf.dtype)
    return torch.cat([lf, hf], dim=1)


def test_waveform_streaming(model, x48):
    offline_hf = _offline_hf(model, x48)
    # offline synthesis via the model's own causal iSTFT
    from nac_bwe.models.latent_bwe_net import causal_istft
    offline_audio = causal_istft(_full_spec(offline_hf), model.window, N_FFT, HOP_LENGTH)

    print("\nPart B: end-to-end streaming waveform")
    edge = N_FFT
    ok = True
    for bf in BLOCK_FRAMES:
        block = bf * HOP_LENGTH
        states = None
        istft = StreamingISTFT(window=model.window)
        outs = []
        i = 0
        while i < x48.shape[-1]:
            n = min(block, x48.shape[-1] - i)
            hf_blk, states = model.forward_stft_streaming(x48[:, i:i + n], states)
            if hf_blk.shape[-1] > 0:
                outs.append(istft(_full_spec(hf_blk)))
            i += n
        streamed = torch.cat(outs, dim=-1)
        m = min(streamed.shape[-1], offline_audio.shape[-1])
        interior = slice(edge, m - edge)
        err = (streamed[:, interior] - offline_audio[:, interior]).abs().max().item()
        status = "✓" if err < 1e-4 else "✗"
        print(f"  block={bf:>3} frames: max|streamed-offline audio| (interior) "
              f"= {err:.2e}  {status}")
        ok = ok and err < 1e-4
    assert ok, "audio waveform streaming is not equivalent to offline synthesis"


def main():
    torch.manual_seed(0)
    model = AudioBWENet(hidden_dim=256, num_blocks=4, center=False).eval()
    n_samples = (T_TOTAL - 1) * HOP_LENGTH + N_FFT
    x48 = torch.randn(2, n_samples)

    print("=== Audio model streaming equivalence test (center=False) ===")
    with torch.no_grad():
        test_analysis_streaming(model, x48)
        test_waveform_streaming(model, x48)

    print("\n=== All audio streaming equivalence checks passed ===")


if __name__ == "__main__":
    main()
