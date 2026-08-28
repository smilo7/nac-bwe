"""
End-to-end bandwidth extension with a released checkpoint.

Runs the full inference path (encode to EnCodec codes, decode the low band,
generate the high band, sum the two) and checks that the result is what
bandwidth extension is supposed to produce:

  1. the codec's own output really is empty above 12 kHz,
  2. the model puts energy back there,
  3. it does so without disturbing the low band it was given.

Works for both released checkpoints. The default EnCodec low-band backend keeps
this to a single set of pretrained weights. ``--lf vocos`` exercises the
higher-quality path instead, at the cost of a second download.

Needs network access on first run to fetch the EnCodec weights.

Usage:
    python tests/test_end_to_end.py
    python tests/test_end_to_end.py --checkpoint checkpoints/audio_small_gan.pt
    python tests/test_end_to_end.py --audio some_file.wav --save out.wav
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from nac_bwe.checkpoints import load_release
from nac_bwe.codec import EncodecProcessor
from nac_bwe.models.latent_bwe_net import SAMPLE_RATE, make_lf_extractor

ENCODEC_SR   = 24000
BANDWIDTH_ID = 3      # 12 kbps
CUTOFF_HZ    = 12000  # the band EnCodec 24 kHz discards


def band_energy(x: torch.Tensor, lo_hz: float, hi_hz: float, sr: int = SAMPLE_RATE) -> float:
    """Energy of x in [lo_hz, hi_hz), via the magnitude spectrum."""
    spec = torch.fft.rfft(x.flatten())
    freqs = torch.fft.rfftfreq(x.flatten().numel(), d=1.0 / sr)
    band = (freqs >= lo_hz) & (freqs < hi_hz)
    return spec[band].abs().pow(2).sum().item()


def make_test_signal(seconds: float = 2.0) -> torch.Tensor:
    """
    Synthetic 48 kHz stand-in for real audio: a harmonic stack plus noise, both
    rolled off above 8 kHz.

    The roll-off matters. Energy sitting right at 12 kHz lands in the transition
    band of the low-band path's 24->48 kHz resampling filter and leaks above the
    cutoff, which would blunt the "codec output is empty up there" check. Real
    recordings roll off on their own, flat noise does not.
    """
    torch.manual_seed(0)
    n = int(seconds * SAMPLE_RATE)
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE

    harmonics = sum(torch.sin(2 * torch.pi * f * t) / (i + 1)
                    for i, f in enumerate((220.0, 440.0, 880.0, 1760.0, 3520.0)))
    noise = torch.randn(n) * 0.5

    x = harmonics + noise
    spec = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    spec *= torch.exp(-torch.clamp(freqs - 8000.0, min=0.0) / 1500.0)
    x = torch.fft.irfft(spec, n=n) * torch.hann_window(n)

    return (x / x.abs().max() * 0.7).unsqueeze(0)


def load_audio(path: Path, seconds: float) -> torch.Tensor:
    x, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)
    if x.shape[0] > 1:
        x = x.mean(0, keepdim=True)
    return x[:, : int(seconds * SAMPLE_RATE)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/latent_small_gan.pt")
    ap.add_argument("--audio", default=None, help="48 kHz file, default is a synthetic signal")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--lf", choices=["encodec", "vocos"], default="encodec")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", default=None, help="write the extended audio here")
    args = ap.parse_args()

    device = args.device
    print(f"\n=== End-to-end BWE ({Path(args.checkpoint).name}, lf={args.lf}) ===\n")

    model, meta = load_release(args.checkpoint, device=device)
    model_type = meta["model_type"]
    print(f"1. Checkpoint: {model_type} model, epoch {meta['epoch']}, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    x_clean_48k = (load_audio(Path(args.audio), args.seconds) if args.audio
                   else make_test_signal(args.seconds))
    print(f"2. Input: {x_clean_48k.shape[-1] / SAMPLE_RATE:.2f}s at {SAMPLE_RATE} Hz")

    processor = EncodecProcessor(sr=ENCODEC_SR, device=device)
    x_clean_24k = torchaudio.functional.resample(x_clean_48k, SAMPLE_RATE, ENCODEC_SR)
    latents_list, _ = processor.audio_to_latents(x_clean_24k)
    codes, meta_c = processor.latents_to_codes(latents_list, kbps=12.0)
    latents_q = processor.codes_to_latents(codes)
    print(f"3. Encoded: codes {tuple(codes.shape)} -> latents {tuple(latents_q.shape)}")

    if args.lf == "vocos":
        from vocos import Vocos
        lf = make_lf_extractor("vocos",
                               vocos_model=Vocos.from_pretrained("charactr/vocos-encodec-24khz"),
                               device=device)
    else:
        lf = make_lf_extractor("encodec", processor=processor, device=device)

    # Low band: frozen decoder, band-limited to <12 kHz by construction.
    lf_audio = lf.decode_lf_audio(codes.to(device), bandwidth_id=BANDWIDTH_ID)

    # High band: the audio model conditions on the codec's 24 kHz reconstruction,
    # the latent model on the quantized latents. Both emit 12-24 kHz only.
    if model_type == "audio":
        codec_24k = processor.decode_latents_audio(latents_q, meta_c)
        if codec_24k.dim() == 3:
            codec_24k = codec_24k.squeeze(0)
        model_in = codec_24k[:, : x_clean_24k.shape[-1]].to(device)
    else:
        model_in = latents_q.to(device)

    with torch.no_grad():
        hf_audio = model(model_in).squeeze(1)

    # The two bands do not overlap, so summation is exact.
    extended = lf.combine_time_domain(lf_audio, hf_audio).cpu()
    print(f"4. LF {tuple(lf_audio.shape)} + HF {tuple(hf_audio.shape)} "
          f"-> {tuple(extended.shape)}")

    # ---- checks -----------------------------------------------------------
    n = min(lf_audio.shape[-1], extended.shape[-1])
    lf_only = lf_audio[..., :n].cpu()
    ext = extended[..., :n]

    lf_hf_band = band_energy(lf_only, CUTOFF_HZ, SAMPLE_RATE / 2)
    ext_hf_band = band_energy(ext, CUTOFF_HZ, SAMPLE_RATE / 2)
    lf_low_band = band_energy(lf_only, 0, CUTOFF_HZ)
    ext_low_band = band_energy(ext, 0, CUTOFF_HZ)

    print("\n5. Checks")
    ok = True

    # The codec output must genuinely be missing the high band, or there is
    # nothing to extend and the rest of the test proves nothing.
    ratio_codec = lf_hf_band / max(lf_low_band, 1e-12)
    passed = ratio_codec < 1e-3
    ok &= passed
    print(f"   codec HF/LF energy      = {ratio_codec:.2e}  "
          f"{'✓ band is empty' if passed else '✗ expected < 1e-3'}")

    # The model must fill it. Measured across both checkpoints on speech, music
    # and texture recordings the gain runs +11 to +36 dB, so +6 dB is a floor
    # that still means real content rather than a threshold fitted to one clip.
    gain_db = 10 * torch.log10(torch.tensor(ext_hf_band / max(lf_hf_band, 1e-12))).item()
    passed = gain_db > 6
    ok &= passed
    print(f"   HF energy gained        = {gain_db:+.1f} dB  "
          f"{'✓ model filled the band' if passed else '✗ expected > +6 dB'}")

    # And must leave the low band it was handed alone: the HF branch synthesises
    # bins 641+ only, so anything below 12 kHz should pass through untouched.
    low_drift_db = abs(10 * torch.log10(
        torch.tensor(ext_low_band / max(lf_low_band, 1e-12))).item())
    passed = low_drift_db < 0.5
    ok &= passed
    print(f"   LF band drift           = {low_drift_db:.3f} dB  "
          f"{'✓ low band preserved' if passed else '✗ expected < 0.5 dB'}")

    passed = torch.isfinite(ext).all().item() and ext.abs().max().item() < 10.0
    ok &= passed
    print(f"   output peak             = {ext.abs().max().item():.3f}  "
          f"{'✓ finite and sane' if passed else '✗ non-finite or clipping hard'}")

    if args.save:
        torchaudio.save(args.save, ext, SAMPLE_RATE)
        print(f"\n   wrote {args.save}")

    print(f"\n=== {'All end-to-end checks passed' if ok else 'FAILED'} ===\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
