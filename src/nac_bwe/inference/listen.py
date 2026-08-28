"""
Inference script for LatentBWENet.

Combines:
  - LF (0-12 kHz): pretrained Vocos EnCodec 24 kHz model
  - HF (12-24 kHz): LatentBWENet checkpoint

Each band is synthesised to a 48 kHz waveform by its own iSTFT and the two
are summed in the time domain.

Output files per input:
    <stem>_clean.wav          original 48 kHz reference
    <stem>_reconstructed.wav  zero-padded codec, brick-wall at 12 kHz
    <stem>_enhanced.wav       Vocos LF + HF model

Usage:
    python -m nac_bwe.inference.listen \
        --input  data/textures/ \
        --checkpoint runs/latent_bwe_hf_small/best.pt \
        --output_dir runs/listen_hf
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from vocos import Vocos


from nac_bwe.codec import EncodecProcessor
from nac_bwe.models.latent_bwe_net import (
    LatentBWENet, LFExtractor, make_lf_extractor,
    SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH,
)
from nac_bwe.models.audio_bwe_net import AudioBWENet
from nac_bwe.training.config import DataConfig, ModelConfig, DiscriminatorConfig, LossConfig, TrainingConfig, OutputConfig, TrainConfig  # noqa: F401

ENCODEC_SR       = 24000
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".aiff", ".aif"}
VOCOS_REPO       = "charactr/vocos-encodec-24khz"
BANDWIDTH_ID     = 3   # 12 kbps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str) -> tuple[LatentBWENet, str, str]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg  = ckpt["config"]["model"]
    input_mode = ckpt.get("input_mode", "quantized")
    # "latent" (LatentBWENet, latents in) or "audio" (AudioBWENet,
    # reconstructed 24 kHz audio in).
    model_type = ckpt.get("model_type", "latent")
    # A checkpoint must be run with the synthesis framing it was trained
    # with, or it reintroduces the ~n_fft/2 train/stream offset.
    center     = getattr(model_cfg, "center", True)

    model_cls = AudioBWENet if model_type == "audio" else LatentBWENet
    model = model_cls(
        hidden_dim=model_cfg.hidden_dim,
        num_blocks=model_cfg.num_blocks,
        kernel_size=model_cfg.kernel_size,
        expansion=model_cfg.expansion,
        center=center,
    ).to(device)

    model.load_state_dict(ckpt["generator"])
    model.eval()

    print(
        f"Loaded checkpoint: epoch {ckpt['epoch']} | "
        f"val loss {ckpt['val_loss']:.4f} | "
        f"model_type={model_type} | input_mode={input_mode} | center={center}"
    )
    return model, input_mode, model_type


def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def zero_pad_to_48k(waveform_24k: torch.Tensor) -> torch.Tensor:
    n           = waveform_24k.shape[-1]
    spec        = torch.fft.rfft(waveform_24k, dim=-1)
    target_bins = n + 1
    spec_padded = F.pad(spec, (0, target_bins - spec.shape[-1]))
    return torch.fft.irfft(spec_padded, n=n * 2, dim=-1) * 2


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_file(
    path:       Path,
    hf_model:   LatentBWENet,
    lf_extractor: LFExtractor,
    processor:  EncodecProcessor,
    bandwidth:  float,
    input_mode: str,
    model_type: str,
    source_sr:  int,
    out_dir:    Path,
    device:     str,
    lf_backend: str = "vocos",
):
    print(f"Processing: {path.name}")

    x_clean_48k  = load_audio(path, source_sr)
    x_clean_24k  = torchaudio.functional.resample(x_clean_48k, source_sr, ENCODEC_SR)

    # --- EnCodec: latents + codes ---
    latents_list, _  = processor.audio_to_latents(x_clean_24k)
    codes, meta      = processor.latents_to_codes(latents_list, kbps=bandwidth)
    latents_q        = processor.codes_to_latents(codes)   # quantized latents [1, D, T]

    # --- Reference: zero-padded codec ---
    codec_24k        = processor.decode_latents_audio(latents_q, meta)
    if codec_24k.dim() == 3:
        codec_24k = codec_24k.squeeze(0)
    codec_24k        = codec_24k[:, :x_clean_24k.shape[-1]].cpu()
    x_reconstructed  = zero_pad_to_48k(codec_24k)

    # --- LF band (0-12 kHz): Vocos native decode -> 48 kHz waveform ---
    # codes shape from processor: [n_q, 1, T], vocos expects [K, B, T]
    lf_audio = lf_extractor.decode_lf_audio(codes.to(device), bandwidth_id=BANDWIDTH_ID)  # [1, T_48k]

    # --- HF band (12-24 kHz): our model's own iSTFT -> 48 kHz waveform ---
    # Audio model conditions on the reconstructed 24 kHz waveform; latent model
    # on the latents (continuous or quantized). Both predict HF only.
    if model_type == "audio":
        model_in = codec_24k.to(device)              # [1, T_24k] recon audio
    elif input_mode == "continuous":
        model_in = latents_list[0].to(device)
    else:
        model_in = latents_q.to(device)

    hf_audio = hf_model(model_in).squeeze(1)         # [1, 1, T] -> [1, T_48k]

    # --- Combine in the time domain (bands do not overlap; summation is exact) ---
    x_enhanced_mono = lf_extractor.combine_time_domain(lf_audio, hf_audio)
    x_enhanced = x_enhanced_mono.unsqueeze(1).cpu()   # [1, 1, T]

    min_len = min(x_clean_48k.shape[-1], x_reconstructed.shape[-1], x_enhanced.shape[-1])
    stem    = path.stem

    torchaudio.save(str(out_dir / f"{stem}_clean.wav"),         x_clean_48k[:, :min_len],   SAMPLE_RATE)
    torchaudio.save(str(out_dir / f"{stem}_reconstructed.wav"), x_reconstructed[:, :min_len], SAMPLE_RATE)
    torchaudio.save(str(out_dir / f"{stem}_enhanced_{lf_backend}.wav"),
                    x_enhanced[:, :, :min_len].squeeze(0), SAMPLE_RATE)

    print(f"  -> {stem}_clean | {stem}_reconstructed | {stem}_enhanced_{lf_backend}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       type=str, required=True)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--output_dir",  type=str, default="runs/listen_hf")
    p.add_argument("--bandwidth",   type=float, default=12.0)
    p.add_argument("--source_sr",   type=int, default=48000)
    p.add_argument("--device",      type=str, default="cpu")
    p.add_argument("--lf_backend",  type=str, default="vocos",
                   choices=["vocos", "encodec"],
                   help="LF (0-12 kHz) decoder: vocos (higher quality) or "
                        "encodec (codec's own decoder, natively streamable)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(f for f in input_path.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS)
    elif input_path.is_file():
        files = [input_path]
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")

    if not files:
        print(f"No audio files found at {args.input}")
        return
    print(f"Found {len(files)} file(s)")

    hf_model, input_mode, model_type = load_model(args.checkpoint, args.device)

    print(f"Loading EnCodec ({args.bandwidth} kbps)...")
    processor = EncodecProcessor(sr=ENCODEC_SR, device=args.device)
    processor.model.set_target_bandwidth(args.bandwidth)

    if args.lf_backend == "vocos":
        print(f"LF backend: Vocos ({VOCOS_REPO})...")
        vocos        = Vocos.from_pretrained(VOCOS_REPO)
        lf_extractor = make_lf_extractor("vocos", vocos_model=vocos, device=args.device)
    else:
        print("LF backend: EnCodec decoder (reusing processor)")
        lf_extractor = make_lf_extractor("encodec", processor=processor, device=args.device)

    for path in files:
        try:
            process_file(
                path, hf_model, lf_extractor, processor,
                args.bandwidth, input_mode, model_type, args.source_sr,
                out_dir, args.device, args.lf_backend,
            )
        except Exception as e:
            print(f"  Error on {path.name}: {e}")

    print(f"\nAll files saved to: {out_dir}")
    print("Listen order: _clean -> _reconstructed -> _enhanced")


if __name__ == "__main__":
    main()
