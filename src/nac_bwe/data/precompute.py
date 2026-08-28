"""
Precomputes the shared dataset for the bandwidth-extension experiments.

One run produces every conditioning representation, so LatentBWENet and
AudioBWENet train on identical chunks. For each 24 kHz EnCodec chunk:

  - continuous latents      [D, T_frames]   float32  pre-RVQ encoder output
  - discrete codes          [n_q, T_frames] int16    RVQ codebook indices
  - quantized latents       [D, T_frames]   float32  codebook sum
  - reconstructed audio     [1, T_24k]      float32  decode of those latents

The 48 kHz clean audio is the training target. The reconstructed audio decodes
the same quantized latents the latent model consumes, so both models see the
same codec information as latents or as a waveform.

Output structure:
    output.root/
        index.json
        x_clean_48khz/        000000.pt  <- [1, 48000] float32
        latents/              000000.pt  <- [D, T_frames] float32
        latents_quantized/    000000.pt  <- [D, T_frames] float32
        codes/                000000.pt  <- [n_q, T_frames] int16
        x_reconstructed_24khz/000000.pt  <- [1, T_24k] float32

Usage:
    python -m nac_bwe.data.precompute --config configs/precompute/precompute_latent.yaml
    python -m nac_bwe.data.precompute --config configs/precompute/precompute_latent.yaml --dry-run
"""

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import yaml
from tqdm import tqdm

from nac_bwe.codec import EncodecProcessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    root: str
    extensions: list[str]


@dataclass
class AudioConfig:
    min_duration_s: float = 1.0
    silence_threshold_db: float = -50.0
    # Optional cap: keep only a `max_duration_s`-second window of each file
    # before chunking. Stops a few very long recordings from dominating the
    # dataset. None = no cap (use the whole file).
    max_duration_s: Optional[float] = None
    # Where the kept window comes from for files longer than the cap:
    #   "start"  - first max_duration_s seconds (deterministic, simple)
    #   "random" - a window at a per-file random offset, seeded by filename +
    #              run.seed so it's reproducible and independent of file order.
    #              Better for long soundscapes whose start is often atypical.
    crop_mode: str = "start"


@dataclass
class ChunkingConfig:
    chunk_duration_s: float = 1.0
    overlap_fraction: float = 0.5


@dataclass
class ResamplingConfig:
    source_sr: int = 44100
    target_sr: int = 24000   # EnCodec input rate
    output_sr: int = 48000   # clean target stored at this rate


@dataclass
class EncodecConfig:
    sample_rate: int = 24000
    bandwidth: float = 12.0
    device: str = "cpu"


@dataclass
class OutputConfig:
    root: str = "data/precomputed_latent_bwe"
    x_clean: str = "x_clean_48khz"
    latents: str = "latents"
    latents_quantized: str = "latents_quantized"
    codes: str = "codes"
    x_reconstructed: str = "x_reconstructed_24khz"


@dataclass
class RunConfig:
    seed: int = 42
    max_chunks: Optional[int] = None


@dataclass
class PrecomputeConfig:
    dataset: DatasetConfig
    audio: AudioConfig
    chunking: ChunkingConfig
    resampling: ResamplingConfig
    encodec: EncodecConfig
    output: OutputConfig
    run: RunConfig


def load_config(path: str) -> PrecomputeConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PrecomputeConfig(
        dataset=DatasetConfig(**raw["dataset"]),
        audio=AudioConfig(**raw.get("audio", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        resampling=ResamplingConfig(**raw.get("resampling", {})),
        encodec=EncodecConfig(**raw.get("encodec", {})),
        output=OutputConfig(**raw.get("output", {})),
        run=RunConfig(**raw.get("run", {})),
    )


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def collect_files(root: str, extensions: list[str]) -> list[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in exts)


def cap_duration(
    waveform: torch.Tensor,
    sr: int,
    max_duration_s: Optional[float],
    crop_mode: str = "start",
    file_key: str = "",
    seed: int = 42,
) -> torch.Tensor:
    """Keep a `max_duration_s`-second window of the waveform.

    No-op if the cap is None or the file is already shorter than the cap.
    crop_mode "start" takes the first window; "random" takes a window at a
    deterministic per-file offset (hash of file_key + seed), so it's
    reproducible and independent of the order files are processed in.
    """
    if max_duration_s is None:
        return waveform
    max_samples = int(max_duration_s * sr)
    n = waveform.shape[-1]
    if n <= max_samples:
        return waveform
    if crop_mode == "random":
        h = int(hashlib.sha256(f"{seed}:{file_key}".encode()).hexdigest(), 16)
        start = h % (n - max_samples + 1)
    elif crop_mode == "start":
        start = 0
    else:
        raise ValueError(f"Unknown crop_mode: {crop_mode!r} (use 'start' or 'random')")
    return waveform[:, start : start + max_samples]


def is_silent(waveform: torch.Tensor, threshold_db: float) -> bool:
    rms = waveform.pow(2).mean().sqrt()
    if rms < 1e-9:
        return True
    return (20 * torch.log10(rms)).item() < threshold_db


def chunk_waveform(
    waveform: torch.Tensor,
    chunk_samples: int,
    stride_samples: int,
) -> list[torch.Tensor]:
    chunks = []
    _, length = waveform.shape
    start = 0
    while start + chunk_samples <= length:
        chunks.append(waveform[:, start : start + chunk_samples].clone())
        start += stride_samples
    return chunks


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_chunk(
    chunk_24k: torch.Tensor,
    processor: EncodecProcessor,
    bandwidth: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    [1, T] at 24kHz -> four tensors:
        continuous latents     [D, T_frames] float32  (pre-RVQ encoder output)
        codes                  [n_q, T_frames] int16  (post-RVQ codebook indices)
        quantized latents      [D, T_frames] float32  (codebook sum, what a code-generating model produces)
        reconstructed audio    [1, T] float32         (EnCodec decode of the quantized latents)
    """
    original_length = chunk_24k.shape[-1]

    latents_list, latents_meta = processor.audio_to_latents(chunk_24k)
    latents = latents_list[0]  # [1, D, T_frames]

    codes, codes_meta = processor.latents_to_codes(
        latents_list, kbps=bandwidth, latents_meta=latents_meta,
    )
    # torchaudio 24kHz model: codes [n_q, 1, T_frames]

    latents_quantized = processor.codes_to_latents(codes)  # [1, D, T_frames]

    # Decode the quantized latents back to 24kHz audio, the audio model's input.
    # Same quantized representation the latent model consumes, just as a waveform.
    reconstructed = processor.decode_latents_audio(latents_quantized, codes_meta)
    if reconstructed.dim() == 3:
        reconstructed = reconstructed.squeeze(0)        # [1, T]
    reconstructed = reconstructed[:, :original_length]  # decoder may pad

    return (
        latents[0].cpu(),                   # [D, T_frames] float32
        codes[:, 0, :].short().cpu(),       # [n_q, T_frames] int16
        latents_quantized[0].cpu(),         # [D, T_frames] float32
        reconstructed.cpu(),                # [1, T] float32
    )


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run_count(
    files: list[Path],
    cfg: PrecomputeConfig,
    chunk_samples_out: int,
    stride_samples_out: int,
) -> int:
    output_sr = cfg.resampling.output_sr
    total = 0

    for f in tqdm(files, desc="Scanning"):
        try:
            waveform, sr = torchaudio.load(f)
        except Exception:
            continue

        if sr != output_sr:
            waveform = torchaudio.functional.resample(waveform, sr, output_sr)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = cap_duration(
            waveform, output_sr, cfg.audio.max_duration_s,
            cfg.audio.crop_mode, f.name, cfg.run.seed,
        )

        if waveform.shape[-1] / output_sr < cfg.audio.min_duration_s:
            continue

        if is_silent(waveform, cfg.audio.silence_threshold_db):
            continue

        chunks = chunk_waveform(waveform, chunk_samples_out, stride_samples_out)
        total += sum(1 for c in chunks if not is_silent(c, cfg.audio.silence_threshold_db))

        if cfg.run.max_chunks and total >= cfg.run.max_chunks:
            break

    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    source_sr  = cfg.resampling.source_sr
    encodec_sr = cfg.resampling.target_sr
    output_sr  = cfg.resampling.output_sr

    chunk_samples_out  = int(cfg.chunking.chunk_duration_s * output_sr)
    chunk_samples_enc  = int(cfg.chunking.chunk_duration_s * encodec_sr)
    stride_samples_out = max(1, int(chunk_samples_out * (1.0 - cfg.chunking.overlap_fraction)))

    log.info(f"Source:     {cfg.dataset.root}")
    log.info(f"Output:     {cfg.output.root}")
    log.info(f"Bandwidth:  {cfg.encodec.bandwidth} kbps")
    log.info(f"Source SR:  {source_sr} Hz")
    log.info(f"Output SR:  {output_sr} Hz  (x_clean)")
    log.info(f"EnCodec SR: {encodec_sr} Hz (latents/codes)")
    log.info(
        f"Chunk: {cfg.chunking.chunk_duration_s}s | "
        f"output samples: {chunk_samples_out} | "
        f"encodec samples: {chunk_samples_enc} | "
        f"stride: {stride_samples_out} samples at {output_sr}Hz"
    )

    files = collect_files(cfg.dataset.root, cfg.dataset.extensions)
    if not files:
        log.error(f"No files found in {cfg.dataset.root}")
        return
    log.info(f"Found {len(files)} source files")

    if args.dry_run:
        log.info("--- DRY RUN ---")
        total = dry_run_count(files, cfg, chunk_samples_out, stride_samples_out)
        capped = min(total, cfg.run.max_chunks or total)
        log.info(f"Would produce ~{capped} chunks")
        return

    out_root           = Path(cfg.output.root)
    clean_dir          = out_root / cfg.output.x_clean
    latents_dir        = out_root / cfg.output.latents
    latents_quant_dir  = out_root / cfg.output.latents_quantized
    codes_dir          = out_root / cfg.output.codes
    recon_dir          = out_root / cfg.output.x_reconstructed
    for d in (clean_dir, latents_dir, latents_quant_dir, codes_dir, recon_dir):
        d.mkdir(parents=True, exist_ok=True)

    log.info("Loading EncodecProcessor...")
    processor = EncodecProcessor(sr=encodec_sr, device=cfg.encodec.device)
    processor.model.set_target_bandwidth(cfg.encodec.bandwidth)
    log.info("Model loaded.")

    index: list[dict] = []
    sample_idx = 0

    for file_path in tqdm(files, desc="Files"):
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            log.warning(f"Could not load {file_path.name}: {e}")
            continue

        if sr != output_sr:
            waveform = torchaudio.functional.resample(waveform, sr, output_sr)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = cap_duration(
            waveform, output_sr, cfg.audio.max_duration_s,
            cfg.audio.crop_mode, file_path.name, cfg.run.seed,
        )

        if waveform.shape[-1] / output_sr < cfg.audio.min_duration_s:
            log.debug(f"Skipping {file_path.name}: too short")
            continue

        if is_silent(waveform, cfg.audio.silence_threshold_db):
            log.debug(f"Skipping {file_path.name}: silent")
            continue

        for chunk_out in chunk_waveform(waveform, chunk_samples_out, stride_samples_out):
            if is_silent(chunk_out, cfg.audio.silence_threshold_db):
                continue

            chunk_enc = torchaudio.functional.resample(chunk_out, output_sr, encodec_sr)

            latents, codes, latents_quantized, reconstructed = encode_chunk(
                chunk_enc, processor, cfg.encodec.bandwidth
            )

            fname = f"{sample_idx:06d}.pt"
            torch.save(chunk_out,         clean_dir         / fname)
            torch.save(latents,           latents_dir       / fname)
            torch.save(latents_quantized, latents_quant_dir / fname)
            torch.save(codes,             codes_dir         / fname)
            torch.save(reconstructed,     recon_dir         / fname)

            index.append({
                "id":                sample_idx,
                "source_file":       str(file_path),
                "clean":             str((clean_dir         / fname).relative_to(out_root)),
                "latents":           str((latents_dir       / fname).relative_to(out_root)),
                "latents_quantized": str((latents_quant_dir / fname).relative_to(out_root)),
                "codes":             str((codes_dir         / fname).relative_to(out_root)),
                "reconstructed":     str((recon_dir         / fname).relative_to(out_root)),
            })

            sample_idx += 1

            if cfg.run.max_chunks and sample_idx >= cfg.run.max_chunks:
                log.info(f"Reached max_chunks={cfg.run.max_chunks}, stopping.")
                break

        if cfg.run.max_chunks and sample_idx >= cfg.run.max_chunks:
            break

    index_path = out_root / "index.json"
    with open(index_path, "w") as f:
        json.dump(
            {
                "version":           1,
                "num_samples":       len(index),
                "source_sr":         source_sr,
                "output_sr":         output_sr,
                "encodec_sr":        encodec_sr,
                "chunk_samples_48k": chunk_samples_out,
                "chunk_samples_24k": chunk_samples_enc,
                "bandwidth":         cfg.encodec.bandwidth,
                "samples":           index,
            },
            f,
            indent=2,
        )

    log.info(f"Done. {sample_idx} chunks written to {out_root}")
    log.info(f"Index: {index_path}")
    log.info(f"Estimated disk: {_estimate_disk_gb(sample_idx, chunk_samples_out):.2f} GB")


def _estimate_disk_gb(n: int, samples_out: int) -> float:
    frames = 75  # ~75 frames per second at 24kHz encodec rate
    latents_bytes = frames * 128 * 4   # float32 (continuous)
    quant_bytes   = frames * 128 * 4   # float32 (quantized latents)
    codes_bytes   = frames * 8   * 2   # int16, 8 codebooks at 12kbps
    clean_bytes   = samples_out  * 4   # float32, 48kHz
    recon_bytes   = (samples_out // 2) * 4  # float32, 24kHz reconstructed audio
    per = latents_bytes + quant_bytes + codes_bytes + clean_bytes + recon_bytes
    return n * per / 1e9


if __name__ == "__main__":
    main()
