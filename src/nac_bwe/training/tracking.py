"""
Dual-sink experiment tracking for the training scripts. Every scalar is written
to both:

  * ``metrics.csv`` in the run's ``output_dir``, the source of truth for
    figures. Needs no internet and no W&B API.
  * Weights & Biases, the live dashboard. Use ``offline`` mode where compute
    nodes have no outbound internet and sync afterwards with ``wandb sync``.

Validation audio and spectrograms are always saved under
``<output_dir>/samples``. Setting ``log_media_to_wandb`` also uploads a few
clips.

If W&B is unavailable or disabled, CSV logging still works.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# Imported defensively so a broken or absent wandb cannot take down a run.
# CSV logging is the guaranteed sink.
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    wandb = None
    _WANDB_AVAILABLE = False


class ExperimentTracker:
    """Scalar/media logger with a CSV sink and an optional W&B sink.

    Parameters
    ----------
    out_dir
        The run's output directory. ``metrics.csv``, ``samples/`` and the W&B
        offline files (``wandb/``) all live here.
    wandb_cfg
        A ``WandbConfig`` (see ``nac_bwe.training.config``). If ``None`` or
        ``enabled=False``, only the CSV sink is active.
    run_config
        Flat/nested dict of hyperparameters recorded once at run start (W&B
        config + ``config_snapshot.json`` next to the CSV).
    resume
        When True, existing ``metrics.csv`` rows are kept and the same W&B run
        is resumed, so a requeued job appends rather than forks.
    """

    def __init__(
        self,
        out_dir: str | Path,
        wandb_cfg: Any = None,
        run_config: Optional[Dict[str, Any]] = None,
        resume: bool = False,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir = self.out_dir / "samples"
        self.csv_path = self.out_dir / "metrics.csv"

        self._rows: list[dict[str, Any]] = []
        self._fieldnames: list[str] = ["step"]
        if resume and self.csv_path.exists():
            self._load_existing_csv()

        self._cfg = wandb_cfg
        # WANDB_MODE overrides the config's mode, so the environment wins.
        self._mode = os.environ.get("WANDB_MODE") or getattr(wandb_cfg, "mode", "disabled")
        self._use_wandb = (
            _WANDB_AVAILABLE
            and wandb_cfg is not None
            and getattr(wandb_cfg, "enabled", False)
            and self._mode != "disabled"
        )
        self.run = None
        if self._use_wandb:
            self._init_wandb(run_config, resume)

        if run_config is not None:
            (self.out_dir / "config_snapshot.json").write_text(
                json.dumps(_jsonify(run_config), indent=2)
            )

    # -- W&B setup ---------------------------------------------------------
    def _init_wandb(self, run_config, resume) -> None:
        cfg = self._cfg
        run_name = cfg.run_name or self.out_dir.name
        # Stable id on disk so a resumed job rejoins its own run. An existing
        # id file always attaches with resume="allow": a run that crashed before
        # its first epoch leaves an id but no checkpoint, and resume="never"
        # would reject the id and fall back to CSV-only.
        id_file = self.out_dir / "wandb_run_id.txt"
        if id_file.exists():
            run_id = id_file.read_text().strip()
            resume_mode = "allow"
        else:
            run_id = wandb.util.generate_id()
            id_file.write_text(run_id)
            resume_mode = "allow" if resume else "never"
        try:
            self.run = wandb.init(
                project=cfg.project,
                entity=cfg.entity,
                name=run_name,
                id=run_id,
                mode=self._mode,             # env-overridable; "offline" on HPC
                group=cfg.group,             # groups an ablation sweep's runs
                tags=list(cfg.tags),
                dir=str(self.out_dir),
                config=_jsonify(run_config or {}),
                resume=resume_mode,
            )
        except Exception as e:  # pragma: no cover
            print(f"[tracking] W&B init failed ({e!r}); continuing CSV-only.")
            self._use_wandb = False
            self.run = None

    # -- scalar logging ----------------------------------------------------
    def log(self, metrics: Dict[str, float], step: int) -> None:
        """Log a flat dict of scalars at ``step`` (epoch) to CSV and W&B."""
        row = {"step": step}
        for k, v in metrics.items():
            row[k] = float(v)
            if k not in self._fieldnames:
                self._fieldnames.append(k)
        self._rows.append(row)
        self._flush_csv()
        if self._use_wandb and self.run is not None:
            self.run.log(dict(metrics), step=step)

    def set_summary(self, summary: Dict[str, Any]) -> None:
        """Record run-level constants (param count, receptive field, RTF, …).

        Written to ``summary.json`` and into the W&B run summary, where they
        become sortable columns in the runs table.
        """
        (self.out_dir / "summary.json").write_text(
            json.dumps(_jsonify(summary), indent=2)
        )
        if self._use_wandb and self.run is not None:
            for k, v in summary.items():
                self.run.summary[k] = v

    # -- media logging -----------------------------------------------------
    def log_media(
        self,
        step: int,
        audio: Optional[Dict[str, Tuple["torch.Tensor", int]]] = None,
        spectrograms: Optional[Dict[str, "torch.Tensor"]] = None,
    ) -> None:
        """Save validation media locally; optionally upload a lean set to W&B.

        ``audio``: name -> (waveform [T] or [1,T] float tensor, sample_rate).
        ``spectrograms``: name -> log-magnitude 2D tensor [freq, time].
        """
        if not getattr(self._cfg, "log_media", False):
            return
        # A missing matplotlib or a write failure must not stop training.
        try:
            self._log_media_inner(step, audio, spectrograms)
        except Exception as e:
            print(f"[tracking] media logging skipped at epoch {step} ({e!r})")

    def _log_media_inner(
        self,
        step: int,
        audio: Optional[Dict[str, Tuple["torch.Tensor", int]]] = None,
        spectrograms: Optional[Dict[str, "torch.Tensor"]] = None,
    ) -> None:
        step_dir = self.samples_dir / f"epoch_{step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        wandb_payload: dict[str, Any] = {}

        if audio:
            # soundfile rather than torchaudio.save, which needs a decoder
            # backend that is not always present.
            import soundfile as sf
            for name, (wav, sr) in audio.items():
                w = wav.detach().cpu().float()
                if w.dim() == 1:
                    w = w.unsqueeze(0)
                path = step_dir / f"{name}.wav"
                sf.write(str(path), w.squeeze(0).numpy(), sr)   # [1,T] -> [T]
                if self._wandb_media:
                    wandb_payload[f"audio/{name}"] = wandb.Audio(
                        w.squeeze(0).numpy(), sample_rate=sr
                    )

        if spectrograms:
            for name, spec in spectrograms.items():
                path = step_dir / f"{name}.png"
                _save_spectrogram_png(spec, path)
                if self._wandb_media:
                    wandb_payload[f"spec/{name}"] = wandb.Image(str(path))

        if self._use_wandb and wandb_payload and self.run is not None:
            self.run.log(wandb_payload, step=step)

    @property
    def _wandb_media(self) -> bool:
        return (
            self._use_wandb
            and self.run is not None
            and getattr(self._cfg, "log_media_to_wandb", False)
        )

    def should_log_media(self, epoch: int) -> bool:
        every = getattr(self._cfg, "log_media_every", 0) or 0
        return bool(getattr(self._cfg, "log_media", False)) and every > 0 and epoch % every == 0

    # -- teardown ----------------------------------------------------------
    def finish(self) -> None:
        if self._use_wandb and self.run is not None:
            self.run.finish()

    # -- CSV helpers -------------------------------------------------------
    def _load_existing_csv(self) -> None:
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for name in reader.fieldnames or []:
                if name not in self._fieldnames:
                    self._fieldnames.append(name)
            for r in reader:
                row = {k: _maybe_float(v) for k, v in r.items()}
                if "step" in row and row["step"] == row["step"]:  # not NaN
                    row["step"] = int(row["step"])
                self._rows.append(row)

    def _flush_csv(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for r in self._rows:
                writer.writerow(r)


# ---------------------------------------------------------------------------
# Run-level RTF / latency measurement (for the "how small / how low-latency"
# ablation axes). Model-agnostic: caller supplies one real input tensor.
# ---------------------------------------------------------------------------
def measure_rtf(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    out_sample_rate: int,
    device: str,
    n_warmup: int = 3,
    n_iters: int = 10,
) -> Dict[str, float]:
    """Return real-time factor and per-call latency for one forward pass.

    ``rtf`` = audio-seconds produced / wall-seconds of compute (>1 = faster than
    real time). ``latency_ms`` = mean forward-pass time for ``sample_input``.
    The output's audio duration is inferred from its last dim and
    ``out_sample_rate``.
    """
    model.eval()
    x = sample_input.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            y = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            y = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_iters
    audio_seconds = y.shape[-1] / out_sample_rate
    return {
        "rtf": audio_seconds / dt if dt > 0 else float("nan"),
        "latency_ms": dt * 1e3,
        "audio_seconds_per_call": audio_seconds,
    }


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _save_spectrogram_png(spec: "torch.Tensor", path: Path) -> None:
    """Save a 2D log-magnitude spectrogram as a PNG (Agg backend, no display)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = spec.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(s, origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("frame")
    ax.set_ylabel("freq bin")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _jsonify(obj: Any) -> Any:
    """Best-effort conversion of dataclasses/tensors to JSON-serialisable form."""
    from dataclasses import is_dataclass, asdict

    if is_dataclass(obj):
        return _jsonify(asdict(obj))
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _maybe_float(v: Any) -> Any:
    try:
        return float(v)
    except (TypeError, ValueError):
        return v
