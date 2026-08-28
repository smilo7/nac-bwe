"""
Release checkpoints: small, self-describing weights for inference.

A training checkpoint carries the optimizer, scheduler and discriminator state
needed to resume a run, around 25x the size of the weights themselves. It also
stores its config as pickled dataclasses, so loading it requires this package to
be importable and ``weights_only=False``.

A release checkpoint keeps only the generator weights plus the handful of
hyperparameters needed to rebuild it, with the config flattened to plain types.
It loads under ``weights_only=True``, which executes no pickled code.

    python -m nac_bwe.checkpoints export runs/<run>/epoch_0099.pt out.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

# Hyperparameters needed to reconstruct the generator.
_MODEL_FIELDS = ("hidden_dim", "num_blocks", "kernel_size", "expansion", "center")

FORMAT_VERSION = 1


def export_release(train_ckpt: str | Path, out_path: str | Path) -> Dict[str, Any]:
    """Strip a training checkpoint down to a release checkpoint and write it."""
    ckpt = torch.load(train_ckpt, map_location="cpu", weights_only=False)

    model_cfg = ckpt["config"]["model"]
    release = {
        "format_version": FORMAT_VERSION,
        "generator": {k: v.cpu() for k, v in ckpt["generator"].items()},
        # Plain types only, so the file loads with weights_only=True.
        "model_config": {f: getattr(model_cfg, f) for f in _MODEL_FIELDS},
        # "latent" (LatentBWENet) or "audio" (AudioBWENet).
        "model_type": ckpt.get("model_type", "latent"),
        "input_mode": ckpt.get("input_mode", "quantized"),
        "epoch": int(ckpt["epoch"]),
        "val_loss": float(ckpt["val_loss"]),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, out_path)
    return release


def load_release(path: str | Path, device: str = "cpu"):
    """
    Load a release checkpoint and return an eval-mode model ready for inference.

    Returns ``(model, meta)``, where ``meta`` carries model_type, input_mode and
    the provenance fields.
    """
    from nac_bwe.models.latent_bwe_net import LatentBWENet
    from nac_bwe.models.audio_bwe_net import AudioBWENet

    ckpt = torch.load(path, map_location=device, weights_only=True)

    model_cls = AudioBWENet if ckpt["model_type"] == "audio" else LatentBWENet
    model = model_cls(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k != "generator"}
    return model, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="strip a training checkpoint for release")
    p_exp.add_argument("train_ckpt")
    p_exp.add_argument("out_path")

    p_ins = sub.add_parser("inspect", help="print a release checkpoint's metadata")
    p_ins.add_argument("path")

    args = ap.parse_args()

    if args.cmd == "export":
        rel = export_release(args.train_ckpt, args.out_path)
        before = Path(args.train_ckpt).stat().st_size
        after = Path(args.out_path).stat().st_size
        n_params = sum(t.numel() for t in rel["generator"].values())
        print(f"{args.train_ckpt} -> {args.out_path}")
        print(f"  {n_params:,} params | {rel['model_type']} | epoch {rel['epoch']}")
        print(f"  {before/1e6:.1f} MB -> {after/1e6:.1f} MB")
    else:
        _, meta = load_release(args.path)
        for k, v in meta.items():
            print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
