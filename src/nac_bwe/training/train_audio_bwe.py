"""
Training script for AudioBWENet, which predicts HF STFT bins 641-1280
(12-24 kHz) from EnCodec's decoded 24 kHz audio.

Shares the config schema, train/val loop, losses and target with
``train_latent_bwe``. The differences are the model class and the dataset
input mode ("audio" rather than a latent mode). Run it on the same precomputed
dataset as the latent model so both train on identical chunks.

Usage:
    python -u -m nac_bwe.training.train_audio_bwe --config configs/train/train_audio_small.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader


from nac_bwe.data.dataset import LatentBWEDataset
from nac_bwe.models.audio_bwe_net import AudioBWENet
from nac_bwe.models.latent_bwe_net import count_parameters, SAMPLE_RATE
from nac_bwe.losses.losses import (
    MultiResolutionDiscriminator,
    DiscriminatorLoss,
    GeneratorLoss,
    count_discriminator_parameters,
)
# Reuse the latent trainer's config schema + train/val loop + media builder
# verbatim so the two experiments differ only in model and input representation.
from nac_bwe.training.train_latent_bwe import (
    load_config, train_one_epoch, validate, build_val_media, resolve_resume_path,
    init_generator_from,
)
from nac_bwe.training.tracking import ExperimentTracker, measure_rtf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to an epoch_XXXX.pt checkpoint to resume from. Pass 'auto' "
             "to pick the latest epoch_*.pt in the run's output_dir.",
    )
    parser.add_argument(
        "--auto-resume", action="store_true",
        help="Resume from the latest epoch_*.pt in output_dir if one exists, "
             "otherwise start fresh. Safe to always pass on HPC: a resubmitted "
             "job continues, a first submission starts from scratch.",
    )
    parser.add_argument(
        "--init-generator", type=str, default=None,
        help="Warm-start the generator's WEIGHTS from another run's checkpoint "
             "(two-stage training: reconstruction pretrain -> adversarial "
             "fine-tune). Optimiser, scheduler, epoch counter and discriminator "
             "all start fresh. Ignored once this run has its own checkpoint, so "
             "it composes with --auto-resume on HPC.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_adversarial = cfg.loss.use_adversarial

    if cfg.data.input_mode != "audio":
        print(f"WARNING: data.input_mode is '{cfg.data.input_mode}', forcing "
              f"'audio' for the audio model.")
        cfg.data.input_mode = "audio"

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else "cpu" if args.device == "auto"
        else args.device
    )

    print(f"Device:          {device}")
    print(f"Input:           reconstructed 24 kHz audio (input_mode=audio)")
    print(f"Adversarial GAN: {use_adversarial}")
    print(f"STFT center:     {cfg.model.center}")

    out_dir = Path(cfg.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg.__dict__, f)

    # Resolved before the tracker so a resumed run appends to metrics.csv and
    # rejoins its existing W&B run instead of forking a new one.
    resume_path = resolve_resume_path(out_dir, args.resume, args.auto_resume)

    tracker = ExperimentTracker(
        out_dir, wandb_cfg=cfg.wandb, run_config=cfg.__dict__,
        resume=(resume_path is not None),
    )

    train_set = LatentBWEDataset(
        cfg.data.index_path, split="train", input_mode="audio",
        val_fraction=cfg.data.val_fraction, seed=cfg.data.seed,
        in_memory=cfg.data.in_memory,
    )
    val_set = LatentBWEDataset(
        cfg.data.index_path, split="val", input_mode="audio",
        val_fraction=cfg.data.val_fraction, seed=cfg.data.seed,
        in_memory=cfg.data.in_memory,
    )
    # Workers only help when streaming from disk; with in_memory the data is
    # already cached, so workers would just duplicate it across processes.
    nw = 0 if cfg.data.in_memory else cfg.data.num_workers
    _dl = dict(num_workers=nw, pin_memory=(device == "cuda"),
               persistent_workers=(nw > 0))
    print(f"DataLoader workers:       {nw} (in_memory={cfg.data.in_memory})")
    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True,  **_dl)
    val_loader   = DataLoader(val_set,   batch_size=cfg.training.batch_size, shuffle=False, **_dl)

    generator = AudioBWENet(
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks,
        kernel_size=cfg.model.kernel_size,
        expansion=cfg.model.expansion,
        center=cfg.model.center,
    ).to(device)
    print(f"Generator parameters:     {count_parameters(generator):,}")
    print(f"Receptive field:          {generator.receptive_field_ms():.0f} ms (backbone)")

    opt_g = torch.optim.AdamW(
        generator.parameters(), lr=cfg.training.lr_generator,
        betas=(0.8, 0.99), weight_decay=cfg.training.weight_decay,
    )
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=cfg.training.epochs, eta_min=cfg.training.lr_generator * 0.1,
    )

    discriminator = None
    opt_d = sched_d = disc_loss_fn = None
    if use_adversarial:
        discriminator = MultiResolutionDiscriminator(
            resolutions=[tuple(r) for r in cfg.discriminator.resolutions],
            n_filters=cfg.discriminator.n_filters,
            n_layers=cfg.discriminator.n_layers,
        ).to(device)
        print(f"Discriminator parameters: {count_discriminator_parameters(discriminator):,}")
        opt_d = torch.optim.AdamW(
            discriminator.parameters(), lr=cfg.training.lr_discriminator,
            betas=(0.8, 0.99), weight_decay=cfg.training.weight_decay,
        )
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_d, T_max=cfg.training.epochs, eta_min=cfg.training.lr_discriminator * 0.1,
        )
        disc_loss_fn = DiscriminatorLoss()

    gen_loss_fn = GeneratorLoss(
        sample_rate=SAMPLE_RATE,
        lambda_mel=cfg.loss.lambda_mel,
        lambda_hf=cfg.loss.lambda_hf,
        lambda_adv=cfg.loss.lambda_adv if use_adversarial else 0.0,
        lambda_fm=cfg.loss.lambda_fm   if use_adversarial else 0.0,
        center=cfg.model.center,
    ).to(device)

    best_val_g = float("inf")
    start_epoch = 1

    # Warm start only when this run has no checkpoint of its own: on a requeue
    # --auto-resume must win, or the job would silently restart stage 2 from
    # stage 1 and throw away the fine-tuning done so far.
    if args.init_generator and resume_path is None:
        init_generator_from(generator, Path(args.init_generator), device,
                            expect_input_mode="audio")
    elif args.init_generator:
        print(f"--init-generator ignored: resuming this run's own {resume_path.name}")

    if resume_path is not None:
        print(f"Resuming from:   {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        if use_adversarial:
            if "discriminator" not in ckpt or "opt_d" not in ckpt:
                raise KeyError(
                    f"{resume_path} has no discriminator/opt_d state but this "
                    "config is adversarial. Resume from a GAN checkpoint."
                )
            discriminator.load_state_dict(ckpt["discriminator"])
            opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = int(ckpt["epoch"]) + 1
        # If the checkpoint carries no scheduler state, replay the schedule
        # instead, exact for CosineAnnealingLR, whose LR depends only on the
        # step count.
        if "sched_g" in ckpt:
            sched_g.load_state_dict(ckpt["sched_g"])
            if sched_d is not None and "sched_d" in ckpt:
                sched_d.load_state_dict(ckpt["sched_d"])
        else:
            print("  (no scheduler state in checkpoint, replaying LR schedule)")
            for _ in range(ckpt["epoch"]):
                sched_g.step()
                if sched_d is not None:
                    sched_d.step()
        # Recover the running best from best.pt when the checkpoint does not
        # carry it, so a resume cannot overwrite a better checkpoint on its
        # first epoch.
        if "best_val_g" in ckpt:
            best_val_g = float(ckpt["best_val_g"])
        elif (out_dir / "best.pt").exists():
            best_val_g = float(torch.load(out_dir / "best.pt", map_location="cpu",
                                          weights_only=False)["val_loss"])
        print(f"Resumed at epoch {start_epoch} (best_val_g={best_val_g:.4f})")

    # Run-level constants for the ablation grid (params / latency / RTF).
    summary = {
        "generator_params": count_parameters(generator),
        "receptive_field_ms": generator.receptive_field_ms(),
        "center": cfg.model.center,
    }
    if discriminator is not None:
        summary["discriminator_params"] = count_discriminator_parameters(discriminator)
    try:
        sample_input = next(iter(val_loader))[0]
        summary.update(measure_rtf(generator, sample_input, SAMPLE_RATE, device))
    except Exception as e:
        print(f"[tracking] RTF measurement skipped ({e!r})")
    tracker.set_summary(summary)

    print(f"\nTraining epochs {start_epoch}-{cfg.training.epochs} -> {out_dir}\n")

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        _t0 = time.perf_counter()
        train_metrics = train_one_epoch(
            generator, discriminator, train_loader, opt_g, opt_d,
            gen_loss_fn, disc_loss_fn, cfg.training.grad_clip, device, use_adversarial,
            cfg.model.center, desc=f"E{epoch}/{cfg.training.epochs} train",
        )
        val_metrics = validate(
            generator, discriminator, val_loader,
            gen_loss_fn, disc_loss_fn, device, use_adversarial,
            cfg.model.center, desc=f"E{epoch}/{cfg.training.epochs} val",
        )
        _dt = time.perf_counter() - _t0

        sched_g.step()
        if sched_d is not None:
            sched_d.step()

        log_row = {f"train/{k}": v for k, v in train_metrics.items()}
        log_row.update({f"val/{k}": v for k, v in val_metrics.items()})
        log_row["lr/generator"] = sched_g.get_last_lr()[0]
        if sched_d is not None:
            log_row["lr/discriminator"] = sched_d.get_last_lr()[0]
        log_row["time/epoch_sec"] = _dt
        tracker.log(log_row, step=epoch)

        if tracker.should_log_media(epoch):
            audio, specs = build_val_media(
                generator, next(iter(val_loader)), device, cfg.model.center,
                cfg.wandb.num_media_samples, SAMPLE_RATE,
            )
            tracker.log_media(epoch, audio=audio, spectrograms=specs)

        adv_str = (
            f" adv {train_metrics['adv']:.3f} fm {train_metrics['fm']:.3f}"
            f" | D: {train_metrics['d_total']:.4f}"
            if use_adversarial else ""
        )
        _eta_h = _dt * (cfg.training.epochs - epoch) / 3600
        print(
            f"Epoch {epoch:3d}/{cfg.training.epochs} | "
            f"G: {train_metrics['g_total']:.4f} "
            f"(hf {train_metrics['hf']:.3f}{adv_str}) | "
            f"val_G: {val_metrics['g_total']:.4f} | "
            f"{_dt:.0f}s/ep  ETA {_eta_h:.1f}h",
            flush=True,
        )

        def _ckpt(extra: dict) -> dict:
            d = {
                "epoch":      epoch,
                "generator":  generator.state_dict(),
                "val_loss":   val_metrics["g_total"],
                "config":     cfg.__dict__,
                "input_mode": "audio",
                "model_type": "audio",   # so inference loads AudioBWENet
            }
            d.update(extra)
            if use_adversarial:
                d["discriminator"] = discriminator.state_dict()
            return d

        if val_metrics["g_total"] < best_val_g:
            best_val_g = val_metrics["g_total"]
            torch.save(_ckpt({}), out_dir / "best.pt")

        if epoch % cfg.output.save_every == 0:
            extra = {
                "opt_g":      opt_g.state_dict(),
                "sched_g":    sched_g.state_dict(),
                "best_val_g": best_val_g,
            }
            if use_adversarial:
                extra["opt_d"]   = opt_d.state_dict()
                extra["sched_d"] = sched_d.state_dict()
            torch.save(_ckpt(extra), out_dir / f"epoch_{epoch:04d}.pt")

    tracker.finish()
    print(f"\nDone. Best val loss: {best_val_g:.4f}")
    print(f"Checkpoints: {out_dir}")


if __name__ == "__main__":
    main()
