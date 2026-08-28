"""
Training script for LatentBWENet, the HF-only model that predicts
STFT bins 641-1280 (12-24 kHz) from EnCodec latents.

Target: highpass-filtered x_clean (same filter as HFMRSTFTLoss).
Loss:   HF MRSTFT (primary) + optional adversarial on HF waveforms.
Vocos is not in the training loop, only at inference.

Usage:
    python -u -m nac_bwe.training.train_latent_bwe --config configs/train/headline/latent_small_gan.yaml
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from nac_bwe.data.dataset import LatentBWEDataset
from nac_bwe.models.latent_bwe_net import (
    LatentBWENet, count_parameters,
    SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH, HF_BIN_START,
)
from nac_bwe.losses.losses import (
    MultiResolutionDiscriminator,
    DiscriminatorLoss,
    GeneratorLoss,
    spectral_highpass,
    count_discriminator_parameters,
)
# Config dataclasses live in nac_bwe.training.config (stable pickle home);
# re-exported here so `from nac_bwe.training.train_latent_bwe import load_config`
# and checkpoint code keep working.
from nac_bwe.training.config import (  # noqa: F401
    DataConfig, ModelConfig, DiscriminatorConfig, LossConfig,
    TrainingConfig, OutputConfig, WandbConfig, TrainConfig, load_config,
)
from nac_bwe.training.tracking import ExperimentTracker, measure_rtf

HF_CUTOFF = 12000.0


# ---------------------------------------------------------------------------
# Qualitative validation media (shared by the latent and audio trainers)
# ---------------------------------------------------------------------------
def _logmag_spec(wav: torch.Tensor) -> torch.Tensor:
    """Log-magnitude STFT (dB) of a 1-D waveform, for spectrogram previews."""
    win = torch.hann_window(WIN_LENGTH, device=wav.device)
    S = torch.stft(wav, N_FFT, HOP_LENGTH, WIN_LENGTH, window=win,
                   return_complex=True, center=True)
    return 20.0 * torch.log10(S.abs() + 1e-7)


def resolve_resume_path(out_dir: Path, resume: Optional[str],
                        auto_resume: bool) -> Optional[Path]:
    """Decide which checkpoint (if any) to resume from.

    ``--resume <path>``  an explicit checkpoint file.
    ``--resume auto``    latest ``epoch_*.pt`` in ``out_dir``, errors if none.
                         use when you *know* you're continuing a run.
    ``--auto-resume``    latest ``epoch_*.pt`` if one exists, else start fresh.
                         Use this for batch jobs: re-running after a
                         wall-clock kill continues where it left off, while the
                         first run still starts from scratch.
    """
    if resume is not None:
        if resume == "auto":
            ckpts = sorted(out_dir.glob("epoch_*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"--resume auto: no epoch_*.pt in {out_dir}")
            return ckpts[-1]
        return Path(resume)
    if auto_resume:
        ckpts = sorted(out_dir.glob("epoch_*.pt"))
        if ckpts:
            return ckpts[-1]
        print("--auto-resume: no checkpoint in output_dir, starting fresh.")
    return None


def init_generator_from(generator, init_path: Path, device: str,
                        expect_input_mode: Optional[str] = None) -> None:
    """Warm-start ``generator`` from another run's checkpoint (weights only).

    This is the two-stage recipe: pretrain with the reconstruction loss alone
    (cheap, stable, no discriminator), then fine-tune adversarially from those
    weights. Unlike ``--resume`` it deliberately loads *nothing* but the
    generator. The optimiser, scheduler, epoch counter and discriminator all
    start fresh, because stage 2 is a new run with a different objective, not a
    continuation of stage 1. Stage 1's cosine LR has annealed to ~0, so
    inheriting its optimiser state would leave the fine-tune unable to move.

    ``expect_input_mode`` guards against warm-starting an audio model from a
    latent checkpoint or vice versa. The shapes differ (Linear(641, h) against
    Linear(128, h)) so ``load_state_dict`` would fail anyway, just obscurely.
    """
    ckpt = torch.load(init_path, map_location=device, weights_only=False)
    if "generator" not in ckpt:
        raise KeyError(f"{init_path} has no 'generator' state dict.")
    got = ckpt.get("input_mode")
    if expect_input_mode and got and got != expect_input_mode:
        raise ValueError(
            f"{init_path} was trained with input_mode='{got}' but this config "
            f"is '{expect_input_mode}'. Warm-start from a matching model."
        )
    generator.load_state_dict(ckpt["generator"])
    print(f"Warm-started generator from: {init_path}")
    print(f"  (stage-1 epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.4f}; "
          f"optimiser/scheduler/discriminator start fresh)")


def build_val_media(generator, batch, device, center, n_samples, sample_rate):
    """Predicted-vs-target HF audio + spectrograms for a few validation items.

    Model-agnostic (latent or audio): ``generator(input)`` yields the HF-band
    waveform for either. Returns ``(audio, spectrograms)`` dicts ready for
    ``ExperimentTracker.log_media``.
    """
    generator.eval()
    inp, x_clean = batch
    inp = inp.to(device)[:n_samples]
    x_clean = x_clean.to(device)[:n_samples]
    hp_window = torch.hann_window(WIN_LENGTH, device=device)
    x_target = spectral_highpass(x_clean, hp_window, N_FFT, HOP_LENGTH,
                                 WIN_LENGTH, HF_BIN_START, center)
    with torch.no_grad():
        x_pred = generator(inp)
    min_len = min(x_pred.shape[-1], x_target.shape[-1])
    x_pred, x_target = x_pred[..., :min_len], x_target[..., :min_len]

    audio, specs = {}, {}
    for i in range(x_pred.shape[0]):
        p = x_pred[i].reshape(-1)
        t = x_target[i].reshape(-1)
        audio[f"pred_hf_{i}"] = (p.cpu(), sample_rate)
        audio[f"target_hf_{i}"] = (t.cpu(), sample_rate)
        specs[f"pred_hf_{i}"] = _logmag_spec(p)
        specs[f"target_hf_{i}"] = _logmag_spec(t)
    return audio, specs


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def _disc_confidence(real_outputs, fake_outputs):
    """Mean discriminator output on real and on generated HF, and their gap.

    Logged because the loss values alone hide whether the critic is learning
    anything. Under LSGAN a useful critic and a critic that answers ~0.5 to
    everything can sit at a similar d_total, so d_total is not diagnostic. What
    matters is the SEPARATION d_real - d_fake: near zero means the
    discriminator cannot tell real HF from generated HF, and the adversarial
    and feature-matching terms then carry no information regardless of the
    weights placed on them.
    """
    with torch.no_grad():
        dr = torch.stack([o[0].mean() for o in real_outputs]).mean().item()
        df = torch.stack([o[0].mean() for o in fake_outputs]).mean().item()
    return dr, df


def train_one_epoch(
    generator:       LatentBWENet,
    discriminator:   Optional[MultiResolutionDiscriminator],
    loader:          DataLoader,
    opt_g:           torch.optim.Optimizer,
    opt_d:           Optional[torch.optim.Optimizer],
    gen_loss_fn:     GeneratorLoss,
    disc_loss_fn:    Optional[DiscriminatorLoss],
    grad_clip:       float,
    device:          str,
    use_adversarial: bool,
    center:          bool = True,
    desc:            str  = "train",
):
    generator.train()
    if discriminator is not None:
        discriminator.train()

    totals = {"g_total": 0., "mel": 0., "hf": 0., "adv": 0., "fm": 0., "d_total": 0.,
              "d_real": 0., "d_fake": 0., "d_sep": 0.}
    n = 0
    hp_window = torch.hann_window(WIN_LENGTH, device=device)

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for latents, x_clean in pbar:
        latents   = latents.to(device)
        x_clean   = x_clean.to(device)
        # Sharp brickwall target matched to the model's synthesis band (no soft
        # biquad -> no 12 kHz crossover pile-up). `center` must match the model.
        x_target  = spectral_highpass(x_clean, hp_window, N_FFT, HOP_LENGTH, WIN_LENGTH, HF_BIN_START, center)

        x_enhanced = generator(latents)

        min_len    = min(x_enhanced.shape[-1], x_target.shape[-1])
        x_enhanced = x_enhanced[:, :, :min_len]
        x_target_t = x_target[:, :, :min_len]

        if use_adversarial:
            opt_d.zero_grad()
            real_outputs = discriminator(x_target_t)
            fake_outputs = discriminator(x_enhanced.detach())
            d_loss, _, _ = disc_loss_fn(real_outputs, fake_outputs)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
            opt_d.step()

            opt_g.zero_grad()
            fake_outputs_g = discriminator(x_enhanced)
            real_outputs_g = discriminator(x_target_t)
            g_loss, loss_dict = gen_loss_fn(x_enhanced, x_target_t, fake_outputs_g, real_outputs_g)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            opt_g.step()

            totals["d_total"] += d_loss.item()
            _dr, _df = _disc_confidence(real_outputs, fake_outputs)
            totals["d_real"] += _dr
            totals["d_fake"] += _df
            totals["d_sep"]  += _dr - _df
        else:
            opt_g.zero_grad()
            g_loss, loss_dict = gen_loss_fn(x_enhanced, x_target_t, None, None)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            opt_g.step()

        totals["g_total"] += loss_dict["total"]
        totals["mel"]     += loss_dict["mel"]
        totals["hf"]      += loss_dict["hf"]
        totals["adv"]     += loss_dict.get("adv", 0.)
        totals["fm"]      += loss_dict.get("fm", 0.)
        n += 1
        pbar.set_postfix(g=f"{loss_dict['total']:.3f}", hf=f"{loss_dict['hf']:.3f}")

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(
    generator:       LatentBWENet,
    discriminator:   Optional[MultiResolutionDiscriminator],
    loader:          DataLoader,
    gen_loss_fn:     GeneratorLoss,
    disc_loss_fn:    Optional[DiscriminatorLoss],
    device:          str,
    use_adversarial: bool,
    center:          bool = True,
    desc:            str  = "val",
):
    generator.eval()
    if discriminator is not None:
        discriminator.eval()

    totals = {"g_total": 0., "mel": 0., "hf": 0., "adv": 0., "fm": 0., "d_total": 0.,
              "d_real": 0., "d_fake": 0., "d_sep": 0.}
    n = 0
    hp_window = torch.hann_window(WIN_LENGTH, device=device)

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for latents, x_clean in pbar:
        latents   = latents.to(device)
        x_clean   = x_clean.to(device)
        # Sharp brickwall target matched to the model's synthesis band (no soft
        # biquad -> no 12 kHz crossover pile-up). `center` must match the model.
        x_target  = spectral_highpass(x_clean, hp_window, N_FFT, HOP_LENGTH, WIN_LENGTH, HF_BIN_START, center)

        x_enhanced = generator(latents)

        min_len    = min(x_enhanced.shape[-1], x_target.shape[-1])
        x_enhanced = x_enhanced[:, :, :min_len]
        x_target_t = x_target[:, :, :min_len]

        if use_adversarial:
            real_outputs = discriminator(x_target_t)
            fake_outputs = discriminator(x_enhanced)
            d_loss, _, _ = disc_loss_fn(real_outputs, fake_outputs)
            _, loss_dict = gen_loss_fn(x_enhanced, x_target_t, fake_outputs, real_outputs)
            totals["d_total"] += d_loss.item()
            _dr, _df = _disc_confidence(real_outputs, fake_outputs)
            totals["d_real"] += _dr
            totals["d_fake"] += _df
            totals["d_sep"]  += _dr - _df
        else:
            _, loss_dict = gen_loss_fn(x_enhanced, x_target_t, None, None)

        totals["g_total"] += loss_dict["total"]
        totals["mel"]     += loss_dict["mel"]
        totals["hf"]      += loss_dict["hf"]
        totals["adv"]     += loss_dict.get("adv", 0.)
        totals["fm"]      += loss_dict.get("fm", 0.)
        n += 1
        pbar.set_postfix(g=f"{loss_dict['total']:.3f}", hf=f"{loss_dict['hf']:.3f}")

    return {k: v / max(n, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
             "otherwise start fresh. Safe to always pass in a batch job: a "
             "re-run continues, a first run starts from scratch.",
    )
    parser.add_argument(
        "--init-generator", type=str, default=None,
        help="Warm-start the generator's weights from another run's checkpoint "
             "(two-stage training: reconstruction pretrain -> adversarial "
             "fine-tune). Optimiser, scheduler, epoch counter and discriminator "
             "all start fresh. Ignored once this run has its own checkpoint, so "
             "it composes with --auto-resume.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_adversarial = cfg.loss.use_adversarial

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else "cpu" if args.device == "auto"
        else args.device
    )

    print(f"Device:          {device}")
    print(f"Input mode:      {cfg.data.input_mode}")
    print(f"Adversarial GAN: {use_adversarial}")
    print(f"HF cutoff:       {HF_CUTOFF} Hz")

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
        cfg.data.index_path, split="train",
        input_mode=cfg.data.input_mode,
        val_fraction=cfg.data.val_fraction, seed=cfg.data.seed,
        in_memory=cfg.data.in_memory,
    )
    val_set = LatentBWEDataset(
        cfg.data.index_path, split="val",
        input_mode=cfg.data.input_mode,
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

    generator = LatentBWENet(
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks,
        kernel_size=cfg.model.kernel_size,
        expansion=cfg.model.expansion,
        center=cfg.model.center,
    ).to(device)
    print(f"Generator parameters:     {count_parameters(generator):,}")
    print(f"Receptive field:          {generator.receptive_field_ms():.0f} ms")
    print(f"STFT center:              {cfg.model.center}")

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

    # Warm start only when this run has no checkpoint of its own: on a re-run
    # --auto-resume must win, or the job would silently restart stage 2 from
    # stage 1 and throw away the fine-tuning done so far.
    if args.init_generator and resume_path is None:
        init_generator_from(generator, Path(args.init_generator), device,
                            expect_input_mode=cfg.data.input_mode)
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

    # Run-level constants for the ablation grid (params / latency / RTF). RTF is
    # measured on a real val batch; guarded so a measurement hiccup can't abort.
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
            # sep = D(real)-D(fake): the number that says whether the critic is
            # learning at all. ~0 means it cannot separate real from generated.
            f" real {train_metrics['d_real']:.3f} fake {train_metrics['d_fake']:.3f}"
            f" sep {train_metrics['d_sep']:.3f}"
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

        if val_metrics["g_total"] < best_val_g:
            best_val_g = val_metrics["g_total"]
            ckpt = {
                "epoch":     epoch,
                "generator": generator.state_dict(),
                "val_loss":  val_metrics["g_total"],
                "config":    cfg.__dict__,
                "input_mode": cfg.data.input_mode,
            }
            if use_adversarial:
                ckpt["discriminator"] = discriminator.state_dict()
            torch.save(ckpt, out_dir / "best.pt")

        if epoch % cfg.output.save_every == 0:
            ckpt = {
                "epoch":      epoch,
                "generator":  generator.state_dict(),
                "opt_g":      opt_g.state_dict(),
                "sched_g":    sched_g.state_dict(),
                "best_val_g": best_val_g,
                "val_loss":   val_metrics["g_total"],
                "config":     cfg.__dict__,
                "input_mode": cfg.data.input_mode,
            }
            if use_adversarial:
                ckpt["discriminator"] = discriminator.state_dict()
                ckpt["opt_d"]         = opt_d.state_dict()
                ckpt["sched_d"]       = sched_d.state_dict()
            torch.save(ckpt, out_dir / f"epoch_{epoch:04d}.pt")

    tracker.finish()
    print(f"\nDone. Best val loss: {best_val_g:.4f}")
    print(f"Checkpoints: {out_dir}")


if __name__ == "__main__":
    main()
