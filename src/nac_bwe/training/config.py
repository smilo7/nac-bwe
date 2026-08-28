"""
Training config dataclasses, shared by the latent- and audio-conditioned
trainers.

They live in their own module rather than beside a trainer because checkpoints
pickle them by module path: a module that is never run as ``__main__`` gives
them a stable path (``nac_bwe.training.config.ModelConfig``) that survives
moving or renaming the training entry points.
"""

from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class DataConfig:
    index_path:   str
    input_mode:   str   = "quantized"
    val_fraction: float = 0.1
    seed:         int   = 42
    in_memory:    bool  = False
    num_workers:  int   = 4     # DataLoader workers (used only when in_memory=False)


@dataclass
class ModelConfig:
    hidden_dim:  int = 256
    num_blocks:  int = 4
    kernel_size: int = 7
    expansion:   int = 4
    # STFT synthesis framing. False gives causal framing matching
    # StreamingISTFT at inference. Applied to model, losses and target alike.
    # Defaults to True when absent from a checkpoint.
    center:      bool = True


@dataclass
class DiscriminatorConfig:
    n_filters:   int             = 32
    n_layers:    int             = 4
    resolutions: List[List[int]] = field(default_factory=lambda: [
        [1024, 256,  1024],
        [2048, 512,  2048],
        [512,  128,  512 ],
    ])


@dataclass
class LossConfig:
    use_adversarial: bool  = True
    lambda_mel:      float = 0.0    # mel on HF-only signal is not meaningful
    lambda_hf:       float = 15.0
    lambda_adv:      float = 5.0
    lambda_fm:       float = 2.0


@dataclass
class TrainingConfig:
    epochs:           int   = 100
    batch_size:       int   = 16
    lr_generator:     float = 2e-4
    lr_discriminator: float = 5e-5
    weight_decay:     float = 1e-4
    grad_clip:        float = 5.0


@dataclass
class OutputConfig:
    output_dir: str = "runs/latent_bwe_hf"
    save_every: int = 10


@dataclass
class WandbConfig:
    """Experiment-tracking config for ExperimentTracker.

    mode:
        "online"    stream to wandb.ai, needs ``wandb login`` once.
        "offline"   write under ``<output_dir>/wandb``, sync later.
        "disabled"  CSV only.

    ``WANDB_MODE`` overrides this field. Media is always saved under
    ``<output_dir>/samples``; ``log_media_to_wandb`` also uploads a few clips.
    """
    enabled:             bool       = True
    project:             str        = "nac-bwe"
    entity:              str | None = None
    mode:                str        = "online"    # online | offline | disabled
    tags:                List[str]  = field(default_factory=list)
    group:               str | None = None         # W&B group; set per sweep for ablation grids
    run_name:            str | None = None         # defaults to output_dir basename
    log_media:           bool       = True         # save val audio/spectrograms locally
    log_media_to_wandb:  bool       = False        # also upload media to W&B (quota!)
    log_media_every:     int        = 10           # epochs between media dumps
    num_media_samples:   int        = 3


@dataclass
class TrainConfig:
    data:          DataConfig
    model:         ModelConfig
    discriminator: DiscriminatorConfig
    loss:          LossConfig
    training:      TrainingConfig
    output:        OutputConfig
    wandb:         WandbConfig = field(default_factory=WandbConfig)


def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return TrainConfig(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        discriminator=DiscriminatorConfig(**raw["discriminator"]),
        loss=LossConfig(**raw["loss"]),
        training=TrainingConfig(**raw["training"]),
        output=OutputConfig(**raw["output"]),
        # Optional block, defaults apply when absent.
        wandb=WandbConfig(**raw.get("wandb", {})),
    )
