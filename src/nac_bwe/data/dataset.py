"""
Loads precomputed (model_input, x_clean_48khz) pairs produced by
``nac_bwe.data.precompute``.

input_mode selects the conditioning representation:

    "continuous"  pre-RVQ encoder latents.
    "quantized"   post-RVQ dequantized latents, what a code generator emits.
    "mixed"       picks continuous or quantized at random per sample.
    "audio"       EnCodec-decoded 24 kHz audio, for AudioBWENet.

The "audio" mode returns the decode of the same quantized latents the
"quantized" mode returns, so both models can train on identical chunks.

Each item is (input, x_clean):

    input     [D, T_frames] float32 for latent modes,
              [1, T_24k]    float32 for "audio"
    x_clean   [1, T_48k]    float32, the training target

Usage:
    dataset = LatentBWEDataset(
        index_path="data/precomputed_latent_bwe/index.json",
        split="train",
        input_mode="quantized",
    )
"""

import json
import random
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class LatentBWEDataset(Dataset):
    """
    Args:
        index_path:   Path to index.json from precompute_latent.py
        split:        "train", "val", or "all"
        input_mode:   "continuous", "quantized", or "mixed"
        val_fraction: Fraction reserved for validation
        seed:         Fixed seed for reproducible split
        in_memory:    Load all tensors into RAM at init (fast for small datasets)
    """

    def __init__(
        self,
        index_path: str,
        split: Literal["train", "val", "all"] = "train",
        input_mode: Literal["continuous", "quantized", "mixed", "audio"] = "quantized",
        val_fraction: float = 0.1,
        seed: int = 42,
        in_memory: bool = False,
    ):
        self.index_path = Path(index_path)
        self.root       = self.index_path.parent
        self.input_mode = input_mode
        self.in_memory  = in_memory
        self._rng       = random.Random(seed + 1)

        with open(self.index_path) as f:
            meta = json.load(f)

        self.encodec_sr        = meta["encodec_sr"]
        self.output_sr         = meta["output_sr"]
        self.chunk_samples_48k = meta["chunk_samples_48k"]
        self.chunk_samples_24k = meta["chunk_samples_24k"]
        self.bandwidth         = meta["bandwidth"]

        all_samples = meta["samples"]

        rng     = random.Random(seed)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)
        val_size = max(1, int(len(indices) * val_fraction))

        if split == "train":
            self.samples = [all_samples[i] for i in indices[val_size:]]
        elif split == "val":
            self.samples = [all_samples[i] for i in indices[:val_size]]
        elif split == "all":
            self.samples = all_samples
        else:
            raise ValueError(f"split must be 'train', 'val', or 'all', got '{split}'")

        print(
            f"LatentBWEDataset | {split} | "
            f"input_mode={input_mode} | "
            f"bandwidth={self.bandwidth}kbps | "
            f"{len(self.samples)} chunks"
        )

        if self.in_memory:
            self._cache = self._build_cache()

    def _load_latents(self, entry: dict) -> torch.Tensor:
        """Load the model input for this entry per input_mode.

        Returns latents [D, T_frames] for the latent modes, or the reconstructed
        24 kHz waveform [1, T_24k] for "audio".
        """
        if self.input_mode == "continuous":
            key = "latents"
        elif self.input_mode == "quantized":
            key = "latents_quantized"
        elif self.input_mode == "audio":
            key = "reconstructed"
        else:  # mixed
            key = "latents" if self._rng.random() < 0.5 else "latents_quantized"
        return torch.load(self.root / entry[key], weights_only=True)

    def _build_cache(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        print(f"  Loading {len(self.samples)} chunks into memory...")
        cache = []
        for entry in tqdm(self.samples, desc="  Caching", leave=False):
            latents = self._load_latents(entry)
            x_clean = torch.load(self.root / entry["clean"], weights_only=True)
            cache.append((latents, x_clean))

        latent_dim = cache[0][0].shape[0] if cache else 128
        t_frames   = cache[0][0].shape[1] if cache else 75
        bytes_per  = (latent_dim * t_frames + self.chunk_samples_48k) * 4
        print(f"  Cache built. Memory: ~{len(cache) * bytes_per / 1e6:.0f} MB")
        return cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.in_memory:
            return self._cache[idx]

        entry   = self.samples[idx]
        latents = self._load_latents(entry)
        x_clean = torch.load(self.root / entry["clean"], weights_only=True)
        return latents, x_clean

    def get_metadata(self, idx: int) -> dict:
        return self.samples[idx]
