"""
Losses for BWE GAN training, following "Fast and Flexible Audio Bandwidth
Extension via Vocos" (2603.07285).

    L_G = lambda_mel * L_mel + lambda_hf * L_hf_mrstft
          + lambda_adv * L_adv + lambda_fm * L_fm
    L_D = L_real + L_fake

Components:
    MultiResolutionDiscriminator  sub-discriminators over complex STFT at
                                  several resolutions, returning logits and
                                  intermediate features
    DiscriminatorLoss             hinge loss for the discriminator update
    GeneratorAdversarialLoss      adversarial loss for the generator update
    FeatureMatchingLoss           L1 on intermediate discriminator features
    MelReconstructionLoss         multi-scale mel spectrogram L1
    HFMRSTFTLoss                  MRSTFT on the high-passed signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import auraloss.freq as af
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Sharp spectral high-pass (brickwall), matched to the HF model's synthesis band
# ---------------------------------------------------------------------------

def spectral_highpass(
    x:            torch.Tensor,
    window:       torch.Tensor,
    n_fft:        int = 2560,
    hop_length:   int = 640,
    win_length:   int = 2560,
    hf_bin_start: int = 641,
    center:       bool = True,
) -> torch.Tensor:
    """
    Brickwall high-pass: STFT -> zero bins [0, hf_bin_start) -> iSTFT.

    Mirrors exactly how LatentBWENet synthesises (same n_fft/hop/win, same
    zeroed bins, same torch.istft framing), so the supervision band edge is
    bit-for-bit identical to the model's. This avoids the soft-biquad vs sharp-
    brickwall mismatch that piles energy up at the 12 kHz crossover.

    `center` MUST match the model's synthesis `center` (LatentBWENet.center /
    the audio model). Otherwise target and output are framed differently and
    the supervision misaligns by ~n_fft/2 samples.

    x: [B, 1, T] or [B, T]. Returns the same shape, band-limited to >= 12 kHz.
    """
    squeezed = x.dim() == 3
    if squeezed:
        x = x.squeeze(1)

    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=center, return_complex=True,
    )
    mask = torch.ones(spec.shape[-2], device=spec.device, dtype=spec.real.dtype)
    mask[:hf_bin_start] = 0.0
    spec = spec * mask[None, :, None]                 # out-of-place (autograd-safe)

    if center:
        y = torch.istft(
            spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=True, length=x.shape[-1],
        )
    else:
        # Same clamped causal overlap-add the model synthesises with, since
        # torch.istft rejects the Hann envelope at center=False. Lazy import
        # keeps this module free of a hard model dependency on center=True.
        from nac_bwe.models.latent_bwe_net import causal_istft
        y = causal_istft(spec, window, n_fft, hop_length, length=x.shape[-1])
    return y.unsqueeze(1) if squeezed else y


# ---------------------------------------------------------------------------
# STFT sub-discriminator
# ---------------------------------------------------------------------------

class DiscriminatorSTFT(nn.Module):
    """
    Single STFT sub-discriminator operating on the complex spectrogram
    at one resolution. Applies 2D convolutions over frequency × time.

    Returns (logit, feature_maps) where feature_maps is a list of
    intermediate activations used for feature matching loss.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_filters: int = 32,
        n_layers: int = 4,
    ):
        super().__init__()

        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

        # Input: 2 channels (real + imaginary)
        in_ch = 2
        layers = []
        for i in range(n_layers):
            out_ch = n_filters * (2 ** min(i, 3))   # cap channel growth at 8x
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch, out_ch,
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_ch = out_ch

        self.convs = nn.ModuleList(layers)
        self.final = nn.Conv2d(in_ch, 1, kernel_size=(3, 3), padding=(1, 1))

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, 1, T] -> [B, 2, freq, time]  (real and imag as channels)
        """
        B = x.shape[0]
        x = x.squeeze(1)    # [B, T]
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )   # [B, freq, time]
        # Stack real and imaginary as two channels: [B, 2, freq, time]
        return torch.stack([stft.real, stft.imag], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: [B, 1, T]
        Returns: (logit [B, 1, f', t'], feature_maps list)
        """
        x = self._stft(x)       # [B, 2, freq, time]
        feature_maps = []
        for conv in self.convs:
            x = conv(x)
            feature_maps.append(x)
        logit = self.final(x)
        return logit, feature_maps


# ---------------------------------------------------------------------------
# Multi-Resolution Discriminator
# ---------------------------------------------------------------------------

class MultiResolutionDiscriminator(nn.Module):
    """
    MRD: a set of STFT sub-discriminators at multiple resolutions.

    Default resolutions follow the VOCOS / BWE paper:
        (n_fft, hop_length, win_length)
        (1024, 256, 1024)
        (2048, 512, 2048)
        (512,  128, 512 )

    Returns a list of (logit, feature_maps) tuples, one per sub-discriminator.
    """

    def __init__(
        self,
        resolutions: List[Tuple[int, int, int]] = None,
        n_filters: int = 32,
        n_layers: int = 4,
    ):
        super().__init__()

        if resolutions is None:
            resolutions = [
                (1024, 256,  1024),
                (2048, 512,  2048),
                (512,  128,  512 ),
            ]

        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                n_filters=n_filters,
                n_layers=n_layers,
            )
            for n_fft, hop, win in resolutions
        ])

    def forward(
        self, x: torch.Tensor
    ) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        x: [B, 1, T]
        Returns: list of (logit, feature_maps) for each sub-discriminator
        """
        return [d(x) for d in self.discriminators]


# ---------------------------------------------------------------------------
# Discriminator loss (least-squares GAN)
# ---------------------------------------------------------------------------

class DiscriminatorLoss(nn.Module):
    """
    Least-squares GAN discriminator loss.
    Trains discriminator to output 1 for real, 0 for generated.

    L_D = mean((D(real) - 1)^2) + mean(D(fake)^2)
    """

    def forward(
        self,
        real_outputs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
        fake_outputs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (total_loss, loss_real, loss_fake)
        """
        loss_real = torch.tensor(0.0)
        loss_fake = torch.tensor(0.0)

        for (real_logit, _), (fake_logit, _) in zip(real_outputs, fake_outputs):
            loss_real = loss_real + F.mse_loss(
                real_logit, torch.ones_like(real_logit)
            )
            loss_fake = loss_fake + F.mse_loss(
                fake_logit, torch.zeros_like(fake_logit)
            )

        loss_real = loss_real / len(real_outputs)
        loss_fake = loss_fake / len(fake_outputs)
        return loss_real + loss_fake, loss_real, loss_fake


# ---------------------------------------------------------------------------
# Generator adversarial loss
# ---------------------------------------------------------------------------

class GeneratorAdversarialLoss(nn.Module):
    """
    Generator adversarial loss. Tries to fool the discriminator.

    L_adv = mean((D(fake) - 1)^2)
    """

    def forward(
        self,
        fake_outputs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for fake_logit, _ in fake_outputs:
            loss = loss + F.mse_loss(fake_logit, torch.ones_like(fake_logit))
        return loss / len(fake_outputs)


# ---------------------------------------------------------------------------
# Feature matching loss
# ---------------------------------------------------------------------------

class FeatureMatchingLoss(nn.Module):
    """
    L1 distance between intermediate discriminator features of real
    and generated audio, averaged over all layers and sub-discriminators.

    Provides a perceptual-style loss signal without needing a separate
    pretrained network. Stabilises GAN training significantly.
    """

    def forward(
        self,
        real_outputs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
        fake_outputs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0)
        n    = 0
        for (_, real_fmaps), (_, fake_fmaps) in zip(real_outputs, fake_outputs):
            for real_f, fake_f in zip(real_fmaps, fake_fmaps):
                loss = loss + F.l1_loss(fake_f, real_f.detach())
                n   += 1
        return loss / max(n, 1)


# ---------------------------------------------------------------------------
# Mel reconstruction loss
# ---------------------------------------------------------------------------

class MelReconstructionLoss(nn.Module):
    """
    Multi-scale mel spectrogram L1 loss.
    Computes mel spectrograms at multiple scales and averages the L1 distances.

    Perceptually motivated, since the mel scale weights lower frequencies more heavily,
    matching human auditory perception. Standard in vocoder literature.

    Scales chosen so n_fft is large enough to avoid zero mel filterbanks:
    with n_mels=80 at 48kHz we need n_fft >= 1024 to have enough frequency bins.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: int = 80,
        scales: List[Tuple[int, int]] = None,   # (n_fft, hop_length) pairs
    ):
        super().__init__()

        if scales is None:
            # Minimum n_fft=1024 to avoid zero filterbanks with n_mels=80 at 48kHz
            scales = [
                (1024, 240),
                (2048, 480),
                (4096, 960),
            ]

        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop,
                n_mels=n_mels,
                power=1.0,   # magnitude spectrogram
            )
            for n_fft, hop in scales
        ])

    def to(self, *args, **kwargs):
        # Override to ensure mel filterbanks move to device with the module
        super().to(*args, **kwargs)
        for t in self.mel_transforms:
            t.to(*args, **kwargs)
        return self

    def forward(
        self, x_enhanced: torch.Tensor, x_clean: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=x_enhanced.device)
        for mel_transform in self.mel_transforms:
            mel_enh   = mel_transform(x_enhanced.squeeze(1))
            mel_clean = mel_transform(x_clean.squeeze(1))
            mel_enh   = torch.log(mel_enh   + 1e-5)
            mel_clean = torch.log(mel_clean + 1e-5)
            loss = loss + F.l1_loss(mel_enh, mel_clean)
        return loss / len(self.mel_transforms)


# ---------------------------------------------------------------------------
# HF MRSTFT loss
# ---------------------------------------------------------------------------

class HFMRSTFTLoss(nn.Module):
    """
    MRSTFT loss computed on the high-passed signal above cutoff_hz.

    This directly supervises the frequency region the model is predicting.
    Without this, mel loss alone can be dominated by LF energy and the
    model can achieve low loss while producing poor HF content.

    Key addition in the BWE paper over vanilla VOCOS.
    """

    def __init__(
        self,
        sample_rate:  int   = 48000,
        cutoff_hz:    float = 12000.0,
        n_fft:        int   = 2560,
        hop_length:   int   = 640,
        win_length:   int   = 2560,
        hf_bin_start: int   = 641,
        center:       bool  = True,
    ):
        super().__init__()
        self.sample_rate  = sample_rate
        self.cutoff_hz    = cutoff_hz
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.win_length   = win_length
        self.hf_bin_start = hf_bin_start
        self.center       = center
        self.register_buffer("hp_window", torch.hann_window(win_length))
        self.mrstft = af.MultiResolutionSTFTLoss(
            fft_sizes=[256, 512, 1024],
            hop_sizes=[64,  128,  256],
            win_lengths=[256, 512, 1024],
            w_sc=0.0,
            w_log_mag=1.0,
            w_lin_mag=0.0,
            sample_rate=sample_rate,
        )

    def _hf(self, x: torch.Tensor) -> torch.Tensor:
        return spectral_highpass(
            x, self.hp_window, self.n_fft, self.hop_length,
            self.win_length, self.hf_bin_start, self.center,
        )

    def forward(
        self, x_enhanced: torch.Tensor, x_clean: torch.Tensor
    ) -> torch.Tensor:
        # Sharp brickwall on both, using the model's edge rather than a soft biquad,
        # which would re-introduce the 12 kHz crossover pile-up.
        return self.mrstft(self._hf(x_enhanced), self._hf(x_clean))


# ---------------------------------------------------------------------------
# Combined generator loss (convenience wrapper)
# ---------------------------------------------------------------------------

class GeneratorLoss(nn.Module):
    """
    Full generator loss following the BWE paper formulation:

        L_G = λ_mel * L_mel + λ_hf * L_hf + λ_adv * L_adv + λ_fm * L_fm

    Returns a dict of all loss components for logging.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        lambda_mel: float = 45.0,
        lambda_hf:  float = 1.0,
        lambda_adv: float = 1.0,
        lambda_fm:  float = 2.0,
        center:     bool  = True,
    ):
        super().__init__()
        self.lambda_mel = lambda_mel
        self.lambda_hf  = lambda_hf
        self.lambda_adv = lambda_adv
        self.lambda_fm  = lambda_fm

        self.mel_loss  = MelReconstructionLoss(sample_rate=sample_rate)
        self.hf_loss   = HFMRSTFTLoss(sample_rate=sample_rate, center=center)
        self.adv_loss  = GeneratorAdversarialLoss()
        self.fm_loss   = FeatureMatchingLoss()

    def forward(
        self,
        x_enhanced: torch.Tensor,
        x_clean: torch.Tensor,
        fake_outputs,   # None when use_adversarial=False
        real_outputs,   # None when use_adversarial=False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns: (total_loss, loss_dict)
        loss_dict contains individual components for TensorBoard logging.

        Pass fake_outputs=None and real_outputs=None for reconstruction-only
        training (use_adversarial=False). Adversarial terms are skipped.
        """
        l_mel = self.mel_loss(x_enhanced, x_clean)
        l_hf  = self.hf_loss(x_enhanced, x_clean)

        total = self.lambda_mel * l_mel + self.lambda_hf * l_hf

        loss_dict = {
            "mel":   l_mel.item(),
            "hf":    l_hf.item(),
            "total": 0.,   # filled below
        }

        if fake_outputs is not None and real_outputs is not None:
            l_adv = self.adv_loss(fake_outputs)
            l_fm  = self.fm_loss(real_outputs, fake_outputs)
            total = total + self.lambda_adv * l_adv + self.lambda_fm * l_fm
            loss_dict["adv"] = l_adv.item()
            loss_dict["fm"]  = l_fm.item()

        loss_dict["total"] = total.item()
        return total, loss_dict


def count_discriminator_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)