"""
Audio-conditioned bandwidth extension.

AudioBWENet predicts the same 12-24 kHz STFT band as LatentBWENet but
conditions on EnCodec's decoded 24 kHz audio rather than its latents. It
subclasses LatentBWENet and replaces only the input stem:

    24 kHz audio -> resample to 48 kHz
                 -> STFT (n_fft=2560, hop=640, win=2560)
                 -> log-magnitude of bins 0..640 (0-12 kHz)
                 -> Linear(641, hidden_dim)

The analysis STFT follows the same ``center`` flag as synthesis, so with
center=False the features depend only on past samples.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from nac_bwe.models.latent_bwe_net import (
    LatentBWENet, causal_istft,
    ENCODEC_SR, SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH,
    HF_BIN_START, N_HF_BINS, N_BINS_FULL,
    count_parameters,
)

# LF bins fed as input: 0..640 inclusive (0-12 kHz).
N_LF_BINS = HF_BIN_START   # 641


class AudioBWENet(LatentBWENet):
    """
    Predicts HF STFT coefficients (bins 641-1280, 12-24 kHz) from EnCodec's
    decoded 24 kHz audio.

    Args:
        hidden_dim:  backbone width.
        num_blocks:  number of ConvNeXt blocks.
        kernel_size: depthwise conv kernel.
        expansion:   pointwise expansion factor.
        center:      STFT framing. False gives causal analysis and synthesis.

    Methods:
        forward(x_recon_24k):       HF-only waveform [B, 1, T_48k].
        forward_stft(x_recon_24k):  complex HF STFT  [B, N_HF_BINS, T].

    Combine the result with an LFExtractor's low band using
    ``LFExtractor.combine_time_domain``.
    """

    def __init__(
        self,
        hidden_dim:  int = 256,
        num_blocks:  int = 8,
        kernel_size: int = 7,
        expansion:   int = 4,
        center:      bool = True,
    ):
        super().__init__(hidden_dim, num_blocks, kernel_size, expansion, center)
        # Input projection takes LF log-magnitude (641 bins) in place of the
        # latent model's Linear(128, hidden). Everything else is inherited.
        self.input_proj = nn.Linear(N_LF_BINS, hidden_dim)

    # -- input stem: recon audio to LF log-mag features ---------------------

    def _audio_to_features(self, x_recon_24k: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        recon 24 kHz audio [B, 1, T_24k] (or [B, T_24k])
          -> LF log-magnitude features [B, N_LF_BINS, T], and the 48 kHz length L.

        L is returned so synthesis can be trimmed back to the input duration.
        """
        if x_recon_24k.dim() == 3:
            x_recon_24k = x_recon_24k.squeeze(1)

        x48 = torchaudio.functional.resample(x_recon_24k, ENCODEC_SR, SAMPLE_RATE)  # [B, L]
        L = x48.shape[-1]

        if not self.center:
            # center=False does not pad, so right-pad to cover L and let the
            # causal iSTFT emit L samples back.
            T = max(1, math.ceil((L - N_FFT) / HOP_LENGTH) + 1)
            L_pad = (T - 1) * HOP_LENGTH + N_FFT
            if L_pad > L:
                x48 = F.pad(x48, (0, L_pad - L))

        return self._features_from_48k(x48), L

    def _features_from_48k(self, x48: torch.Tensor) -> torch.Tensor:
        """48 kHz audio [B, L] -> LF log-magnitude features [B, N_LF_BINS, T]."""
        spec = torch.stft(
            x48, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
            window=self.window, center=self.center, return_complex=True,
        )                                              # [B, N_BINS_FULL, T]
        lf = spec[:, :N_LF_BINS, :]                    # [B, 641, T]  (0-12 kHz)
        return torch.log(lf.abs() + 1e-5)

    # -- streaming analysis (carried STFT state, center=False) --------------

    def _stream_features(
        self, x48_block: torch.Tensor, buf: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Causal block-wise version of _features_from_48k for center=False.

        Prepends `buf`, the unconsumed tail of under n_fft samples from the
        previous call, extracts every full frame and returns the new tail.
        Concatenating
        the per-block features is identical to _features_from_48k over the whole
        stream (no center padding, so frame i looks only backward).

        x48_block: [B, T_block]
        buf:       [B, carry]  (None = cold start)
        returns (features [B, N_LF_BINS, n_frames], new_buf [B, carry])
        """
        if x48_block.dim() == 3:
            x48_block = x48_block.squeeze(1)
        x = x48_block if buf is None else torch.cat([buf, x48_block], dim=-1)

        n = x.shape[-1]
        n_frames = 0 if n < N_FFT else (n - N_FFT) // HOP_LENGTH + 1
        if n_frames == 0:
            empty = x.new_zeros(x.shape[0], N_LF_BINS, 0)
            return empty, x

        consumed = n_frames * HOP_LENGTH
        # Frame up to the last full frame; carry everything from `consumed` on.
        spec = torch.stft(
            x[:, : (n_frames - 1) * HOP_LENGTH + N_FFT],
            n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
            window=self.window, center=False, return_complex=True,
        )
        feats = torch.log(spec[:, :N_LF_BINS, :].abs() + 1e-5)
        return feats, x[:, consumed:]

    def forward_stft_streaming(self, x48_block: torch.Tensor, states=None):
        """
        Streaming inference from a block of 48 kHz audio [B, T_block].

        states = {"analysis": tail buffer, "backbone": per-block conv caches}
        (None = cold start). Returns (hf_stft [B, N_HF_BINS, n_frames], states).

        Resampling 24->48 kHz is left to an external streaming resampler (e.g.
        a stateful resampler), as the LF path is, so this method takes 48 kHz
        blocks. Requires center=False.

        Block-wise outputs concatenate to the same result as forward_stft() on
        the whole signal. See test_audio_streaming.py.
        """
        if self.center:
            raise RuntimeError("streaming requires center=False (causal framing)")
        if states is None:
            states = {"analysis": None, "backbone": None}

        feats, new_buf = self._stream_features(x48_block, states["analysis"])
        if feats.shape[-1] == 0:
            empty = feats.new_zeros(feats.shape[0], N_HF_BINS, 0,
                                    dtype=torch.complex64)
            return empty, {"analysis": new_buf, "backbone": states["backbone"]}

        real, imag, new_bb = self._backbone_streaming(feats, states["backbone"])
        return torch.complex(real, imag), {"analysis": new_buf, "backbone": new_bb}

    # -- forward paths (convert audio to features first) --------------------

    def forward(self, x_recon_24k: torch.Tensor) -> torch.Tensor:
        """HF-only waveform [B, 1, L_48k] for training (LF bins zeroed)."""
        feats, L = self._audio_to_features(x_recon_24k)
        real, imag = self._backbone(feats)             # [B, N_HF_BINS, T] each
        B, _, T = real.shape

        lf_zeros  = torch.zeros(B, HF_BIN_START, T, device=real.device)
        full      = torch.complex(
            torch.cat([lf_zeros, real], dim=1),
            torch.cat([lf_zeros, imag], dim=1),
        )

        if self.center:
            audio = torch.istft(
                full, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                window=self.window, length=L, center=True,
            )
        else:
            audio = causal_istft(full, self.window, N_FFT, HOP_LENGTH, length=L)
        return audio.unsqueeze(1)

    def forward_stft(self, x_recon_24k: torch.Tensor) -> torch.Tensor:
        """Complex HF STFT coefficients [B, N_HF_BINS, T] for inference."""
        feats, _ = self._audio_to_features(x_recon_24k)
        real, imag = self._backbone(feats)
        return torch.complex(real, imag)


if __name__ == "__main__":
    for center in (True, False):
        model = AudioBWENet(hidden_dim=256, num_blocks=4, center=center)
        dummy = torch.randn(2, 1, 24000)              # 1 s of 24 kHz recon audio
        waveform = model(dummy)
        stft     = model.forward_stft(dummy)
        print(
            f"center={center} | params {count_parameters(model):,} | "
            f"waveform {tuple(waveform.shape)} | stft {tuple(stft.shape)}"
        )
    print(f"Input LF bins: {N_LF_BINS} (0..{N_LF_BINS - 1}) | "
          f"HF bins {HF_BIN_START}..{N_BINS_FULL - 1} ({N_HF_BINS})")