"""
Latent-conditioned bandwidth extension.

LatentBWENet predicts STFT bins 641-1280 (12-24 kHz) from EnCodec latents and
synthesises them at 48 kHz. The 0-12 kHz band comes from a frozen decoder via
LFExtractor, and the two waveforms are summed.

The STFT grids are aligned so bin k is the same frequency in both:

    Vocos, 24 kHz:      n_fft=1280, hop=320  ->  18.75 Hz/bin, 75 frames/s
    this model, 48 kHz: n_fft=2560, hop=640  ->  18.75 Hz/bin, 75 frames/s
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

ENCODEC_SR = 24000   # Vocos / EnCodec native rate for the LF path

SAMPLE_RATE  = 48000
LATENT_DIM   = 128

N_FFT        = 2560
HOP_LENGTH   = 640
WIN_LENGTH   = 2560
N_BINS_FULL  = N_FFT // 2 + 1        # 1281 bins, 0-24 kHz
HF_BIN_START = 641                    # 641 * 18.75 Hz = 12 018.75 Hz ≈ 12 kHz
N_HF_BINS    = N_BINS_FULL - HF_BIN_START   # 640 bins, 12-24 kHz


# ---------------------------------------------------------------------------
# Causal ConvNeXt backbone
# ---------------------------------------------------------------------------

class CausalConvNeXtBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 7, expansion: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.dw_conv  = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                   padding=0, groups=hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)
        self.pw_conv1 = nn.Linear(hidden_dim, hidden_dim * expansion)
        self.act      = nn.GELU()
        self.pw_conv2 = nn.Linear(hidden_dim * expansion, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = x.transpose(1, 2)
        return x + residual

    def forward_streaming(self, x: torch.Tensor, state: torch.Tensor | None):
        """
        Streaming counterpart of forward(), prepending the previous call's
        trailing frames instead of zero-padding the left edge. Only the
        depthwise conv carries state, the rest is frame-local.

        Args:
            x:     [B, C, T_block]
            state: [B, C, kernel_size-1], None for a cold start.

        Returns:
            (out [B, C, T_block], new_state [B, C, kernel_size-1])
        """
        residual = x
        pad = self.kernel_size - 1
        if state is None:
            state = x.new_zeros(x.shape[0], x.shape[1], pad)
        x = torch.cat([state, x], dim=-1)
        new_state = x[..., -pad:] if pad > 0 else state
        x = self.dw_conv(x)                  # no internal padding now
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = x.transpose(1, 2)
        return x + residual, new_state


# ---------------------------------------------------------------------------
# HF model
# ---------------------------------------------------------------------------

class LatentBWENet(nn.Module):
    """
    Predicts HF STFT coefficients (bins 641-1280, 12-24 kHz) from EnCodec
    latents.

    Args:
        hidden_dim:  backbone width.
        num_blocks:  number of ConvNeXt blocks.
        kernel_size: depthwise conv kernel.
        expansion:   pointwise expansion factor.
        center:      STFT framing. False gives causal synthesis.

    Methods:
        forward(latents):       HF-only waveform [B, 1, T_48k].
        forward_stft(latents):  complex HF STFT  [B, N_HF_BINS, T].
    """

    def __init__(
        self,
        hidden_dim:  int = 256,
        num_blocks:  int = 8,
        kernel_size: int = 7,
        expansion:   int = 4,
        center:      bool = True,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_blocks  = num_blocks
        self.kernel_size = kernel_size
        # STFT framing for the synthesis iSTFT. center=True adds n_fft/2
        # lookahead, center=False matches the causal StreamingISTFT used at
        # inference. Not a learned parameter, but a checkpoint must be run with
        # the value it was trained with, and spectral_highpass must match.
        self.center      = center

        self.register_buffer("window", torch.hann_window(WIN_LENGTH))

        self.input_proj  = nn.Linear(LATENT_DIM, hidden_dim)
        self.input_norm  = nn.LayerNorm(hidden_dim)
        self.blocks      = nn.Sequential(
            *[CausalConvNeXtBlock(hidden_dim, kernel_size, expansion)
              for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, N_HF_BINS * 2)

    def _stem(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents [B, D, T] -> backbone input [B, hidden_dim, T] (frame-local)."""
        h = latents.transpose(1, 2)
        h = self.input_proj(h)
        h = self.input_norm(h)
        return h.transpose(1, 2)

    def _to_complex(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Backbone output [B, hidden_dim, T] -> (real, imag) each [B, N_HF_BINS, T]."""
        h = h.transpose(1, 2)
        out = self.output_proj(h)
        log_mag, phase = out.chunk(2, dim=-1)     # [B, T, N_HF_BINS] each
        log_mag = log_mag.transpose(1, 2)
        phase   = phase.transpose(1, 2)

        # Vocos ISTFTHead parameterisation: magnitude via clamped exp,
        # unit-circle phase via cos/sin, giving
        # S = mag * (cos(phase) + j*sin(phase)).
        mag  = torch.exp(log_mag).clamp(max=1e2)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        return real, imag

    def _backbone(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Latents [B, D, T] -> (pred_real, pred_imag), each [B, N_HF_BINS, T]."""
        h = self._stem(latents)
        h = self.blocks(h)
        return self._to_complex(h)

    def _backbone_streaming(self, latents: torch.Tensor, states):
        """
        Streaming counterpart of _backbone(). `states` is a list of per-block
        caches (or None for cold start); returns (real, imag, new_states).
        """
        h = self._stem(latents)
        if states is None:
            states = [None] * len(self.blocks)
        new_states = []
        for block, st in zip(self.blocks, states):
            h, ns = block.forward_streaming(h, st)
            new_states.append(ns)
        real, imag = self._to_complex(h)
        return real, imag, new_states

    def forward_stft(self, latents: torch.Tensor) -> torch.Tensor:
        """Returns complex HF STFT coefficients [B, N_HF_BINS, T] for inference."""
        real, imag = self._backbone(latents)
        return torch.complex(real, imag)

    def forward_stft_streaming(self, latents: torch.Tensor, states=None):
        """
        Streaming inference: process a block of latents [B, D, T_block] given
        per-block conv caches `states` (None = cold start). Returns
        (hf_stft [B, N_HF_BINS, T_block], new_states).

        Feeding the full sequence in blocks of any size and concatenating the
        outputs is bit-identical (to float tolerance) to forward_stft() on the
        whole sequence. See tests/test_hf_streaming.py.
        """
        real, imag, new_states = self._backbone_streaming(latents, states)
        return torch.complex(real, imag), new_states

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Returns HF-only waveform [B, 1, T_48k] for training.
        LF bins are zeroed so the output is purely above 12 kHz.
        """
        B, D, T = latents.shape
        target_length = T * HOP_LENGTH

        real, imag = self._backbone(latents)

        lf_zeros  = torch.zeros(B, HF_BIN_START, T, device=real.device)
        full_real = torch.cat([lf_zeros, real], dim=1)
        full_imag = torch.cat([lf_zeros, imag], dim=1)
        full      = torch.complex(full_real, full_imag)

        if self.center:
            audio = torch.istft(
                full, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                window=self.window, length=target_length, center=True,
            )
        else:
            # center=False: clamped causal overlap-add, since torch.istft
            # rejects a Hann envelope that reaches zero. Matches
            # StreamingISTFT.
            audio = causal_istft(
                full, self.window, N_FFT, HOP_LENGTH, length=target_length,
            )
        return audio.unsqueeze(1)

    def receptive_field_ms(self) -> float:
        frames = self.num_blocks * (self.kernel_size - 1) + 1
        return frames * (1000 / 75)


# ---------------------------------------------------------------------------
# Offline causal iSTFT (center=False)
# ---------------------------------------------------------------------------

def causal_istft(
    spec:       torch.Tensor,
    window:     torch.Tensor,
    n_fft:      int = N_FFT,
    hop_length: int = HOP_LENGTH,
    length:     int | None = None,
) -> torch.Tensor:
    """
    Differentiable overlap-add iSTFT with center=False framing and clamped
    window normalisation.

    torch.istft(center=False) cannot be used here. The Hann window is zero at
    its endpoints, so the overlap-add envelope hits zero at the signal edges and
    torch.istft rejects it, requiring a non-zero envelope everywhere. Clamping
    the denominator instead tolerates the edges and matches StreamingISTFT,
    keeping training-time synthesis consistent with inference.

    spec:   complex [B, n_bins, T]
    window: [<=n_fft]  (centre-padded to n_fft to match torch's convention)
    length: samples to return from the start (default (T-1)*hop + n_fft)
    returns real waveform [B, length]
    """
    B, _, T = spec.shape
    if window.numel() < n_fft:
        pad = n_fft - window.numel()
        window = F.pad(window, (pad // 2, pad - pad // 2))

    frames = torch.fft.irfft(spec.transpose(1, 2), n=n_fft, dim=-1)   # [B, T, n_fft]
    frames = frames * window
    out_len = (T - 1) * hop_length + n_fft

    # Overlap-add numerator and window-squared denominator via fold.
    num = F.fold(
        frames.transpose(1, 2), output_size=(out_len, 1),
        kernel_size=(n_fft, 1), stride=(hop_length, 1),
    ).reshape(B, out_len)
    win_sq = (window ** 2).view(1, n_fft, 1).expand(1, n_fft, T)
    den = F.fold(
        win_sq, output_size=(out_len, 1),
        kernel_size=(n_fft, 1), stride=(hop_length, 1),
    ).reshape(out_len)

    y = num / den.clamp_min(1e-8)
    return y[:, :length] if length is not None else y


# ---------------------------------------------------------------------------
# Streaming iSTFT (overlap-add with carried state)
# ---------------------------------------------------------------------------

class StreamingISTFT:
    """
    Causal block-wise iSTFT reproducing torch.istft(center=False) over a
    stream.

    Each block accumulates a windowed-frame numerator and a window-squared
    denominator, adds the tail carried from the previous block, emits the
    samples no future frame can touch, and carries the remaining n_fft-hop
    samples.

    Use one instance per stream and call reset() to start a new one.
    """

    def __init__(self, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                 window: torch.Tensor | None = None):
        self.n_fft   = n_fft
        self.hop     = hop_length
        self.overlap = n_fft - hop_length
        win = window if window is not None else torch.hann_window(win_length)
        # Pad/center the window to n_fft exactly as torch.istft does.
        if win.numel() < n_fft:
            pad = (n_fft - win.numel())
            win = F.pad(win, (pad // 2, pad - pad // 2))
        self.window = win
        self.win_sq = win ** 2
        self.reset()

    def reset(self):
        self._num_tail = None   # [B, overlap]
        self._den_tail = None   # [overlap]

    @torch.no_grad()
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        spec: complex [B, n_bins, T]  (full-band, n_bins == n_fft//2 + 1)
        returns real waveform [B, T*hop] for this block.
        """
        B, _, T = spec.shape
        window = self.window.to(spec.device)
        win_sq = self.win_sq.to(spec.device)

        # complex frames to time frames [B, T, n_fft], apply synthesis window
        frames = torch.fft.irfft(spec.transpose(1, 2), n=self.n_fft, dim=-1)
        frames = frames * window

        out_len = (T - 1) * self.hop + self.n_fft
        num = frames.new_zeros(B, out_len)
        den = frames.new_zeros(out_len)
        for t in range(T):
            s = t * self.hop
            num[:, s:s + self.n_fft] += frames[:, t, :]
            den[s:s + self.n_fft]    += win_sq

        if self._num_tail is not None:
            num[:, :self.overlap] += self._num_tail
            den[:self.overlap]    += self._den_tail

        emit = T * self.hop
        self._num_tail = num[:, emit:].clone()
        self._den_tail = den[emit:].clone()

        return num[:, :emit] / den[:emit].clamp_min(1e-8)


# ---------------------------------------------------------------------------
# LF extractors (pluggable backends)
# ---------------------------------------------------------------------------

class LFExtractor(ABC):
    """
    Common interface for the LF band (0-12 kHz): decode EnCodec codes to a
    48 kHz waveform, band-limited to < 12 kHz so it sums cleanly with the HF
    model's 12-24 kHz output. Backends differ only in the decoder used (Vocos
    vs EnCodec's own decoder). Construct directly or via make_lf_extractor().
    """

    @abstractmethod
    def decode_lf_audio(
        self,
        codes:        torch.Tensor,
        bandwidth_id: int = 3,
        target_sr:    int = SAMPLE_RATE,
    ) -> torch.Tensor:
        """codes -> LF waveform [B, T_48k], band-limited to < 12 kHz."""

    @staticmethod
    def combine_time_domain(lf_audio: torch.Tensor, hf_audio: torch.Tensor) -> torch.Tensor:
        """
        Sum the LF and HF 48 kHz waveforms, each [B, T]. Lengths are trimmed
        to the shorter of the two.
        """
        n = min(lf_audio.shape[-1], hf_audio.shape[-1])
        return lf_audio[..., :n] + hf_audio[..., :n]


class VocosLFExtractor(LFExtractor):
    """
    Synthesises the 0-12 kHz band with the pretrained Vocos EnCodec 24 kHz
    model. Higher quality than EnCodec's own decoder at the same bitrate. See
    EncodecLFExtractor for the lighter, natively streamable alternative.

    Do not combine this band with the HF band in the STFT domain. Vocos uses a
    custom iSTFT with padding="same", a different overlap-add convention from
    torch.istft. Passing its spectrum through torch.istft preserves the energy
    spectrum but scrambles the waveform, measured at roughly 0 dB SNR against
    vocos.decode where Vocos's own iSTFT gives over 140 dB. Each band is
    reconstructed with its own iSTFT and summed in the time domain instead.
    """

    def __init__(self, vocos_model, device: str = "cpu"):
        self.vocos  = vocos_model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def decode_lf_audio(
        self,
        codes:        torch.Tensor,
        bandwidth_id: int = 3,
        target_sr:    int = SAMPLE_RATE,
    ) -> torch.Tensor:
        """
        Decode the LF band to a 48 kHz waveform using Vocos's native (correct)
        iSTFT, then resample 24 kHz -> 48 kHz.

        codes:        [K, T] or [K, B, T], EnCodec codebook indices
        bandwidth_id: 0=1.5 kbps, 1=3 kbps, 2=6 kbps, 3=12 kbps

        Returns: [B, T_48k] float, band-limited to < 12 kHz.
        """
        features = self.vocos.codes_to_features(codes)
        bid      = torch.tensor([bandwidth_id], device=self.device)
        lf_24k   = self.vocos.decode(features, bandwidth_id=bid)   # [B, T_24k]
        if lf_24k.dim() == 1:
            lf_24k = lf_24k.unsqueeze(0)
        return torchaudio.functional.resample(lf_24k, ENCODEC_SR, target_sr)

    @torch.no_grad()
    def get_lf_stft(self, codes: torch.Tensor, bandwidth_id: int = 3) -> torch.Tensor:
        """
        Returns Vocos's complex LF spectrum [B, 641, T].

        Do not reconstruct this with torch.istft. See the class docstring.
        Kept only for analysis/visualisation. The inference path uses
        decode_lf_audio() instead.
        """
        features     = self.vocos.codes_to_features(codes)
        bid_tensor   = torch.tensor([bandwidth_id], device=self.device)
        x = self.vocos.backbone(features, bandwidth_id=bid_tensor)

        head = self.vocos.head
        x    = head.out(x).transpose(1, 2)   # [B, 642, T]
        mag, p = x.chunk(2, dim=1)
        mag  = torch.exp(mag).clamp(max=1e2)
        return mag * (torch.cos(p) + 1j * torch.sin(p))   # [B, 641, T]


class EncodecLFExtractor(LFExtractor):
    """
    LF backend using EnCodec's own decoder (facebook/encodec_24khz) rather than
    Vocos. Lower quality than Vocos at the same bitrate, but it's the codec's
    native decoder, a single model family and natively streamable, which makes
    it the natural choice for a low-latency real-time path. Reuses an existing
    EncodecProcessor so EnCodec is not loaded a second time.

    bandwidth_id is accepted for interface parity but ignored: the bitrate is
    already fixed by how many codebooks are present in `codes`.
    """

    def __init__(self, processor, device: str = "cpu"):
        self.processor = processor
        self.device    = device

    @torch.no_grad()
    def decode_lf_audio(
        self,
        codes:        torch.Tensor,
        bandwidth_id: int = 3,
        target_sr:    int = SAMPLE_RATE,
    ) -> torch.Tensor:
        latents = self.processor.codes_to_latents(codes)        # [B, C, T]
        audio   = self.processor.decode_latents_audio(latents)  # [B, 1, T_24k]
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)
        return torchaudio.functional.resample(audio, ENCODEC_SR, target_sr)


def make_lf_extractor(
    backend:     str,
    *,
    vocos_model=None,
    processor=None,
    device:      str = "cpu",
) -> LFExtractor:
    """
    Factory for LF backends.
        backend="vocos"   -> VocosLFExtractor(vocos_model)
        backend="encodec" -> EncodecLFExtractor(processor)
    """
    backend = backend.lower()
    if backend == "vocos":
        if vocos_model is None:
            raise ValueError("vocos backend requires vocos_model=")
        return VocosLFExtractor(vocos_model, device=device)
    if backend == "encodec":
        if processor is None:
            raise ValueError("encodec backend requires processor=")
        return EncodecLFExtractor(processor, device=device)
    raise ValueError(f"unknown LF backend {backend!r} (use 'vocos' or 'encodec')")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LatentBWENet(hidden_dim=256, num_blocks=4)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Receptive field: {model.receptive_field_ms():.0f} ms")
    print(f"N_HF_BINS: {N_HF_BINS}  (bins {HF_BIN_START}-{N_BINS_FULL - 1})")

    dummy = torch.randn(2, LATENT_DIM, 75)
    waveform = model(dummy)
    stft     = model.forward_stft(dummy)
    print(f"Waveform output: {waveform.shape}")   # [2, 1, 48000]
    print(f"STFT output:     {stft.shape}")        # [2, 640, 75]
