# nac-bwe

Real-time bandwidth extension for small generative audio models.

Currently just for EnCodec based models.

EnCodec 24 kHz caps bandwidth at 12 kHz, so anything generating its tokens
sounds band-limited. These models synthesise the missing 12-24 kHz band from the
codec stream and add it back, giving 48 kHz. They are causal and stream block by
block, so they run alongside a live generator.

| Model | Conditioned on | Params |
| --- | --- | --- |
| `LatentBWENet` | EnCodec latents, `[B, 128, T]` | 482k |
| `AudioBWENet` | EnCodec-decoded 24 kHz audio, `[B, 1, T]` | 548k |

Each predicts STFT bins 641-1280 and runs its own iSTFT. The low band comes from
a frozen decoder (Vocos or EnCodec) as a separate 48 kHz waveform, and the two
are summed in the time domain. The bands meet exactly on bin 641, so they do
not overlap.

## Install

```bash
pip install -e .
```

EnCodec is fetched on first use. The BWE checkpoints are in this repo.

## Checkpoints

[`checkpoints/`](checkpoints) holds the two pretrained models, trained
100 epochs with the adversarial objective at `hidden_dim=128, num_blocks=3,
center=False`: `latent_small_gan.pt` (482k) and `audio_small_gan.pt` (548k).
Inference-only, and load under `weights_only=True`.

```python
from nac_bwe.checkpoints import load_release
model, meta = load_release("checkpoints/latent_small_gan.pt")
```

Export one from your own run: `python -m nac_bwe.checkpoints export runs/<run>/epoch_0099.pt out.pt`

## Use with your own generative model

Any generator emitting valid EnCodec 24 kHz tokens works. Use
`latent_small_gan.pt` for codes or latents, `audio_small_gan.pt` for a decoded
24 kHz waveform.

**Match the bitrate.** The checkpoints were trained at 12 kbps, `n_q=16`. A
different `n_q` degrades quality with no error raised. Check `codes.shape[0]`.

```python
import torch
from nac_bwe.checkpoints import load_release
from nac_bwe.codec import EncodecProcessor
from nac_bwe.models.latent_bwe_net import make_lf_extractor

processor = EncodecProcessor(sr=24000)
model, _ = load_release("checkpoints/latent_small_gan.pt")
lf = make_lf_extractor("encodec", processor=processor)

codes = your_generator.sample()          # [n_q, B, T], n_q == 16

with torch.no_grad():
    hf_48k = model(processor.codes_to_latents(codes)).squeeze(1)
    full_band_48k = lf.combine_time_domain(lf.decode_lf_audio(codes), hf_48k)
```

Streaming, one frame being 640 samples at 48 kHz (13.3 ms):

```python
from nac_bwe.models.latent_bwe_net import StreamingISTFT, HF_BIN_START

istft = StreamingISTFT(window=model.window)
state = None

for code_block in your_generator.stream():        # [n_q, B, T_block]
    with torch.no_grad():
        latents = processor.codes_to_latents(code_block)
        hf_stft, state = model.forward_stft_streaming(latents, state)

        # The model synthesises bins 641+ only. Zero the low bins so the
        # iSTFT sees a full spectrum.
        pad = torch.zeros(hf_stft.shape[0], HF_BIN_START, hf_stft.shape[-1],
                          dtype=hf_stft.dtype)
        hf_48k = istft(torch.cat([pad, hf_stft], dim=1))

        lf_48k = lf.decode_lf_audio(code_block)
        out_block = lf.combine_time_domain(lf_48k, hf_48k)

    play(out_block)
```

`istft.reset()` and `state = None` start a new stream. Run the low-band decoder
streaming too, or the blocks click at the seams.
[nac-bwe-experiments](https://github.com/smilo7/nac-bwe-experiments) has a
working example in `demo/realtime_bwe.py`, along with configs, evaluation and
dataset prep.

## Command line

```bash
python -m nac_bwe.data.precompute            --config <precompute.yaml>
python -m nac_bwe.training.train_latent_bwe  --config <train.yaml>
python -m nac_bwe.training.train_audio_bwe   --config <train.yaml>
python -m nac_bwe.inference.listen --input <dir> --checkpoint <best.pt>
```

## Tests

```bash
python tests/test_package.py          # every module ships and imports
python tests/test_hf_streaming.py     # streaming == offline (LatentBWENet)
python tests/test_audio_streaming.py  # streaming == offline (AudioBWENet)
python tests/test_end_to_end.py       # full path on a released checkpoint
```

The streaming tests need no checkpoint or network. `test_end_to_end.py` takes
`--audio` and `--save` to extend a real file. Run `test_package.py` against an
installed package rather than the source tree, or it cannot see a packaging
omission.

## License

MIT. See [LICENSE](LICENSE). The system is non-commercial regardless, since it
runs on Meta's EnCodec weights (CC-BY-NC 4.0). The checkpoints were trained on
Creative Commons audio from Freesound.
