<p align="center">
  <img src="assets/screenshot.jpg" alt="Villager Sound web app: upload a song, choose the best 30 seconds or the whole song, and get a Minecraft villager cover" width="100%">
</p>

<h1 align="center">Villager Sound</h1>

<p align="center"><b>Every song, sung by a Minecraft villager.</b><br>
Drop in a track and get back a villager cover, made locally on your own machine: one “hrmm” per word, hit-sound consonants and all.</p>

<p align="center">
  <a href="https://github.com/kenny2077/villager-sound/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/kenny2077/villager-sound/ci.yml?branch=main&style=for-the-badge&label=CI"></a>
  <a href="LICENSE"><img alt="MIT licence" src="https://img.shields.io/badge/licence-MIT-3fe0c8?style=for-the-badge"></a>
  <img alt="Python 3.10 to 3.12" src="https://img.shields.io/badge/python-3.10%E2%80%933.12-3aa0e8?style=for-the-badge">
  <img alt="GPU optional" src="https://img.shields.io/badge/GPU-optional-ffb14a?style=for-the-badge">
</p>

---

## Features

<table>
<tr><td><b>Two ways to cover</b></td><td>The best 30 seconds, with the chorus found automatically, or the whole song.</td></tr>
<tr><td><b>Sounds like the viral covers</b></td><td>The recipe is reverse-engineered and measured against a popular AI villager cover, not guessed. See <a href="docs/how-it-works.md">how it works</a>.</td></tr>
<tr><td><b>Keeps the song recognisable</b></td><td>The original instrumental stays as it is. The villager follows the singer's own pitch, timing and phrasing.</td></tr>
<tr><td><b>Web app and CLI</b></td><td>A local web app with live progress and side-by-side playback, or one command in a terminal or script.</td></tr>
<tr><td><b>Fast on a laptop GPU</b></td><td>About 25 s for a 30-second cover on an RTX 4050. CPU works too, just slower.</td></tr>
<tr><td><b>Private</b></td><td>Your music never leaves your machine. Models are downloaded once and cached.</td></tr>
</table>

## Quick install

You need Python 3.10–3.12 and [ffmpeg](docs/faq.md).

```bash
pip install "villager-sound[gpu,web] @ git+https://github.com/kenny2077/villager-sound"   # NVIDIA GPU
pip install "villager-sound[cpu,web] @ git+https://github.com/kenny2077/villager-sound"   # CPU only / macOS
villager-sound doctor
```

Or with Docker (NVIDIA Container Toolkit):

```bash
git clone https://github.com/kenny2077/villager-sound && cd villager-sound
docker compose up --build      # then open http://127.0.0.1:7860
```

## Getting started

```bash
villager-sound serve                                   # web app at http://127.0.0.1:7860
villager-sound cover song.mp3                          # best 30 s (chorus), "classic" villager sound
villager-sound cover song.mp3 --full                   # the entire song
villager-sound cover song.mp3 --shift in-key           # an octave down, in tune with the music
villager-sound cover song.mp3 --start 43 -d 20         # pick the excerpt yourself
villager-sound cover song.mp3 --stems -o covers/       # also save villager vocal + instrumental
villager-sound voices --download villager              # pre-fetch models (~1.2 GB, once)
villager-sound doctor                                  # check ffmpeg, CUDA and the cache
```

Each cover writes `<song>_villager_<start>s.mp3` and `.wav`, plus a loudness-matched `…_original.mp3` of the same excerpt for A/B listening.

### From Python

```python
from villager_sound import CoverOptions, make_cover

result = make_cover("song.mp3", "covers/", CoverOptions(mode="hook", shift=-8))
print(result.outputs["mp3"], result.start, result.timings["total"])
```

## How it works

```
song ─▶ find the hook ─▶ separate vocal ─▶ RVC villager voice ─▶ EQ + mix ─▶ master (−16 LUFS)
        (or whole song)   Mel-Band RoFormer   −8 st "classic" /     dry, mono,
                                              −12 st "in key"       −3.9 dB under the music
```

The popular villager covers turned out not to be villager clips pasted onto notes. They are the singer's own vocal re-voiced by a model trained on villager sounds. That is why the phrasing stays human and the consonants come out as villager *hurt* noises.

[docs/how-it-works.md](docs/how-it-works.md) has every measurement behind the settings. [research/](research/) has the experiment log and the scripts.

| Song (30 s hook) | Hook found | Time on RTX 4050 |
|---|---|---|
| 凉凉 | 91.5 s (chorus at 90.3 s) | 24 s |
| Shape of You | 204.9 s (final chorus) | 28 s |
| Never Gonna Give You Up (`--start 29.5`) | — | ~30 s |

## Documentation

| Page | What's covered |
|---|---|
| [How it works](docs/how-it-works.md) | Pipeline, the reverse-engineered recipe, the hook finder, performance |
| [Configuration](docs/configuration.md) | CLI options, environment variables, cache layout |
| [FAQ and troubleshooting](docs/faq.md) | ffmpeg, CUDA, model downloads, off-key results, rights |
| [Research log](research/README.md) | Five iterations from pasted clips to voice conversion |

## Responsible use

- Covers of copyrighted songs are derivative works. Make sure you have the right to use the song before you publish anything.
- The default voice is a community RVC model ([r3gm/villager](https://huggingface.co/r3gm/villager)), downloaded at runtime. It is not part of this repository, so check its terms.
- Villager Sound is a fan project. It is not affiliated with Mojang Studios or Microsoft. Minecraft is a trademark of Mojang Studios.

## Contributing

Issues and PRs are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md): setup, tests, and the Conventional Commits we use. Report security problems privately, as described in [SECURITY.md](SECURITY.md).

---

MIT, see [LICENSE](LICENSE). Includes vendored [infer_rvc_python](https://github.com/R3gm/infer_rvc_python) (MIT).
