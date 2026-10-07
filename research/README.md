# Research log

These are the scripts from the experiments behind `villager-sound`. They aren't part of the installed package, and they assume the original working layout: `input/`, `work/`, `ref/`, `out/`, and a `.venv-conv` environment. They show how the method was found and measured, so new ideas can be tested the same way.

## Iterations

| Version | Idea | Script | Outcome |
|---|---|---|---|
| v0 | Lyrics sung by TTS, re-timbred with WORLD DSP or zero-shot Seed-VC | `villager_sing.py`, `evaluate.py` | Seed-VC kept the words. Too slow on CPU. |
| v1–v2 | Song → separate → pitch tracking (RMVPE) → one real villager "hrmm" clip pasted per syllable | `convert.py`, `faithfulness.py` | In tune and recognisable, but noisy: several clips per word and a growl. |
| v3 | Copy the "essence" of a viral AI cover: clip mix, staccato, EQ | `analyze_reference.py` | Closer on spectra, but still built from pasted clips. |
| v4 | Cleaner pasted-clip hum (envelopes, jitter, vowel colour) | `map_sounds.py`, `compare_style.py` | Cleaner, but still not the right sound. |
| **v5** | **RVC voice conversion of the full vocal stem with a villager model, −8 st, dry and mono, −4 dB under the instrumental** | `rvc_cover.py`, `analysis/` | **Matches the reference.** This became the package. |

## What the reference cover turned out to be
The full write-up is in [`FINDINGS.md`](FINDINGS.md). In short:
- **No pasted samples.** It is RVC voice conversion of the singer's vocal. A clip-identification test that recovers 76% of clips on a control track finds only chance-level matches in the reference.
- **The villager "hit" sounds are real.** They are the singer's consonants, which the villager model turns into noise bursts. There are about 37 bursts a minute, and their spectra are closest to `entity.villager.hurt`.
- **Transposed −8 semitones**, measured as −801.8 cents. That leaves the vocal a major third off the backing track.
- **Mix:** dry, mono, uncompressed, 3.9 dB under the original instrumental, with a high-pass at about 120 Hz and a darkened top end.

## Tools worth reusing
- `map_sounds.py catalog` downloads the villager-family voice clips from Mojang's asset server and labels them. It covers 88 clips with their in-game event names. The clips are not committed here.
- `compare_style.py` gives per-syllable cleanliness metrics: single bump, jitter, aperiodicity, subharmonic level and timbre jump.
- `faithfulness.py` checks pitch, rhythm, chroma and a "name that tune" localisation.
- `analysis/bursts*.py` count hit-like bursts and find the nearest library sound for each.
