# How it works

```
song.mp3
  │  1  find the hook       loudness × chroma self-similarity, best 30 s, snapped to a beat
  │                         (skipped with --full)
  │  2  separate            Mel-Band RoFormer → full vocal (lead + backing) | instrumental
  │  3  villagerize         RVC v2 villager voice: rmvpe pitch, index 0.75, protect 0.33,
  │                         shift −8 st ("classic") or −12 st ("in-key")
  │  4  vocal EQ            4th-order high-pass 120 Hz, −8 dB shelf above 5 kHz
  │  5  mix                 mono and dry, 3.9 dB under the instrumental's level
  ▼  6  master              EBU R128 loudness to −16 LUFS, −1 dBTP → MP3 + WAV (+ original excerpt)
villager cover
```

## Why this recipe
Every setting comes from measuring a popular "Minecraft Villager – Never Gonna Give You Up (AI generated)" cover against the original song. The scripts and numbers are in [`research/`](../research/).

| What we measured in the reference | Value | Where it shows up |
|---|---|---|
| Pitch offset from the original singer | −801.8 cents, following the singer's contour (r = 0.79) | `shift = -8` |
| Syllables per 31 s chorus | 50 (raw RVC from the full vocal stem: 46–48) | full vocal stem, not lead only |
| Short noise bursts | about 37/min, nearest library sound `entity.villager.hurt` | comes from RVC turning consonants into villager noise |
| Vocal level | 3.9 dB under the instrumental | `mix.VOCAL_DB` |
| Vocal EQ vs raw RVC | −10 to −12 dB at 50–100 Hz, −4 to −11 dB above 4 kHz | `mix.vocal_eq` |
| Reverb / stereo / compression | none: decay 98 dB/s, side channel −32.6 dB, crest factor 22 dB | dry mono vocal |
| Overall loudness | −15.8 LUFS | `--loudness -16` |

Our reproduction matches the reference on syllable count (50 vs 50), single-bump syllables (68% vs 72%), aperiodicity (0.19 vs 0.19) and hit-like bursts (present vs present). The plain sample-pasting approach we tried first had none of the bursts.

## The hook finder
Choruses are usually the loudest part of a pop song and the part whose harmony repeats most. `segment.hook_scores` multiplies mean RMS by mean chroma recurrence for every 1-second start position and takes the maximum. On the songs we tested, it picked the chorus of 凉凉 at 91.5 s (the chorus starts at 90.3 s) and the final chorus of *Shape of You*. Use `--start` to choose a section yourself.

## Performance
On an RTX 4050 laptop GPU (6 GB):
- 30 s hook: about 25–30 s end to end
- whole 3–4 min song: about 1 min
- first run only: about 1.2 GB of model downloads

On CPU it works but runs several times slower.
