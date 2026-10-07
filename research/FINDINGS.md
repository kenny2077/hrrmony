# How the reference villager cover was made

Reference: `ref/target/villager_rickroll_ai.mp3`. The raw numbers are in `analysis/*.json`. The analysis scripts are in `analysis/*.py`, and they were run on the GPU laptop.

## Verdict
The reference is **RVC voice conversion of Rick's full vocal stem with a community Minecraft-villager RVC v2 model, transposed −8 semitones**. It is mixed dry and in mono, about 4 dB under the original instrumental.

## Evidence
- **Pitch.** The vocal sits a constant −801.8 cents from Rick. Within each syllable it follows Rick's pitch movement (median correlation 0.79). It does not follow the pitch shapes of the library clips.
- **Not pasted samples.**
  - The clip-identification test recovers 76% of clips in a known one-clip-per-syllable control.
  - On the reference it identifies only 10% confidently, the same rate as the RVC outputs and as Rick himself.
  - There is no reuse of a small clip set.
- **The "hit" sounds are real, but they come from the conversion.**
  - The reference has about 37–43 short noise bursts per minute. Their spectra are closest to the villager `hit1-4` clips.
  - 42% of the bursts sit on Rick's consonants, against 9% at shuffled times.
  - The villager RVC model turns consonants into these bursts. Plain RVC model A produces the same burst types.
  - Our sample-based hum (v4) deletes the consonants, so it has 0 bursts per minute.
- **Mix.**
  - The instrumental is the original song at about −2 dB (correlation 0.88). There is no bleed from Rick's vocal.
  - The villager vocal is mono (side channel −32.6 dB), dry (decay 98 dB/s), uncompressed (crest factor 22 dB), and 3.9 dB under the instrumental.
  - EQ: −10 to −12 dB at 50–100 Hz, and −4 to −11 dB above 4 kHz. The mix is −15.8 LUFS overall.
- **Best model.** Of the four public villager RVC models tested, `r3gm/villager` (model A, the same file as kullbunta/xufukx) matches best. Its loudness-envelope correlation is 0.887 on the full vocal stem, against 0.832 on the lead only.

## Reproduction
Script: `rvc_cover.py`. Measured on the 29.5–60.5 s excerpt:

| | Theirs | Reproduction (−8) | Old v4 hum |
|---|---|---|---|
| Syllables | 50 | 47 | 47 |
| Single loudness bump per syllable | 72% | 68% | 87% |
| Aperiodicity | 0.19 | 0.19 | 0.09 |
| Subharmonic (growl) | −43.8 dB | −40.3 dB | −39.4 dB |
| Hit-like bursts per minute (nearest = villager hit) | 36.8 (58%) | 50.3 (73%) | 0 |

**Still open (hypothesis).** The reference has less pitch jitter (5.9 c against our 7.9 c) and more timbre change from word to word. A different checkpoint of the villager model, or extra pitch correction, could explain both.
