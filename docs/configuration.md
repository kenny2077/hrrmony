# Configuration

## Environment variables
| Variable | Default | Purpose |
|---|---|---|
| `VILLAGER_SOUND_HOME` | `~/.cache/villager-sound` | Model cache (`models/`), cached separations (`work/`) and web-app jobs (`web/`) |
| `HF_HOME` | Hugging Face default | Where `transformers` caches the HuBERT encoder |
| `CUDA_VISIBLE_DEVICES` | all | Choose a GPU, or set it to empty to force CPU. The `--cpu` flag does the same. |

## `villager-sound cover`
| Option | Default | Meaning |
|---|---|---|
| `--hook` / `--full` | `--hook` | Best ~30 s (the chorus), or the entire song |
| `-d, --duration` | `30` | Hook length in seconds |
| `-s, --start` | auto | Start time of the hook in seconds; skips auto-detection |
| `--shift` | `classic` | `classic` (−8, the viral sound), `in-key` (−12), or any number of semitones |
| `--vocal-db` | `-3.9` | Villager level relative to the instrumental |
| `--loudness` | `-16` | Target LUFS of the master |
| `--format` | mp3 and wav | Repeat the flag to choose formats |
| `--stems` | off | Also write the villager vocal and the instrumental |
| `--cpu` | off | Ignore the GPU |
| `--json` | off | Print a machine-readable result |

## Cache layout
```
~/.cache/villager-sound/
  models/separator/   Mel-Band RoFormer checkpoint (≈ 870 MB)
  models/voices/      RVC voice models (≈ 65 MB each)
  models/pitch/       rmvpe.pt (≈ 180 MB)
  work/<hash>/        cached separations, keyed by file + window (safe to delete)
  web/<job>/          uploads and results from the web app (safe to delete)
```
