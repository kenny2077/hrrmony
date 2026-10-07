# Contributing

Thanks for helping make the villagers sing.

## Development setup
```bash
git clone https://github.com/kenny2077/villager-sound && cd villager-sound
python -m venv .venv && source .venv/bin/activate      # Python 3.10–3.12
pip install -e ".[gpu,web,dev]"                        # or [cpu,web,dev]
villager-sound doctor
```
You also need `ffmpeg` on your `PATH`.

## Checks
```bash
ruff check .                 # lint (CI runs the same)
pytest                       # fast tests: no models, no GPU, runs in seconds
pytest -m slow               # end-to-end with real models (downloads ~1.2 GB; GPU recommended)
```
Unit tests must not download models or need a GPU. Mock the heavy parts the way `tests/test_server.py` does.

## Project layout
```
src/villager_sound/
  cli.py          command line (cover / serve / doctor / voices)
  pipeline.py     make_cover(): orchestrates the stages below
  segment.py      hook (chorus) finder
  separate.py     Mel-Band RoFormer via audio-separator
  voice.py        RVC voice conversion + model downloads
  mix.py          vocal EQ, level, fades
  audio.py        ffmpeg / soundfile helpers, loudness mastering
  server/         FastAPI app + static web UI (no build step)
  _vendor/        vendored infer_rvc_python (MIT), see its README
research/         experiment scripts behind the recipe (not packaged)
```

## Changing the sound
The recipe is measured, not guessed. If you change separation, conversion or mix settings, compare against a reference with the tools in `research/` (`compare_style.py`, `faithfulness.py`, `analysis/bursts*.py`). Put the before and after numbers in the PR.

## Commits and PRs
- Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`, written in lowercase and imperative, with no trailing period.

  | type | use for |
  |---|---|
  | `feat` | a user-visible feature |
  | `fix` | a bug fix |
  | `refactor` | code change with no behaviour change |
  | `perf` | faster or lighter |
  | `docs` | documentation only |
  | `test` | tests only |
  | `ci` | workflows |
  | `chore` | tooling, deps, housekeeping |

  Examples: `feat(cli): add --stems to export the villager vocal`, `fix(segment): clamp recurrence width for short songs`.
- Name branches like `feat/…`, `fix/…`, `docs/…`.
- Keep PRs focused. Fill in the PR template and add a `CHANGELOG.md` entry under *Unreleased*.
- Never commit audio or model weights. `.gitignore` blocks the common formats.
