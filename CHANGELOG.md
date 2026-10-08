# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Security
- Model downloads are pinned to exact Hugging Face commits: the villager RVC model, RMVPE, and the HuBERT encoder (safetensors, now fetched into the hrrmony cache).
- The RVC and RMVPE checkpoints are loaded with `torch.load(weights_only=True)`, so a tampered pickle can't run code, even on torch older than 2.6.
- The web app rejects requests whose `Host` is not loopback (DNS rebinding) and cross-origin POSTs (CSRF). It warns when it is bound to a non-loopback address. Use `HRRMONY_ALLOWED_HOSTS` to allow names behind a proxy.
- The web app accepts at most 8 queued covers (it answers 429 beyond that) and deletes uploads, results and cached separations after 24 hours.

### Fixed
- The master stage could crash with a shape mismatch when the RVC output and the instrumental differed by a few samples.
- `--full` used ffprobe's container duration, which can cut off the end of VBR MP3s. It now decodes the whole file.
- The cache key now includes the file's modification time and the separator model, so an edited song with the same size is no longer served from a stale cache.
- Cache writes are atomic (decode and separation use temporary files), so an interrupted run no longer leaves files that later runs trust.
- The villager level is capped at 3 dB above the original singer. Near-instrumental tracks no longer get separation noise boosted to vocal level.
- Web UI: a 404 or 500 while polling now shows an error instead of polling forever. A second song can't be submitted while one is running. Players no longer stack event listeners after each cover. The background fade-in recovers after a resize.

### Changed
- Product page: the demo is villager-only (the original-singer take and the voice switch are gone).
- The vocal separator model is loaded once and reused across songs; it used to be reloaded for every web job.
- Models that are already downloaded are used without contacting the Hub, so covers work offline.
- The CLI and the web API share one shift parser, limited to ±24 semitones. `--duration` must be positive. The CLI defaults come from the library constants.

## [0.2.0] - 2026-10-07
### Changed
- **Renamed to Hrrmony**, from the villager's “hrmm” plus harmony. The package and CLI are now `hrrmony`, the Python module is `hrrmony`, the cache variable is `HRRMONY_HOME` (default `~/.cache/hrrmony`), and the repository is `kenny2077/hrrmony`.
- New pixel villager icon for the web app and the product page.

### Added
- Product page (`site/`) in the Aurora Forge Lab style: a pixel Minecraft night-village hero, a cursor block-heat trail and card flashlight, an A/B demo player using a public-domain song, feature cards and install steps. Deployed to GitHub Pages by `.github/workflows/pages.yml`.

## [0.1.0] - 2026-10-06
### Added
- `villager-sound cover` turns a song into a Minecraft villager cover, either from its best 30 seconds (`--hook`, which auto-detects the chorus) or from the whole song (`--full`).
- Pipeline: Mel-Band RoFormer vocal separation, then RVC villager voice conversion (`classic` −8 st or `in-key` −12 st), then a reference-matched vocal EQ and mix, then mastering to −16 LUFS.
- A local web app (`villager-sound serve`) with upload, live progress, side-by-side playback of the cover and the original excerpt, and MP3/WAV download.
- `villager-sound doctor` checks ffmpeg, torch/CUDA and the model cache. `villager-sound voices` lists and pre-downloads voices.
- Docker image (CUDA) and compose file.
- Research log and the measurements behind the recipe (`research/`, `docs/how-it-works.md`).

[Unreleased]: https://github.com/kenny2077/hrrmony/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/kenny2077/hrrmony/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kenny2077/hrrmony/releases/tag/v0.1.0
