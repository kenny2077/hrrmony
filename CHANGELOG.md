# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
