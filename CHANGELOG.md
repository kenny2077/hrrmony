# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Added
- Product page (`site/`) in the Aurora Forge Lab style, with a lamp-field hero, an A/B demo player using a public-domain song, feature cards and install steps. Deployed to GitHub Pages by `.github/workflows/pages.yml`.

## [0.1.0] - 2026-10-06
### Added
- `villager-sound cover` turns a song into a Minecraft villager cover, either from its best 30 seconds (`--hook`, which auto-detects the chorus) or from the whole song (`--full`).
- Pipeline: Mel-Band RoFormer vocal separation, then RVC villager voice conversion (`classic` −8 st or `in-key` −12 st), then a reference-matched vocal EQ and mix, then mastering to −16 LUFS.
- A local web app (`villager-sound serve`) with upload, live progress, side-by-side playback of the cover and the original excerpt, and MP3/WAV download.
- `villager-sound doctor` checks ffmpeg, torch/CUDA and the model cache. `villager-sound voices` lists and pre-downloads voices.
- Docker image (CUDA) and compose file.
- Research log and the measurements behind the recipe (`research/`, `docs/how-it-works.md`).

[Unreleased]: https://github.com/kenny2077/villager-sound/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kenny2077/villager-sound/releases/tag/v0.1.0
