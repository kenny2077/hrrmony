# Security policy

## Reporting a vulnerability
Please **do not open a public issue**. Report it privately through [GitHub Security Advisories](https://github.com/kenny2077/hrrmony/security/advisories/new).

Include:
- the affected version or commit
- steps to reproduce
- what an attacker could achieve

You will get an acknowledgement within 7 days. We aim to release a fix within 90 days and will credit you unless you ask us not to.

## Scope
- In scope: the `hrrmony` package, the CLI, and the local web app (`hrrmony serve`). This includes upload handling, path handling and the job API.
- Out of scope: vulnerabilities in third-party dependencies (report those upstream) and in third-party voice-model files.

## Notes
- **Local web app:** it binds to `127.0.0.1` by default and has no authentication. It rejects foreign `Host` headers (DNS rebinding) and cross-origin POSTs. If you expose it on a network (`--host 0.0.0.0`), put it behind a reverse proxy with authentication, and set `HRRMONY_ALLOWED_HOSTS`.
- **Models:** every download is pinned to a Hugging Face commit. RVC and RMVPE checkpoints are loaded with `weights_only=True`, and HuBERT is loaded from safetensors.
- **Media files:** decoding untrusted media uses ffmpeg, so keep ffmpeg up to date.
