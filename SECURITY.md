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
The web app binds to `127.0.0.1` by default and has no authentication. It is meant for local use. If you expose it on a network (`--host 0.0.0.0`), put it behind a reverse proxy with authentication. RVC voice models are PyTorch pickles, so only load voices from sources you trust. The built-in registry pins specific Hugging Face repos.
