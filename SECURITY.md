# Security

## Reporting

Please report security issues privately to the maintainer (see git history / `pyproject.toml` author email). Do not open a public issue for undisclosed vulnerabilities.

## Scope

This repository is a research-style training library and CLI. It is not a hosted service. Threat model focuses on dependency supply chain, unsafe deserialization of pickles, and untrusted training data.

## Pickle artifacts

Saved models use Python `pickle`. Only load artifacts from trusted sources. Prefer signed or internally verified artifacts in production.

## Supported versions

Security fixes are applied on the latest `main` release line unless stated otherwise.
