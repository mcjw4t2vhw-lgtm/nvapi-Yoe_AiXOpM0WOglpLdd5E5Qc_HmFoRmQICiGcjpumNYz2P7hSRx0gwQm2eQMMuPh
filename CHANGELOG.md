# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-08-28

### Added

- GPU inventory library with mock and `nvidia-smi` backends
- Health classification from temperature and VRAM pressure
- CLI (`nvapi list|get|summary|driver`) and FastAPI HTTP surface
- pytest suite covering models, service, CSV parser, HTTP, and CLI
- GitHub Actions workflow to run `pytest` on push and pull request

## [1.0.0] - 2026-08-08

### Added

- Initial project setup with core infrastructure and documentation
