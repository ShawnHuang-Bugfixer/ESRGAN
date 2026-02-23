# Repository Guidelines

## 要求
使用中文

## 环境
使用 conda 配置 python 环境,已有配置好的环境 realesrgan-py310

## COS SDK 快速入门在线文档
https://cloud.tencent.com/document/product/436/12269

## Project Structure & Module Organization
Core package code lives in `realesrgan/`:
- `realesrgan/archs/` for network architectures.
- `realesrgan/models/` for training/inference model wrappers.
- `realesrgan/data/` for dataset loaders.
- `realesrgan/train.py` as the training entry point.

Utilities and conversion scripts are in `scripts/`. Training configs are in `options/*.yml`. Tests are in `tests/` with sample fixtures under `tests/data/`. User-facing inference entry scripts are `inference_realesrgan.py` and `inference_realesrgan_video.py`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt && python setup.py develop`: install dependencies and editable package.
- `pytest tests/` or `python setup.py test`: run the full test suite (configured in `setup.cfg`).
- `pre-commit install`: install local hooks before committing.
- `pre-commit run --all-files`: run lint/format/spell checks manually.
- `python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs -o results`: quick image inference smoke test.
- `python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --debug`: single-process training debug run.

## Coding Style & Naming Conventions
Python style is enforced by `flake8`, `isort`, and `yapf` via pre-commit.
- Max line length: 120.
- Keep imports sorted (`isort`) and formatting consistent (`yapf`).
- Follow existing naming patterns: snake_case for functions/files/variables, PascalCase for classes.
- Prefer small, focused modules under existing package folders instead of new top-level scripts.

## Testing Guidelines
Use `pytest` and place tests in `tests/` as `test_*.py`. Mirror module behavior with targeted unit tests (datasets, models, utils, and arch components). No explicit coverage gate is configured; contributors should add regression tests for bug fixes and tests for new logic paths.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects (for example: `fix colorspace bug`, `update readme`, `add github release workflow`). Keep subject lines concise and lower-case where practical; include scope when useful.

For PRs:
- Work from a feature branch (not `master`).
- Describe what changed and why.
- Link related issues/discussions for larger changes.
- Run `pre-commit` and `pytest` before opening.
