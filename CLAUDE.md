# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

synthBOLD is a synthetic fMRI generator for BOLD simulations. It produces synthetic tissue images with blood vessel networks or other objects (spheres, cubes, tetrahedra, toroids, etc.) using PyTorch-based volumetric transforms, enabling training-data generation without real imaging data.

## Commands

This project uses `uv` for dependency management. All commands should be run via `uv run`.

```bash
# Run all tests
make test
# or: uv run pytest

# Run a single test file
uv run pytest tests/test_base.py

# Run a single test by name
uv run pytest tests/test_base.py::test_transform_call

# Lint, format, and type-check
make format
# or individually:
uv run ruff check . --fix
uv run ruff format .
uv run mypy synthbold tests

# Build Sphinx docs
make build-docs

# Clean all build/cache artifacts
make clean
```

Pre-commit hooks run ruff (lint + format) and mypy on every commit. Install with `uv run pre-commit install`.

## Architecture

The project is under active development; the structure below reflects the current state and will change. Read the source files directly for authoritative details rather than relying on this summary.

### Key modules

- **`config.py`** — Pydantic models for synthesis parameters. `Config` is the root object, composed of sub-configs for geometry (`GeometryParams`), physiology (`PhysioParams`), and augmentation transforms (`TransformParams`). Supports partial YAML overrides via deep-merge.
- **`base.py`** — Abstract base classes: `BaseGeometry`/`ObjectGeometry` for geometry generators, `Model` for signal-generation models, `Transform`/`Pipeline` for tensor-to-tensor transforms, and `RandomGeneratorMixin` for reproducible seeding.
- **`geometries.py`** — Concrete geometry generators: `Shapes`, `Cylinders`, `Spheres`, `Tetrahedra`, `Cubes`, `Toroids`.
- **`transforms/`** — Augmentation transforms (subclasses of `Transform`), split into `intensity.py`, `noise.py`, `spatial.py`, and `deform.py`, with shared helpers in `functional.py`.
- **`models/`** — Physics-based signal models (subclasses of `Model`) that turn geometry/label volumes into synthetic intensity data, e.g. `models/background.py`.
- **`decorator.py`** — `require_dim` and `accept_unbatched` decorators for tensor dimension validation and transparent batch-dimension handling.
- **`io.py`** — I/O helpers for NIfTI, ZARR, and mesh formats.

### Tensor conventions (current)

- Batch-first layout: `[N, X, Y, Z]`.
- Device defaults to `"cuda"`, falls back to `"cpu"` in tests.
- ZARR chunks are `(1, X, Y, Z)`.

See `TODO.md` for planned work.
