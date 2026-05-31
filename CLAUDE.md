# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

synthBOLD is a synthetic fMRI generator for BOLD simulations. It produces synthetic blood vessel networks and tissue label maps using PyTorch-based volumetric transforms, enabling training-data generation without real imaging data.

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

### Core abstraction layers

**`config.py`** — Pydantic models that define all synthesis parameters. `Config` is the root object (frozen, no extra fields) composed of `GeometryParams` (volume shapes and field-of-view) and `PhysioParams` (tissue and vessel generation parameters). `Config.from_yaml` / `Config.from_dict` use deep-merge semantics so partial YAML files override only the specified keys. `Range` is a reusable validated min/max pair.

**`base.py`** — Abstract base classes:
- `RandomGeneratorMixin` — provides a paired `torch.Generator` + `numpy.random.Generator`, both seedable for reproducibility. Mixed into all stateful generators.
- `BaseLabel(ABC, RandomGeneratorMixin)` — base for label map generators; owns `voxel_grid` and `normalized_grid` as `@cached_property` tensors of shape `[X, Y, Z, 3]`.
- `Transform(ABC, RandomGeneratorMixin)` — base for tensor-to-tensor transforms. Subclasses implement `sample(shape)` (compute the transform parameters) and `apply(x, transform)` (apply them). `forward` is decorated with `@require_dim(3, 4)` and `@ensure_ndim(target_ndim=4)` to normalize input dimensionality. `from_config(config)` is required for factory construction.
- `Pipeline` — chains a list of `Transform` instances sequentially.

**`labels.py`** — Concrete label map generators:
- `TissueLabel` — generates random tissue label maps using SynthMorph-style noise: low-res Gaussian noise → trilinear upsample → smooth warp via random displacement field → argmax over `J` channels assigns a label per voxel.
- `VesselLabel` — stub, not yet implemented.

Both expose `from_config(config: Config)` for config-driven construction and `__call__(n_sample, fname)` to batch-generate and optionally save to disk.

**`utils.py`** — Stateless helpers:
- `require_dim(*dims)` — decorator that validates tensor dimensionality on positional and keyword args.
- `ensure_ndim(target_ndim)` / `auto_batch` — decorators that promote/squeeze batch dimensions transparently.
- I/O: `load_nifti`, `save_nifti`, `load_zarr`, `save_zarr`, `save_mesh` (FreeSurfer format via marching cubes).

### Tensor conventions

- Internal computation device defaults to `"cuda"`, falls back to `"cpu"` in tests.
- Label maps have shape `[N, X, Y, Z]` (batch-first). When saving as NIfTI time series, axes are permuted to `[X, Y, Z, N]` via `save_nifti(..., permute=True)`.
- ZARR chunks are `(1, X, Y, Z)` — one chunk per sample.
- `GeometryParams.input_shape` must be integer-divisible by `output_shape` in every dimension (enforced by validator).

### Status

The project is under active development. `VesselLabel`, several `Transform` subclasses (Sphere, Toroid, noise types), temporal BOLD simulation, and visualization utilities are planned but not yet implemented (see `TODO.md`).
