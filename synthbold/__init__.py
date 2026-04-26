"""Synthetic fMRI generator for BOLD simulations.

This package contains classes and functions used to generate synthetic blood vessel
networks and tissue data. By using synthetic data, a rich learning environment can be
created by high variance sampling and unrealistic domain randomization, which eliminates
the need for real imaging data.

Copyright (c) 2026 Daniel Haenelt
License: GPL-3.0-or-later
"""

import os
import tomllib
from importlib.metadata import PackageNotFoundError, version

try:
    # Read version number from package metadata.
    __version__ = version("synthBOLD")
except PackageNotFoundError:
    # Read version number from pyproject.toml (fallback solution).
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    __version__ = data["project"]["version"]

__author__ = "Daniel Haenelt"
__license__ = "GPL-3.0-or-later"
