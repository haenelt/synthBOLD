"""Configuration file for the Sphinx documentation builder."""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
from datetime import datetime

from synthbold import __version__

# -- Project information -----------------------------------------------------

project = "synthBOLD"
copyright = f"{datetime.now().year}, Daniel Haenelt"
author = "Daniel Haenelt"
version = ".".join(__version__.split(".")[:2])
release = __version__

# -- General configuration ---------------------------------------------------

extensions: list[str] = [
    "sphinx.ext.mathjax",  # clean math rendering
    "sphinx.ext.autodoc",  # if you want to document Python API
    "sphinx.ext.napoleon",  # support Google / NumPy style docstrings
    "sphinxcontrib.autodoc_pydantic",  # to avoid clahses with pydantic
]

templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"

# Furo theme options for clean, literature-style look
html_theme_options = {
    "sidebar_hide_name": True,  # hide project name in sidebar
    "light_css_variables": {
        "color-content-background": "#ffffff",
        "color-sidebar-background": "#f8f8f8",
        "color-admonition-background": "#f4f4f4",
    },
    "dark_css_variables": {
        "color-content-background": "#1a1a1a",
        "color-sidebar-background": "#111111",
        "color-admonition-background": "#222222",
    },
    "navigation_with_keys": False,  # remove UI hints to keep clean
}

html_static_path = ["_static"]
html_css_files = ["academic.css"]  # custom styles
