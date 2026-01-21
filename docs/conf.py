"""Sphinx configuration for otava-test-data."""

project = "otava-test-data"
copyright = "2025, Joe Drumgoole"
author = "Joe Drumgoole"
version = "0.1.3"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# MyST settings for markdown
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Napoleon settings for docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
