# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import pathlib
import sys
from datetime import date

import mccube

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(PROJECT_DIR))

# -- Project information -----------------------------------------------------

project = "MCCube"
copyright = f"{date.today().year}, The MCCube developers"
author = "The MCCube developers"
version = mccube.__version__

master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "myst_nb",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy-1.8.1/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "chex": ("https://chex.readthedocs.io/en/latest/", None),
}
intersphinx_disabled_reftypes = ["*"]

# AutoDoc configuration
add_module_names = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_theme_options = {
    "announcement": (
        "⚠️ MCCube is currently a work in-progress, expect changes, sharp edges, "
        "and treat all results with a healthy degree of skepticism! ⚠️"
    ),
    # Visual options
    "sidebar_hide_name": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_title = "MCCube: Markov Chain Cubature via JAX"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_show_sphinx = False
html_css_files = ["custom.css"]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}

nb_execution_mode = "auto"
nb_execution_timeout = 300
suppress_warnings = ["mystnb.unknown_mime_type"]

nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}
myst_enable_extensions = ["colon_fence", "dollarmath"]


bibtex_bibfiles = ["_static/references.bib"]
bibtex_default_style = "alpha"  # alpha, plain, unsrt, unsrtalpha
