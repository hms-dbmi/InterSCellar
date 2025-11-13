# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    import interscellar
except ImportError:
    pass

project = 'InterSCellar'
copyright = '2025, Eunice Lee'
author = 'Eunice Lee'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc", 
    "sphinx.ext.autosummary", 
    "sphinx.ext.napoleon", 
    "sphinx.ext.viewcode", 
    "myst_parser"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_baseurl = os.environ.get('SPHINX_HTML_BASEURL', 'https://euniceyl.github.io/InterSCellar/')

# Theme options
html_theme_options = {
    "repository_url": "https://github.com/euniceyl/InterSCellar",
    "use_repository_button": True,
    "use_edit_page_button": True,
}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_mock_imports = ['napari', 'anndata', 'duckdb']
