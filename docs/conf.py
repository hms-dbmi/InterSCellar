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

master_doc = "index"

myst_enable_extensions = ["colon_fence", "substitution"]
myst_substitutions = {
    "image_path": "images/package_workflow.jpeg"
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_baseurl = os.environ.get('SPHINX_HTML_BASEURL', 'https://euniceyl.github.io/InterSCellar/')

# Customize the title (change this to your desired text)
html_title = "InterSCellar"

# Option 1: Add a logo (uncomment and set path if you have a logo)
# html_logo = "images/logo.png"  # Path relative to _static or docs directory
# html_logo = "_static/logo.png"  # If logo is in _static folder

# Option 2: Use a different title format
# html_title = "InterSCellar Documentation"
# html_title = "InterSCellar - Cell Interaction Analysis"

# Theme options
html_theme_options = {
    "repository_url": "https://github.com/euniceyl/InterSCellar",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    # Hide the version in the title if you want
    # "navbar_title": "InterSCellar",  # Custom navbar title
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
