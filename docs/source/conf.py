import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'vulky'
copyright = '2025, Ludwic Leonard'
author = 'Ludwic Leonard'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # Generate documentation from docstrings
    'sphinx.ext.napoleon',          # Support for Google/NumPy-style docstrings
    'sphinx_autodoc_typehints',     # Type hints in documentation
    'sphinx.ext.viewcode',          # Add links to source code
    'sphinx.ext.autosummary'
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '__init__',
    'show-inheritance': True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
