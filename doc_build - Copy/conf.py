# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']

project = 'vulky'
copyright = '2025, Ludwic Leonard'
author = 'Ludwic Leonard'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../src/'))  # Ensure Sphinx finds your module

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary'
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'special-members': '__init__',
    'show-inheritance': True,
}

# extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# autodoc_member_order = "bysource"
# autodoc_property_type = "method"

autosummary_mock_imports = [
    'vulky.datasets',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

import os
html_baseurl = "https://rendervous.github.io/vulky_project/"

html_theme_options = {
    "navigation_depth": 4,  # Adjust depth for sidebar tree navigation
    "collapse_navigation": True,  # Keeps sidebar expanded
    "sticky_navigation": True,
    "titles_only": False  # Show full structure
}