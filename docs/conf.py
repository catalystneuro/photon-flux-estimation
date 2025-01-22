"""Sphinx configuration."""

import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'Photon Flux Estimation'
copyright = '2024, CatalystNeuro'
author = 'CatalystNeuro'
version = '0.1.0'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Theme
html_theme = 'sphinx_rtd_theme'

# Static files
html_static_path = ['_static']

# Master document
master_doc = 'index'
