import os
import sys

# Make the installed package importable for autodoc
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------

project = 'Oxford Loss Landscapes'
copyright = '2025, Ashley S. Dale'
author = 'Ashley S. Dale, Paige E. Bowling, Christian D. Harding, Alok Ghosh, Ryan Daniels'
release = '0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_title = 'Oxford Loss Landscapes'
