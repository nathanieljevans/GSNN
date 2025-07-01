import os
import sys
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'GSNN'
author = 'GSNN Authors'
copyright = f"{datetime.now().year}, {author}"

# Ideally, this is dynamically imported from the package
try:
    import importlib.metadata as importlib_metadata  # Python 3.8+
    release = importlib_metadata.version('gsnn')
except Exception:
    release = '0.0.0-dev'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'nbsphinx',
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'pyg_sphinx_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'  # Adjust when logo is available
html_theme_options = {
    # Mimic PyTorch Geometric navigation behaviour
    'collapse_navigation': False,
    'navigation_with_keys': True,
    'logo': {
        'text': project,
    },
}

# -- Intersphinx configuration ----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', {}),
    'torch': ('https://pytorch.org/docs/stable/', {}),
    'pyg': ('https://pytorch-geometric.readthedocs.io/en/latest/', {}),
}

# -- Options for myst-parser -------------------------------------------------

myst_enable_extensions = [
    'deflist',
    'colon_fence',
] 