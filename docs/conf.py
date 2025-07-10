import os
import sys
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'Graph Structured Neural Networks'
author = 'Nathaniel Evans'
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
autosummary_imported_members = True  # create stub pages for attributes re-exported at import time

# Mock heavy optional dependencies so that Read the Docs can build without
# having to compile/install the full deep-learning stack.
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "torch_sparse",
    "torch_scatter",
    "torch_cluster",
    "pyg_lib",
    "pyro",
    "pyro_ppl",
    "numpy",
    # Additional optional heavy dependencies mocked for Read the Docs
    "pandas",
    "scipy",
    "networkx",
    "matplotlib",
    "sklearn",
    "sklearn.metrics",
    "sklearn.linear_model",
    "sklearn.preprocessing",
]

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

# -- nbsphinx options ---------------------------------------------------------

nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_prolog = """
.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
    </style>
"""

nbsphinx_epilog = """
.. raw:: html

    <script>
        // Remove automatic TOC generation from notebooks
        document.addEventListener('DOMContentLoaded', function() {
            var toc = document.querySelector('.toctree-wrapper');
            if (toc && toc.querySelector('.toctree-l1')) {
                var items = toc.querySelectorAll('.toctree-l1');
                items.forEach(function(item) {
                    if (item.querySelector('.toctree-l2')) {
                        item.querySelector('.toctree-l2').remove();
                    }
                });
            }
        });
    </script>
"""

# Disable automatic TOC generation for notebooks
nbsphinx_thumbnails = {}
nbsphinx_thumbnails_title = "Gallery"

# Prevent automatic section numbering and TOC generation
nbsphinx_number_source = False
nbsphinx_use_mathjax = True

# Disable automatic TOC generation
nbsphinx_include_source = True
nbsphinx_include_mathjax = False

# Include detailed member documentation automatically
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
} 