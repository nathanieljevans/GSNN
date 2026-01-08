"""Top-level GSNN package.

This ``__init__`` keeps the public surface minimal but makes sure that
Sphinx can import the package during Read-the-Docs builds where many
optional heavy dependencies are *mocked* via ``autodoc_mock_imports``.
"""

from importlib import import_module as _import_module
import sys as _sys

# Re-export commonly used subpackages so that attributes such as
# ``gsnn.models.GSNN`` resolve without the user having to perform the
# intermediate import themselves.
__all__ = [
    "models",
    "simulate",
    "interpret",
    "optim",
    "proc",
]

for _mod in __all__:
    try:
        _import_module(f"{__name__}.{_mod}")
    except ModuleNotFoundError:
        # During docs builds heavy optional dependencies may be mocked or
        # completely absent.  Silently ignore missing submodules so that the
        # top-level import remains lightweight.
        pass

# ---------------------------------------------------------------------------
# Sphinx quirk
# ---------------------------------------------------------------------------
# When the ``currentmodule`` directive is used in the documentation like::
#
#     .. currentmodule:: gsnn
#
# and the autosummary table then lists a bare ``gsnn`` entry, Sphinx tries to
# resolve the object under the fully-qualified name ``gsnn.gsnn``.  This fails
# unless the alias actually exists.  The following line creates the alias by
# pointing ``sys.modules['gsnn.gsnn']`` back to *this* very module object.
_sys.modules.setdefault(f"{__name__}.gsnn", _sys.modules[__name__])

# Clean up helper symbols that are not part of the public interface.
del _import_module, _sys
