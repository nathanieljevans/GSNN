__all__ = [
    "simulate",
    "nx2pyg",
    "utils",
]

from importlib import import_module as _import_module

for _mod in __all__:
    try:
        _import_module(f"{__name__}.{_mod}")
    except ModuleNotFoundError:
        # Optional deps may be mocked during docs build.
        pass

# Re-export the main function directly for convenience
try:
    from .simulate import simulate  # noqa: F401
except Exception:
    # Safe to ignore if heavy deps missing.
    pass

del _import_module
