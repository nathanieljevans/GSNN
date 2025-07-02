__all__ = [
    "AE",
    "GSNN",
    "GroupBatchNorm",
    "GroupLayerNorm",
    "SoftmaxGroupNorm",
    "SparseLinear",
    "ICNN",
    "Logistic",
    "NN",
    "VAE",
    "utils",
]

from importlib import import_module as _import_module

# Import submodules so that they are available as attributes of `gsnn.models`.
for _mod in __all__:
    try:
        _import_module(f"{__name__}.{_mod}")
    except ModuleNotFoundError:
        # During docs build heavy deps are mocked, some optional modules may be missing.
        # Silently ignore to keep import lightweight.
        pass

# Clean up helper symbol.
del _import_module
