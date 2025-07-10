API Reference
=============

The API reference will be automatically generated from the GSNN source code.

.. note::

   The **autosummary** and **autodoc** extensions are enabled in ``conf.py``. Once the package is importable and the modules are stable, add them to the autosummary list below or generate stubs via ``sphinx-apidoc``.

.. currentmodule:: gsnn

.. autosummary::
   :toctree: generated
   :recursive:
   
   gsnn
   gsnn.models
   gsnn.models.GSNN
   gsnn.simulate
   gsnn.simulate.simulate
   gsnn.interpret
   gsnn.interpret.GSNNExplainer
   gsnn.interpret.IGExplainer
   gsnn.interpret.ContrastiveIGExplainer
   gsnn.interpret.extract_entity_function