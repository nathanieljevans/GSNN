Tutorials
=========

A curated collection of **step-by-step notebooks** that demonstrate GSNN's capabilities:

.. toctree::
   :maxdepth: 1
   :hidden:

   00_dev
   01_basic
   02_simulate
   03_comparison
   04_reinforce
   05_bayesopt
   06_checkpointing_and_compiling
   07_UQ_with_HyperNetworks
   08_optimal_transport_with_gsnn
   09_inferring_missing_input_edges
   10_iterative_weight_pruning

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tutorial
     - What you'll learn
   * - :doc:`Development notebook <00_dev>`
     - Rapid prototyping and debugging of GSNN layers inside a development environment.
   * - :doc:`Getting started (basic workflow) <01_basic>`
     - Build and train a minimal GSNN model—your "Hello, graphs!" introduction.
   * - :doc:`Simulating structured data <02_simulate>`
     - Use GSNN to simulate graph dynamics over time.
   * - :doc:`Comparing to baseline GNNs <03_comparison>`
     - Measure GSNN expressiveness and performance against standard GNN architectures.
   * - :doc:`Reinforcement learning with GSNN <04_reinforce>`
     - Combine GSNN with RL to learn optimal prior-knowledge selection on graphs.
   * - :doc:`Bayesian optimisation <05_bayesopt>`
     - Perform Bayesian optimisation with GSNN-based surrogate models.
   * - :doc:`Checkpointing & Torch compile <06_checkpointing_and_compiling>`
     - Speed up training and reduce memory with checkpointing and the new ``torch.compile``.
   * - :doc:`Uncertainty quantification <07_UQ_with_HyperNetworks>`
     - Estimate prediction uncertainty via hyper-network weight sampling.
   * - :doc:`Optimal transport on graphs <08_optimal_transport_with_gsnn>`
     - Solve optimal transport problems using GSNN layers.
   * - :doc:`Inferring missing edges <09_inferring_missing_input_edges>`
     - Reconstruct missing or corrupted edges with generative GSNNs.
   * - :doc:`Iterative weight pruning <10_iterative_weight_pruning>`
     - Compress GSNN models by pruning weights while maintaining accuracy.

.. admonition:: Adding new tutorials

   Place new ``.ipynb`` or ``.rst`` files in this directory and add them to the hidden
   toctree above to have them automatically rendered and linked here. 