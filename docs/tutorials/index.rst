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
   :widths: 25 75

   * - Notebook
     - Description
   * - :doc:`00_dev`
     - Rapid prototyping and debugging of GSNN layers inside a development notebook.
   * - :doc:`01_basic`
     - Build and train a minimal GSNN model — your "Hello, graphs!" introduction.
   * - :doc:`02_simulate`
     - Use GSNN to simulate graph dynamics over time.
   * - :doc:`03_comparison`
     - Compare GSNN performance and expressiveness against baseline GNN architectures.
   * - :doc:`04_reinforce`
     - Integrate GSNN with reinforcement learning for decision-making on graphs.
   * - :doc:`05_bayesopt`
     - Perform Bayesian optimisation with GSNN-based surrogate models.
   * - :doc:`06_checkpointing_and_compiling`
     - Speed up training with Torch compile and seamlessly checkpoint large GSNNs.
   * - :doc:`07_UQ_with_HyperNetworks`
     - Estimate prediction uncertainty via hyper-network based weight sampling.
   * - :doc:`08_optimal_transport_with_gsnn`
     - Solve optimal transport problems on graphs using GSNN layers.
   * - :doc:`09_inferring_missing_input_edges`
     - Infer missing or corrupted edges in input graphs with generative GSNNs.
   * - :doc:`10_iterative_weight_pruning`
     - Compress models by iterative weight pruning while maintaining accuracy.

.. note::

   Place individual tutorial files (``.rst`` or notebooks via ``nbsphinx``) in this directory. They will be automatically picked up by the glob directive above. 