Methods
=======

Graph Structured Neural Network (GSNN)
--------------------------------------

Graph Structured Neural Networks (GSNN) were originally designed to model biological signalling networks by **constraining** a neural network with the structure described by a user-defined *graph* :math:`\mathcal{G}`.  The graph encodes the molecular entities (*nodes*) and their interactions (*edges*), thereby defining which variables may directly influence each other during learning.

The architecture employs three types of nodes:

* **Input nodes** – observed variables
* **Function nodes** – latent variables parameterised by neural networks
* **Output nodes** – target variables

Only *function nodes* are trainable; input and output nodes pass/receive information through the network unchanged.

.. image:: ./gsnn_overview.png
   :width: 100%
   :alt: GSNN Overview
   :align: center

A toy example demonstrating how any given graph structure can be formulated as a feed-forward neural network with masked weight matrices. Each yellow node in the left graph represents a fully-connected one-layer neural network with two hidden channels (function nodes). Panel A shows the structural graph (:math:`\mathcal{G}`) that constrains the GSNN model, while panel B depicts how edge latent values (:math:`e_i`) are updated in a single forward pass. Sparse weight matrices omit nonexistent edges, and the ⊕ symbol indicates a residual connection from the previous layer.

.. note::
    Unlike GNNs, where latent representations typically characterize the state of a *node*, GSNN latent representations characterize the state of an *edge*. This allows the GSNN method to learn nonlinear multivariate relationships between input edges and output edges and still be applicable to cyclic graphs.


Function Nodes
^^^^^^^^^^^^^^
Each function node :math:`f_n` is implemented as a small fully-connected feed-forward network whose shape is determined by the local topology of :math:`\mathcal{G}`:

* **Inputs**  – equal to the in-degree of node *n*
* **Outputs** – equal to the out-degree of node *n*
* **Hidden channels / layers** – user-defined hyper-parameters. While GSNN could theoretically use multi-layer neural networks to parameterize function nodes, we have found that single-layer networks are sufficient for most applications and currently do not support multi-layer networks.

.. note::
    To avoid confusion, we use the term *layer* to refer to the number of sequential sparse linear layers that propagate information across the entire graph. The neural networks that parameterize function nodes are fixed to a single layer.


Layer Updates with Masked Linear Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A single GSNN layer updates **edge** representations via a *sparse linear operation*.  The weight matrix has shape :math:`(E, N \times C)` where

* :math:`E` – number of edges in :math:`\mathcal{G}`
* :math:`N` – number of function nodes
* :math:`C` – hidden channels per function node

.. note::
    There is **no parameter sharing** between function nodes—each learns a distinct mapping from its inputs to its outputs. That said, parameters can optionally be shared across layers.

Iterating the update *L* times enables information to travel a path length of *L* across the input graph.


Sparse Implementation
^^^^^^^^^^^^^^^^^^^^^
A dense implementation of the masked matrices would quickly exhaust memory on realistic graphs.  Instead, GSNN stores the matrices as **sparse tensors**, reducing both memory and compute.  The current PyTorch sparse backend is not optimised for mini-batching, so GSNN leverages **PyTorch Geometric** for fast batched sparse matrix multiplication, especially on GPUs.


Residual Connections & Normalisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
GSNN is [optionally] a residual architecture where the layer output is added to its input:

.. math::

   x_{l+1} = F(x_l) + x_l

Residual connections allow the model to learn *edge latency*—the temporal lag between upstream and downstream signals—and alleviate vanishing gradients in deep networks.

* **Normalisation** – We provide several normalization options:
    * **None** – No normalization is applied.
    * **Batch** – Batch normalization is applied to the entire graph. This approach works well for large batches and is applicable to small channel sizes.
    * **Layer** – Layer normalization is applied within each function node. This approach works well for small batches with large channel sizes. 
    * **Softmax** – Softmax normalization is applied to the entire graph.
* **Self-edges** – Optional self-connections let a node incorporate its previous-layer state.
* **Parameter sharing** – While GSNN supports weight sharing across layers, empirical results typically show better performance when each layer has its own parameters.


Weight Initialisation
^^^^^^^^^^^^^^^^^^^^^
GSNN offers both **Kaiming/He** and **Xavier/Glorot** initialisation adapted to the graph setting.  Let :math:`D_i^{in}` and :math:`D_i^{out}` be the in- and out-degree of function node *i* in :math:`\mathcal{G}`.  Then

.. math::

   w^{\text{kaiming}}_i &\sim \mathcal{N}\!\bigl(0, \tfrac{2}{D_i^{in}}\bigr) \\
   w^{\text{xavier}}_i  &\sim \mathcal{N}\!\bigl(0, \tfrac{2}{D_i^{in}+D_i^{out}}\bigr)

Using degree-aware fan-in/out preserves the variance of activations despite the sparse, non-uniform connectivity.


Efficient Mini-Batching
^^^^^^^^^^^^^^^^^^^^^^
PyTorch's native sparse operations remain slow for large batches.  GSNN therefore reformulates the masked linear layers as a **PyTorch Geometric graph convolution**, gaining substantial speed-ups during training and inference—particularly on GPUs.


Gradient Checkpointing
^^^^^^^^^^^^^^^^^^^^^^

To reduce memory usage, GSNN supports **gradient checkpointing** at each layer, which substantially reduces memory usage at the cost of some compute.





