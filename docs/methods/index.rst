Methods
=======

Graph Structured Neural Network (GSNN)
--------------------------------------

Graph Structured Neural Networks (GSNN) are designed to model biological signalling networks by **constraining** a neural network with the structure described by auser-defined *graph* :math:`\mathcal{G}`.  The graph encodes the molecular entities (*nodes*) and their interactions (*edges*), thereby defining which variables may directly influence each other during learning.

The architecture employs three types of nodes:

* **Input nodes** – observed variables
* **Function nodes** – latent variables parameterised by neural networks
* **Output nodes** – target variables

Only *function nodes* are trainable; input and output nodes pass information through the network unchanged.

.. image:: ./GSNN_overview.png
   :width: 100%
   :alt: GSNN Overview
   :align: center
   A toy example demonstrating how any given graph structure can be formulated as a feed forward neural network with masked weight matrices. Each yellow node in the left graph represents a fully-connected 1-layer neural network with two hidden channels (Note: function node neural networks can optionally be multi-layer). Panel A describes the structural graph ($\mathcal{G}$) which imposes constraints on the GSNN model. Panel B depicts how the edge latent values ($e_i$) can be updated in a single forward pass. Note that panel B shows sparse weight matrices, where the missing edge connections are equal to zero. The plus sign in panel B indicates a skip connection from the previous layer.

.. note::
    Unlike GNNs, where latent representations typically characterize the state of a *node*, GSNN latent representations characterize the state of an *edge*. This allows the GSNN method to learn nonlinear multivariate relationships between input edges and output edges and still be applicable to cyclic graphs.


Function Nodes
^^^^^^^^^^^^^^
Each function node :math:`f_n` is implemented as a small fully-connected feed-forward network whose shape is determined by the local topology of :math:`\mathcal{G}`:

* **Inputs**  – equal to the in-degree of node *n*
* **Outputs** – equal to the out-degree of node *n*
* **Hidden channels / layers** – user-defined hyper-parameters

.. note::
   *GSNN layers* (*L*) – sequential sparse linear layers that propagate information across the entire graph.


Layer Updates with Masked Linear Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A single GSNN layer updates **edge** representations via a *sparselinear operation*.  The weight matrix has shape :math:`(E, N \times C)` where

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

* **Normalisation** – Instead of batch normalisation (ineffective with small batches), each function node applies **layer normalisation** to prevent data leakage across nodes.
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





