
import torch
from torch.utils.checkpoint import checkpoint
from gsnn.models.SparseLinear import SparseLinear, batch_graphs
import warnings
from gsnn.models.ResBlock import ResBlock
from gsnn.models.utils import hetero2homo, get_conv_indices, node2edge, edge2node
from gsnn.models.NodeActivity import NodeActivity

class GSNN(torch.nn.Module): 

    def __init__(self, edge_index_dict, node_names_dict, channels, layers, dropout=0., nonlin=torch.nn.ELU, bias=True, 
                 share_layers=True, add_function_self_edges=True, norm='layer', init='degree_normalized', verbose=False, 
                 edge_channels=1, checkpoint=False, residual=True, norm_first=True, node_attn=False, attn_mlp_hidden=16,
                 node_mlp=False, node_mlp_hidden=16, node_activity=False, node_activity_hidden=16,
                 node_activity_dim=1, node_activity_temperature=1.0, edge_weight_dict=None):

        r"""Graph Structured Neural Network (GSNN) that constrains neural network architecture using a predefined graph structure.
        Unlike traditional GNNs that learn from graph structure, GSNN uses the graph to constrain which variables can directly 
        influence each other. The model operates on edge features rather than node features and supports cyclic graphs.

        The architecture uses three types of nodes:
            1. Input nodes: Represent observed variables
            2. Function nodes: Represent latent variables parameterized by neural networks
            3. Output nodes: Represent target variables

        Only function nodes are trainable; input and output nodes pass/receive information unchanged.

        Args:
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): Dictionary mapping edge types to edge indices.
                Expected keys are ('input', 'to', 'function'), ('function', 'to', 'function'), and 
                ('function', 'to', 'output'). Values should be tensors of shape :obj:`[2, num_edges]`.
            node_names_dict (Dict[str, List[str]]): Dictionary mapping node types ('input', 'function', 'output') 
                to their respective node names.
            channels (int): Number of hidden channels per function node.
            layers (int): Number of sequential sparse linear layers to propagate information across the graph.
            dropout (float, optional): Dropout probability. (default: :obj:`0.`)
            nonlin (torch.nn.Module, optional): Activation function. (default: :obj:`torch.nn.ELU`)
            bias (bool, optional): If set to :obj:`False`, the layer will not learn an additive bias. 
                (default: :obj:`True`)
            share_layers (bool, optional): If set to :obj:`True`, reuse layer parameters across all layers.
                (default: :obj:`True`)
            add_function_self_edges (bool, optional): If set to :obj:`True`, add self-connections to function nodes.
                (default: :obj:`True`)
            norm (str, optional): Normalization type (:obj:`'layer'`, :obj:`'batch'`, :obj:`'softmax'`, 
                :obj:`'groupbatch'`, :obj:`'edgebatch'`, :obj:`'rms'`, :obj:`'ema'`, :obj:`'channelema'` or :obj:`'none'`). (default: :obj:`'groupbatch'`)
            init (str, optional): Weight initialization strategy (:obj:`'xavier'` or :obj:`'kaiming'`).
                (default: :obj:`'xavier'`)
            verbose (bool, optional): If set to :obj:`True`, print debugging information. (default: :obj:`False`)
            edge_channels (int, optional): Number of latent edge feature channels to replicate.
                (default: :obj:`1`)
            checkpoint (bool, optional): If set to :obj:`True`, use gradient checkpointing to reduce memory usage.
                (default: :obj:`False`)
            residual (bool, optional): If set to :obj:`True`, add residual connections. (default: :obj:`True`)
            norm_first (bool, optional): If set to :obj:`True`, apply normalization before nonlinearity. (default: :obj:`True`)
            node_attn (bool, optional): If set to :obj:`True`, apply node attention. (default: :obj:`False`)
            attn_mlp_hidden (int, optional): Hidden dimension of the attention MLP. 
            node_mlp (bool, optional): If set to :obj:`True`, apply additional MLP processing per node to enhance 
                representational capacity while maintaining graph structure constraints. (default: :obj:`True`)
            node_mlp_hidden (int, optional): Hidden dimension size for the node MLP when enabled. 
                (default: :obj:`128`)
            node_activity (bool, optional): If set to :obj:`True`, enable per-function-node gating driven by
                external features (e.g., mutation/expression status). The gate is computed once per forward
                pass and broadcast to every channel of the corresponding node at every layer.
                (default: :obj:`False`)
            node_activity_hidden (int, optional): Hidden dimension of the node-activity MLP.
                (default: :obj:`16`)
            node_activity_dim (int, optional): Number of external feature channels per function node
                expected as input to the node-activity MLP. When :obj:`1`, the user may pass
                :obj:`x_fn` as :obj:`[B, Nf]` and it will be unsqueezed internally; otherwise
                :obj:`x_fn` must be :obj:`[B, Nf, node_activity_dim]`. (default: :obj:`1`)
            node_activity_temperature (float, optional): Sigmoid temperature applied to the node-activity
                logits. Lower values produce sharper (closer to 0/1) gates, higher values produce softer
                (closer to 0.5) gates. (default: :obj:`1.0`)
            edge_weight_dict (Dict[Tuple[str, str, str], Tensor], optional): Dictionary mapping edge types to edge weights.
                Expected keys are ('input', 'to', 'function'), ('function', 'to', 'function'), and 
                ('function', 'to', 'output'). Values should be tensors of shape :obj:`[num_edges]`.
                (default: :obj:`None`)

        Example:
            >>> # Define a simple graph with 2 input nodes, 1 function node, and 1 output node
            >>> edge_index_dict = {
            ...     ('input', 'to', 'function'): torch.tensor([[0, 1], [0, 0]]),  # 2 input edges
            ...     ('function', 'to', 'function'): torch.tensor([[0], [0]]),     # 1 self edge
            ...     ('function', 'to', 'output'): torch.tensor([[0], [0]])        # 1 output edge
            ... }
            >>> node_names_dict = {
            ...     'input': ['in1', 'in2'],
            ...     'function': ['func1'],
            ...     'output': ['out1']
            ... }
            >>> model = GSNN(
            ...     edge_index_dict=edge_index_dict,
            ...     node_names_dict=node_names_dict,
            ...     channels=16,
            ...     layers=3
            ... )
            >>> x = torch.randn(32, 2)  # batch_size=32, num_input_nodes=2
            >>> out = model(x)
            >>> print(out.shape)  # [32, 1] (batch_size, num_output_nodes)
        """
        super().__init__()

        # Optional: add multiple latent edge features per edge
        # NOTE: this will scale the total number of channels (be careful)
        if edge_channels > 1:
            edge_index_dict['function', 'to', 'function'] = edge_index_dict['function', 'to', 'function'].repeat(1, edge_channels)

        edge_index, input_node_mask, output_node_mask, \
            self.num_nodes, self.homo_names, self.edge_weights = hetero2homo(edge_index_dict, 
                                                                             node_names_dict, 
                                                                             edge_weight_dict)


        self.nonlin                     = nonlin
        self.bias                       = bias
        self.share_layers               = share_layers
        self.layers                     = layers 
        self.channels                   = channels
        self.add_function_self_edges    = add_function_self_edges
        self.verbose                    = verbose
        self.edge_channels              = edge_channels
        self.checkpoint                 = checkpoint
        self.norm                       = norm
        self.dropout                    = dropout
        self.residual                   = residual
        self.node_attn                  = node_attn
        self.attn_mlp_hidden            = attn_mlp_hidden
        self.norm_first                 = norm_first
        self.node_mlp                   = node_mlp
        self.node_mlp_hidden            = node_mlp_hidden
        self.node_activity              = node_activity
        self.node_activity_hidden       = node_activity_hidden
        self.node_activity_dim          = node_activity_dim
        self.node_activity_temperature  = node_activity_temperature

        if self.checkpoint: 
            # BUG:  checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
            #       with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
            #       /home/teddy/miniconda3/envs/gsnn-lib/lib/python3.12/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
            warnings.filterwarnings("ignore", category=FutureWarning)

        self.register_buffer('output_node_mask', output_node_mask)
        self.register_buffer('input_node_mask', input_node_mask)

        function_nodes = (~(input_node_mask | output_node_mask)).nonzero(as_tuple=True)[0]
        if add_function_self_edges: 
            if verbose: print('Augmenting `edge index` with function node self-edges.')
            edge_index = torch.cat((edge_index, torch.stack((function_nodes, function_nodes), dim=0)), dim=1)

        self.register_buffer('edge_index', edge_index)
        self.E = self.edge_index.size(1)
        self.N = len(node_names_dict['input']) + len(node_names_dict['function']) + len(node_names_dict['output'])

        func_edge_mask = torch.isin(edge_index[0], function_nodes) & torch.isin(edge_index[1], function_nodes) # edges from function -> function / e.g., not an input or output edge 
        inp_edge_mask = torch.isin(edge_index[0], input_node_mask.nonzero(as_tuple=True)[0]) & torch.isin(edge_index[1], function_nodes) # edges from input -> function
        output_edge_mask = ~(func_edge_mask | inp_edge_mask)
        self.register_buffer('function_edge_mask', func_edge_mask) 
        self.register_buffer('input_edge_mask', inp_edge_mask)
        self.register_buffer('output_edge_mask', output_edge_mask)

        self.indices_params = get_conv_indices(edge_index, channels, function_nodes)
        w_in_indices, w_out_indices, w_in_size, w_out_size, channel_groups = self.indices_params

        if self.share_layers: 
            lin_in = SparseLinear(indices=w_in_indices, size=w_in_size, bias=bias, init=init)
            lin_out = SparseLinear(indices=w_out_indices, size=w_out_size, bias=bias, init=init)
        else: 
            lin_in = None
            lin_out = None

        if self.node_activity: 
            self.node_activity_model = NodeActivity(channel_groups,
                                                    activity_dim=node_activity_dim,
                                                    dropout=dropout,
                                                    temperature=node_activity_temperature,
                                                    channels=node_activity_hidden)
        else: 
            self.node_activity_model = None

        self.ResBlocks = torch.nn.ModuleList([ResBlock(bias             = self.bias,    
                                                       nonlin           = self.nonlin, 
                                                       dropout          = dropout, 
                                                       norm             = norm, 
                                                       init             = init, 
                                                       indices_params   = self.indices_params,
                                                       lin_in           = lin_in,
                                                       lin_out          = lin_out,
                                                       node_attn        = self.node_attn,
                                                       attn_mlp_hidden  = self.attn_mlp_hidden,
                                                       norm_first       = self.norm_first,
                                                       residual         = self.residual,
                                                       node_mlp         = self.node_mlp,
                                                       node_mlp_hidden  = self.node_mlp_hidden, 
                                                       edge_index       = self.edge_index,
                                                       edge_weight      = self.edge_weights) for i in range(self.layers)])
        
        self._B             = None
        self._batch_params  = None

        self.scale = torch.tensor(self.layers**(0.5), dtype=torch.float32)


    def get_batch_params(self, B, device): 
        r"""Retrieves or computes batch-specific indexing parameters for sparse linear layers.

        This method caches the batch parameters to avoid recomputing them for the same batch size.
        The parameters are used to efficiently perform batched sparse matrix operations.

        Args:
            B (int): Batch size.
            device (torch.device): Device on which to place the computed parameters.

        Returns:
            tuple: A tuple containing:
                - batched_indices_in (Tensor): Batched indices for input sparse linear layer
                - batched_indices_out (Tensor): Batched indices for output sparse linear layer

        Example:
            >>> model = GSNN(edge_index_dict, node_names_dict, channels=16, layers=3)
            >>> # Get batch parameters for batch size 32
            >>> batch_params = model.get_batch_params(32, torch.device('cuda'))
            >>> # Parameters are cached for subsequent calls
            >>> same_params = model.get_batch_params(32, torch.device('cuda'))
            >>> # Different batch size triggers recomputation
            >>> new_params = model.get_batch_params(64, torch.device('cuda'))
        """

        if (self._B == B) and (self._batch_params is not None): 
            # caching batch params
            return self._batch_params
        else: 
            self._B = B
            # precompute edge batching so it doesn't have to be done in every resblock 
            batched_edge_indices_in = batch_graphs(N=self.ResBlocks[0].lin_in.N,
                                                            M=self.ResBlocks[0].lin_in.M, 
                                                            edge_index = self.ResBlocks[0].lin_in.indices, 
                                                            B=B, 
                                                            device=device)
            
            batched_edge_indices_out = batch_graphs(N=self.ResBlocks[0].lin_out.N,
                                                            M=self.ResBlocks[0].lin_out.M, 
                                                            edge_index = self.ResBlocks[0].lin_out.indices, 
                                                            B=B, 
                                                            device=device)
            
            self._batch_params = (batched_edge_indices_in, batched_edge_indices_out)
        
        return self._batch_params
    
    def _preprocess(self, x, node_mask=None): 
        r"""Preprocesses input features before applying residual blocks.

        This method:
            1. Converts input node features to a full node feature tensor
            2. Converts node features to edge features
            3. Applies node masking if provided
            4. Computes batch parameters for sparse operations

        Args:
            x (Tensor): Input features of shape :obj:`[batch_size, num_input_nodes]`.
            node_mask (Tensor, optional): node weights for function nodes of shape 
                :obj:`[B, num_nodes]`. (default: :obj:`None`)

        Returns:
            tuple: A tuple containing:
                - x (Tensor): Edge features of shape :obj:`[batch_size, num_edges]`
                - batch_params (tuple): Batch-specific parameters for sparse operations
                - modules (List[ResBlock]): List of residual blocks to apply

        Example:
            >>> model = GSNN(edge_index_dict, node_names_dict, channels=16, layers=3)
            >>> x = torch.randn(32, 2)  # [batch_size, num_input_nodes]
            >>> # Basic preprocessing
            >>> edge_feats, params, blocks = model._preprocess(x, None)
            >>> print(edge_feats.shape)  # [32, num_edges]
            >>> # With node masking
            >>> node_mask = torch.ones(32, 4)  # [B, num_nodes]
            >>> edge_feats, params, blocks = model._preprocess(x, node_mask)
        """

        B = x.size(0)
        x_node = torch.zeros((B, self.num_nodes), device=x.device, dtype=torch.float32)
        idx = self.input_node_mask.nonzero(as_tuple=True)[0].unsqueeze(0).expand(B, -1)  # Shape: (B, num_input_nodes)
        x_node = x_node.scatter_add(1, idx, x)

        x = node2edge(x_node, self.edge_index)  # convert x to edge-indexed
        
        modules = [blk for blk in self.ResBlocks]

        # faster if we do this up front 
        if node_mask is not None: 
            #node_mask = torch.stack([torch.isin(modules[0].channel_groups, node_mask[i].nonzero(as_tuple=True)[0]) for i in range(node_mask.size(0))], dim=0)
            node_mask = node_mask[:, modules[0].channel_groups]
        for mod in modules: mod.set_node_mask(node_mask)

        batch_params = self.get_batch_params(B, x.device)

        return x, batch_params, modules
    
    def prune(self, threshold=1e-2, verbose=False): 
        r"""Prunes the model by removing channels with small weights.

        This method removes channels whose maximum absolute weight value across all layers is below
        the specified threshold. This can significantly reduce model size while maintaining performance.
        Remember to reinitialize the optimizer after pruning if using during training.

        Args:
            threshold (float, optional): The threshold below which weights are considered insignificant.
                (default: :obj:`1e-2`)
            verbose (bool, optional): If set to :obj:`True`, print pruning statistics.
                (default: :obj:`False`)

        Returns:
            int: Number of parameters removed by pruning.

        Example:
            >>> # Create a model with 16 channels per function node
            >>> model = GSNN(edge_index_dict, node_names_dict, channels=16, layers=3)
            >>> # Train the model...
            >>> # Prune channels with small weights
            >>> removed_params = model.prune(threshold=1e-2, verbose=True)
            >>> print(f'Removed {removed_params} parameters')
        """

        w1 = [] ; w2 = []
        for mod in self.ResBlocks: 
            w1.append(mod.lin_in.values)
            w2.append(mod.lin_out.values)
        w1 = torch.stack(w1, dim=0)
        w2 = torch.stack(w2, dim=0)

        w1_abs_max = torch.max(torch.abs(w1), dim=0).values
        w2_abs_max = torch.max(torch.abs(w2), dim=0).values

        keep_idxs1 = (w1_abs_max >= threshold).nonzero(as_tuple=True)[0]
        keep_idxs2 = (w2_abs_max >= threshold).nonzero(as_tuple=True)[0]

        for mod in self.ResBlocks:
            mod.lin_in.prune(keep_idxs1)
            mod.lin_out.prune(keep_idxs2)

            # reset cached batch params 
            self._B = None; self._batch_params = None

        if verbose: 
            print(f'Pruned in/out: {w1.size(1) - len(keep_idxs1)}, {w2.size(1) - len(keep_idxs2)} -> remaining in/out: {len(keep_idxs1)}, {len(keep_idxs2)}')

        # return the number of parameters removed 
        return w1.size(1) - len(keep_idxs1) + w2.size(1) - len(keep_idxs2)

        

    def forward(self, x, node_mask=None, edge_mask=None, ret_edge_out=False, e0=None, node_errs=None, x_fn=None):
        r"""Implements the forward pass of the GSNN model.

        The model first converts node features to edge features, then applies a sequence of sparse linear 
        transformations constrained by the graph structure. Each layer consists of:
            1. Input transformation (W_in)
            2. Normalization (optional)
            3. Nonlinearity
            4. Output transformation (W_out)
            5. Residual connection (optional)

        Args:
            x (Tensor): Input node features of shape :obj:`[batch_size, num_input_nodes]`.
            node_mask (Tensor, optional): Boolean mask for function nodes of shape :obj:`[batch_size, num_nodes]`.
                If provided, masks out specific function nodes during computation. (default: :obj:`None`)
            edge_mask (Tensor, optional): Boolean mask for edges of shape :obj:`[batch_size, num_edges]`.
                If provided, masks out specific edges during computation. (default: :obj:`None`)
            ret_edge_out (bool, optional): If set to :obj:`True`, return edge-level features instead of 
                node-level features. (default: :obj:`False`)
            e0 (Tensor, optional): Initial edge features of shape :obj:`[batch_size, num_edges]`. Used for 
                inferring input errors. (default: :obj:`None`)
            node_errs (List[Tensor], optional): List of node errors per layer, each of shape 
                :obj:`[batch_size, num_nodes]`. Length must match number of layers. (default: :obj:`None`)
            x_fn (Tensor, optional): Function node features of shape :obj:`[batch_size, num_function_nodes]`. Used for 
                computing node activity. (default: :obj:`None`)

        Returns:
            Tensor: If :obj:`ret_edge_out=False`, returns node-level output features of shape 
            :obj:`[batch_size, num_output_nodes]`. Otherwise, returns edge-level features of shape 
            :obj:`[batch_size, num_edges]`.

        Example:
            >>> # Using the model from the class example
            >>> x = torch.randn(32, 2)  # batch_size=32, num_input_nodes=2
            >>> # Basic forward pass
            >>> out = model(x)
            >>> print(out.shape)  # [32, 1]
            >>> # Get edge-level features
            >>> edge_out = model(x, ret_edge_out=True)
            >>> print(edge_out.shape)  # [32, 4] (batch_size, num_edges)
            >>> # Using masks
            >>> node_mask = torch.ones(32, 4)  # [batch_size, num_nodes]
            >>> edge_mask = torch.ones(32, 4)  # [batch_size, num_edges]
            >>> out = model(x, node_mask=node_mask, edge_mask=edge_mask)
            >>> print(out.shape)  # [32, 1]
        """

        ############ in dev ################
        if node_errs is None:
            node_errs = [None]*self.layers
        else: 
            if len(node_errs) != self.layers: 
                raise ValueError('node_errs must be the same length as the number of layers')
        ###################################

        x, batch_params, modules = self._preprocess(x, node_mask)

        if e0 is not None:
            x = x + e0
        
        # mask input edges (otherwise input edges get missed)
        if edge_mask is not None: x = x*edge_mask

        # Node activity 
        fn_activity = None 
        if self.node_activity_model is not None:
            if x_fn is None:
                raise ValueError(
                    "`x_fn` must be provided when `node_activity=True`."
                )
            fn_activity = self.node_activity_model(x_fn)


        if self.checkpoint and self.training: x.requires_grad_(True)
        for i, (mod,nerr) in enumerate(zip(modules, node_errs)): 

            if self.checkpoint and self.training: 
                x = checkpoint(mod, x, batch_params, node_err=nerr, fn_activity=fn_activity, use_reentrant=False).squeeze(-1)
            else: 
                x = mod(x, batch_params, node_err=nerr, fn_activity=fn_activity).squeeze(-1)

            if edge_mask is not None: x = x*edge_mask

        # under assumption that each layer output is iid unit normal (weak assumption since layer outputs will be correlated)
        # then x = N(0,1) + N(0,1) + ... + N(0,1) = N(0, sqrt(layers))
        if self.residual: x = x / self.scale

        if ret_edge_out: 
            return x
        else: 
            out = edge2node(x, self.edge_index, self.output_node_mask)[:, self.output_node_mask]
            return out

    def get_node_activations(self, x, agg='sum', inference=True):

        with torch.inference_mode(mode=inference):  
            for mod in self.ResBlocks: mod._store_activations = True
            preds = self.forward(x)

            activations = []
            for mod in self.ResBlocks:  
                activations.append(mod._last_activation.squeeze(-1)) 
                del mod._last_activation

            activations = torch.stack(activations, dim=0) 

            if agg == 'sum': 
                activations = activations.sum(dim=0) 
            elif agg == 'mean': 
                activations = activations.mean(dim=0)
            elif agg == 'max': 
                activations = activations.max(dim=0).values
            elif agg == 'last': 
                activations = activations[-1, :, :] # L,B,C*N
            elif agg == 'all': 
                activations = activations.permute(1, 0, 2) # (L, B, CN) -> (B, L, CN)
            else: 
                raise ValueError(f'Invalid aggregation method: {agg}')

            for mod in self.ResBlocks: mod._store_activations = False 

            node_activation_dict = {}
            for i in range(int(self.ResBlocks[0].channel_groups.max().item()) + 1): 
                ixs = (self.ResBlocks[0].channel_groups == i).to(activations.device)
                
                if len(activations.shape) == 2: 
                    node_acts = activations[:, ixs] 
                elif len(activations.shape) == 3: 
                    B, L, CN = activations.shape
                    node_acts = activations[:, :, ixs].reshape(B, -1)

                node_name = self.homo_names[i] 
                node_activation_dict[node_name] = node_acts 

            return node_activation_dict

    def get_node_attention(self, x):
        """Return per-layer node-level attention weights.

        Parameters
        ----------
        x : Tensor (B, num_input_nodes)
            Input features; typically supply a single sample (B=1).

        Returns
        -------
        Dict[str, Tensor]
            Mapping from node name to a tensor of shape (L, B) with attention
            weights per layer (L) and batch element (B).
        """
        assert self.node_attn, "Node attention is not enabled in this model."

        # Run a forward pass to cache _last_alpha inside each NodeAttention
        _ = self.forward(x)

        layer_attns: list[torch.Tensor] = []  # each entry: (B, N_fn)
        B = x.size(0)

        # mask and size for function nodes (those with attention)
        function_mask = ~(self.input_node_mask | self.output_node_mask)
        N_fn = int(function_mask.sum().item())
        device = self.edge_index.device

        for blk in self.ResBlocks:
            if blk.node_attn is not None and hasattr(blk.node_attn, "_last_alpha"):
                alpha = blk.node_attn._last_alpha  # (B, N_fn)
                if alpha is None:
                    raise RuntimeError("Attention not stored; ensure NodeAttention forward was called.")
                # Ensure expected 2D shape (B, N_fn)
                if alpha.dim() == 1:
                    alpha = alpha.unsqueeze(0)
                layer_attns.append(alpha)
            else:
                # If no attention in this block, fill with zeros (B, N_fn)
                layer_attns.append(torch.zeros(B, N_fn, device=device))

        # Stack into (L, B, N_fn)
        layer_attns = torch.stack(layer_attns, dim=0)

        # Build dict: node name -> (L, B)
        function_idxs = function_mask.nonzero(as_tuple=True)[0].tolist()
        node_attn_dict = {}
        for j, node_idx in enumerate(function_idxs):
            node_name = str(self.homo_names[int(node_idx)])
            node_attn_dict[node_name] = layer_attns[..., j]

        return node_attn_dict
