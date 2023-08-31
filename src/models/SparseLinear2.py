import torch
import torch_geometric as pyg 


class Conv(pyg.nn.MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight, bias, size):
        
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        return out.view(-1, 1) + bias.view(-1, 1)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j.view(-1, 1)



class SparseLinear2(torch.nn.Module): 
    def __init__(self, indices, size, d=None, dtype=torch.float32, bias=True):
        '''
        
        We use `Kaiming` Weight Initialization, if `d` (layer dimension) is provided. Remember, each layer can be thought of as 
        several smaller, independantly connected networks, therefore we cannot use the dimensions provided by `size`. Fortunately, 
        we can make the assumption that each layer has the same number of independant `channels` and therefore can be initialized 
        with the same dimension value. 

        @misc{he2015delving,
        title={Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification}, 
        author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
        year={2015},
        eprint={1502.01852},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
}
        '''
        super().__init__() 

        self.N, self.M = size
        self.size = size

        self.conv = Conv()

        _, dst = indices 
        values = torch.randn(indices.size(1), dtype=dtype)

        # scale weight initialization
        if d is not None: 
            std = (2/d)**(0.5)
            values *= std

        # optimizers require dense parameters 
        self.values = torch.nn.Parameter(values)
        self.register_buffer('indices', indices.type(torch.long))
        if bias: self.bias = torch.nn.Parameter(torch.zeros((self.M, 1), dtype=dtype))

        # cache
        self.edge_index = None 
        self.edge_id = None 
        self.B = None
        self.bias_idx = None
        self.batched_size = None

    def forward(self, x): 
        '''
        Assumes x is in shape: (B, N, 1), where B is batch dimension
        weight shape is (N, M)
        Returns shape (B, M, 1)

        batch dimension is handled in `torch_geometric` fashion, e.g., concatenated batch graphs via incremented node idx 
        '''
        B = x.size(0)
        E = self.indices.size(1)

        edge_index = self.indices.repeat(1, B) + torch.stack((torch.arange(B, device=x.device).repeat_interleave(E)*self.N,
                                                                  torch.arange(B, device=x.device).repeat_interleave(E)*self.M), dim=0)
        edge_id = torch.arange(self.indices.size(1)).repeat(B)
        bias_idx = torch.arange(self.M).repeat(B)
        batched_size = (self.N*B, self.M*B)

        edge_id = edge_id.to(x.device)
        bias_idx = bias_idx.to(x.device)
        edge_index = edge_index.to(x.device)

        edge_weight = self.values[edge_id]

        if hasattr(self, 'bias'):
            bias = self.bias[bias_idx]
        else: 
            bias = torch.zeros((self.M, 1)[bias_idx], device=x.device)

        x = x.view(-1,1)

        x = self.conv(x, edge_index, edge_weight, bias, size=batched_size)

        x = x.view(B, -1, 1)

        return x

