import torch

class SparseLinear(torch.nn.Module): 
    def __init__(self, indices, size, d=None, dtype=torch.float32):
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

        _, dst = indices 
        values = torch.randn(indices.size(1), dtype=dtype)

        # scale weight initialization
        # DEPRECATED! 
        if d is not None: 
            std = (2/d)**(0.5)
            values *= std

        # optimizers require dense parameters 
        self.values = torch.nn.Parameter(values)
        self.register_buffer('indices', indices)
        self.bias = torch.nn.Parameter(torch.zeros((self.M, 1), dtype=dtype))

    def forward(self, x, batching='loop'): 
        '''
        Assumes x is in shape: (B, N, 1), where B is batch dimension
        # weight shape is (N, M)
        Returns shape (B, M, 1)
        '''
        if batching == 'loop': 
            # slow but simple
            weight = torch.sparse_coo_tensor(indices=self.indices, values=self.values, size=self.size)
            return torch.stack([torch.sparse.mm(weight.T, xx) + self.bias for xx in x] , dim=0)
        elif batching == 'extend': 
            '''
            BUG: backward pass uses huge amount of memory...not sure why 
            '''
            B = x.size(0)
            X = x.view(-1, 1)
            batched_indices = torch.cat([self.indices + torch.tensor([[self.N], [self.M]], device=x.device)*i for i in range(B)], dim=-1)
            batched_values = self.values.repeat(B)
            weight = torch.sparse_coo_tensor(indices=batched_indices, values=batched_values, size=(self.N*B, self.M*B))
            # X ~ (B*N, 1)
            # weight (N*B, M*B)
            # output (M*B, 1)
            return torch.sparse.mm(weight.T, X).view(B, self.M, 1) + self.bias.view(1,-1,1)
        else: 
            raise ValueError(f'unrecognized `batching` input. Expected one of ["loop", "extend"], got: {batching}.')
