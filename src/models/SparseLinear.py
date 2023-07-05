import torch

class SparseLinear(torch.nn.Module): 
    def __init__(self, indices, size, d=None, dtype=torch.float32):
        '''
        
        We use `Kaiming` Weight Initialization, if `d` (layer dimension) is provided. Remember, each layer can be thought of as 
        several smaller, independantly connected networks, therefore we cannot use the dimensions provided by `size`. Fortuntaly, 
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

        _, dst = indices 
        values = torch.randn(indices.size(1), dtype=dtype)

        # scale weight initialization
        if d is not None: 
            std = (2/d)**(0.5)
            values *= std

        # optimizers require dense parameters 
        self.values = torch.nn.Parameter(values)
        self.register_buffer('indices', indices)
        self.size = size
        #self.weight = torch.sparse_coo_tensor(indices=indices, values=values, size=size)
        self.bias = torch.nn.Parameter(torch.zeros((size[1], 1), dtype=dtype))

    def forward(self, x): 
        '''
        Assumes x is in shape: (B, *, 1), where B is batch dimension

        Returns shape (B, **, 1)
        '''
        weight = torch.sparse_coo_tensor(indices=self.indices, values=self.values, size=self.size)
        return torch.stack([torch.sparse.mm(weight.T, xx) + self.bias for xx in x] , dim=0)
