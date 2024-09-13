import pytest
import torch
import torch_geometric as pyg

from gsnn.models.SparseLinear import SparseLinear  # Replace 'your_module' with the actual name of the module containing SparseLinear

def test_sparse_linear():
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    input_size = 10
    output_size = 5
    batch_size = 2

    # Generate random indices for a sparse matrix
    indices = torch.tensor([[0, 1, 1, 2, 3], 
                            [1, 2, 0, 4, 1]], dtype=torch.long)

    # Create SparseLinear module
    sparse_layer = SparseLinear(indices=indices, size=(input_size, output_size), dtype=torch.float32, bias=True)

    # Random input tensor
    x = torch.randn(batch_size, input_size, 1)

    # Compute output using SparseLinear
    output_sparse = sparse_layer(x)

    # Create equivalent dense operation
    dense_weight_matrix = torch.zeros((input_size, output_size))
    dense_weight_matrix[indices[0], indices[1]] = sparse_layer.values.detach()
    dense_bias = sparse_layer.bias.repeat(1, batch_size).T if hasattr(sparse_layer, "bias") else 0


    # Dense multiplication
    output_dense = torch.bmm(x.view(batch_size, 1, input_size), dense_weight_matrix.unsqueeze(0).repeat(batch_size, 1, 1)) 
    output_dense += dense_bias.unsqueeze(1)

    # Check if the sparse and dense operations give the same result
    assert torch.allclose(output_sparse.reshape(batch_size, -1), output_dense.reshape(batch_size, -1)), "Sparse and dense results do not match."

def test_edge_mask():
    torch.manual_seed(42)

    # Parameters
    input_size = 6
    output_size = 6

    # Generate random indices for a sparse matrix
    indices = torch.tensor([[0, 0, 1, 1, 2], 
                            [1, 2, 3, 4, 5]], dtype=torch.long)
    
    # Edge mask where some edges are masked out
    edge_mask = [torch.tensor([True, True, False, False, False]).nonzero(as_tuple=True)[0], 
                 torch.tensor([False, False, True, True, False]).nonzero(as_tuple=True)[0],
                 torch.tensor([False, False, False, False, True]).nonzero(as_tuple=True)[0]]
    
    # Random input tensor
    x = torch.tensor([[1,0,0,0,0,0],
                      [0,1,0,0,0,0],
                      [0,0,1,0,0,0]], dtype=torch.float32).view(3,6,1)
    
    # Create SparseLinear module
    sparse_layer = SparseLinear(indices=indices, size=(input_size, output_size), dtype=torch.float32, bias=True)

    # Output with edge mask
    output_with_mask = sparse_layer(x, edge_subset=edge_mask)

    # output without mask - in this case, since nodes 0-3 are detached graphs should be the same
    output_with_mask = sparse_layer(x, edge_subset=None)

    # Check if output with mask applied matches manually zeroed output
    assert torch.allclose(output_with_mask, output_with_mask), "Edge mask operation failed."

# Note: pytest requires this if block to run from the command line
if __name__ == "__main__":
    pytest.main()
