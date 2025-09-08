import torch
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
import torch


import torch 
import numpy as np 

def root_mean_squared_picp_error(pred_dist, y_true, alphas=torch.linspace(0.01, 0.95, 10)): 
    ''''''
    return np.mean([(compute_picp(pred_dist, y_true, alpha=alpha)[0] - (1-alpha))**2 for alpha in alphas])**(0.5)

def compute_picp(pred_dist, y_true, alpha=0.05, N=100):
    '''
    Compute the Prediction Interval Coverage Probability (PICP) for given predictions and true values.

    Parameters:
    - pred_dist (torch.distributions.Distribution): Predicted probability distribution.
    - y_true (torch.Tensor): Actual values to compare against.
    - alpha (float, optional): Significance level for prediction interval. Default is 0.05 for 95% PICP.

    Returns:
    - float: The PICP value.

    Note:
    If you're computing a 95% Prediction Interval (which corresponds to an alpha of 0.05), 
    a perfectly calibrated model would have a PICP score of 0.95. This means that 95% of the true values 
    fall within the predicted intervals.
    '''

    rvs = pred_dist.sample((N,)) # (N, B)
    
    # Calculate the lower and upper bounds of the prediction interval
    #lower_bound = pred_dist.icdf(torch.tensor([alpha/2], device=y_true.device)) #pred_dist.icdf(torch.tensor([alpha/2], device=y_true.device))
    #upper_bound = pred_dist.icdf(torch.tensor([1 - alpha/2], device=y_true.device)) #pred_dist.icdf(torch.tensor([1 - alpha/2], device=y_true.device))
    lower_bound = rvs.quantile(torch.tensor([alpha/2], device=y_true.device), dim=0).squeeze(0)
    upper_bound = rvs.quantile(torch.tensor([1 - alpha/2], device=y_true.device), dim=0).squeeze(0)

    # Check if the true values lie within the prediction interval
    is_inside = (y_true >= lower_bound) & (y_true <= upper_bound)
    
    # Compute PICP
    picp = is_inside.float().mean()
    
    return picp.item()

def compute_ECE(pred_dist, y_true, num_intervals=10):
    '''
    Compute the Expected Calibration Error (ECE) using the PICP at different confidence levels.

    Parameters:
    - pred_dist (torch.distributions.Distribution): Predicted probability distribution.
    - y_true (torch.Tensor): Actual values to compare against.
    - num_intervals (int, optional): Number of confidence intervals to use for calibration. Default is 10.

    Returns:
    - float: The ECE value.
    '''

    ece = 0.0

    # Iterate over a range of confidence levels
    for i in range(1, num_intervals + 1):
        # Calculate the alpha for the current interval
        alpha = 1 - i / num_intervals

        # Compute PICP for the current confidence level
        picp = compute_picp(pred_dist, y_true, alpha)

        # The expected coverage for this confidence level
        expected_coverage = 1 - alpha

        # Accumulate the absolute difference between PICP and expected coverage
        ece += abs(picp - expected_coverage)

    # Normalize by the number of intervals
    ece /= num_intervals

    return ece

def dbscan_silhouette_score(embeddings, max_eps=5, min_samples=5):
    """
    Perform DBSCAN clustering on the embeddings and calculate the Silhouette Score.

    Parameters:
    - embeddings: torch.Tensor of shape (num_nodes, embedding_dim), the node embeddings.
    - eps: float, the maximum distance between two samples for them to be considered as in the same neighborhood (DBSCAN parameter).
    - min_samples: int, the number of samples in a neighborhood for a point to be considered a core point (DBSCAN parameter).

    Returns:
    - score: float, the silhouette score of the clustering.
    - labels: torch.Tensor, the cluster labels for each node (-1 means noise).
    """
    # Convert embeddings to numpy for DBSCAN
    embeddings_np = embeddings.cpu().detach().numpy()

    # scale 
    embeddings_np = (embeddings_np - embeddings_np.mean(0))/(embeddings_np.std(0) + 1e-8)

    # Step 1: Apply DBSCAN clustering
    db = OPTICS(max_eps=max_eps, min_samples=min_samples).fit(embeddings_np)
    labels = db.labels_

    # Step 2: Check if valid clusters are formed (more than 1 unique cluster label)
    if len(set(labels)) <= 1:
        return -1, torch.tensor(labels)  # If no clusters or only noise, return invalid score

    # Step 3: Compute the silhouette score (ignoring noise points with label -1)
    score = silhouette_score(embeddings_np, labels, metric='euclidean')

    return score, torch.tensor(labels)


def neighborhood_preservation_score(edge_index, embeddings, k=2):
    """
    Compute the neighborhood preservation score.
    
    Parameters:
    - edge_index: torch.LongTensor of shape (2, num_edges), the COO edge index representing the graph.
    - embeddings: torch.Tensor of shape (num_nodes, embedding_dim), the node embeddings.
    - k: int, the number of nearest neighbors to consider in the embedding space.

    Returns:
    - score: float, the neighborhood preservation score (ratio of preserved neighbors in the embedding space).
    """
    num_nodes = embeddings.size(0)
    
    # Step 1: Build the adjacency list for the graph
    adj_list = {i: set() for i in range(num_nodes)}
    for i, j in zip(edge_index[0], edge_index[1]):
        adj_list[i.item()].add(j.item())
        adj_list[j.item()].add(i.item())  # Assuming an undirected graph

    # Step 2: Compute pairwise distances in the embedding space
    distances = torch.nn.functional.pdist(embeddings, p=2).pow(2)  # squared Euclidean distance
    distance_matrix = torch.zeros((num_nodes, num_nodes), device=embeddings.device)
    
    idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
    distance_matrix[idx[0], idx[1]] = distances
    distance_matrix = distance_matrix + distance_matrix.T

    # Step 3: Find k-nearest neighbors in the embedding space
    _, knn_indices = torch.topk(-distance_matrix, k=k, dim=1)

    # Step 4: Compute the neighborhood preservation score
    preservation_counts = 0
    total_neighbors = 0
    
    for node in range(num_nodes):
        graph_neighbors = adj_list[node]
        knn_neighbors = set(knn_indices[node].tolist())

        # Count how many graph neighbors are preserved in the k-nearest neighbors
        preserved_neighbors = len(graph_neighbors.intersection(knn_neighbors))
        
        preservation_counts += preserved_neighbors
        total_neighbors += len(graph_neighbors)

    # Avoid division by zero
    if total_neighbors == 0:
        return 0.0
    
    # Compute the ratio of preserved neighbors
    score = preservation_counts / total_neighbors
    return score