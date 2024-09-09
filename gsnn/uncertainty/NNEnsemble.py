


import torch 
from torch.func import stack_module_state
from torch.func import functional_call
import copy
from torch import vmap
import torch.distributions as dist

def create_mixture(distributions, device, weights=None):
    # Number of distributions
    N = len(distributions)
    #B = distributions[0].mean.size(0)
    B = 1
    
    # If no specific mixture weights are given, assume uniform weights
    if weights is None:
        weights = torch.ones((B,N), device=device)

    # Create a categorical distribution for the mixture weights
    categorical = dist.Categorical(weights)

    # Check the type of the first distribution to determine the parameter stacking
    if isinstance(distributions[0], dist.Normal):
        means = torch.stack([d.mean for d in distributions], dim=-1)
        stddevs = torch.stack([d.stddev for d in distributions], dim=-1)
        component_distribution = dist.Normal(means, stddevs)
    elif isinstance(distributions[0], dist.Beta):
        concentration1s = torch.stack([d.concentration1 for d in distributions], dim=-1)
        concentration0s = torch.stack([d.concentration0 for d in distributions], dim=-1)
        component_distribution = dist.Beta(concentration1s, concentration0s)
    else:
        raise ValueError("Unsupported distribution type")

    # Create the mixture model
    mixture = dist.MixtureSameFamily(categorical, component_distribution)
    return mixture


class NNEnsemble(torch.nn.Module): 

    def __init__(self, models, use_mixture=True): 
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.use_mixture = use_mixture
        # TODO: https://pytorch.org/tutorials/intermediate/ensembling.html

    def forward(self, x):

        pred_dists = [model(x) for model in self.models] # [N, (B)]

        # ensembling as defined in: https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf
        # optional improvement: https://pytorch.org/docs/stable/distributions.html#mixturesamefamily

        pred_dist = create_mixture(distributions=pred_dists, device=x.device)

        if not self.use_mixture: 
            # use mean and variance of mixture to estimate a single distribution 
            if self.models[0].dist == 'gaussian': 
                mu = pred_dist.mean 
                std = (pred_dist.variance)**(0.5)
                pred_dist = torch.distributions.Normal(mu, std)
            else: 
                raise NotImplementedError

        return pred_dist
