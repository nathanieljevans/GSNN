
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