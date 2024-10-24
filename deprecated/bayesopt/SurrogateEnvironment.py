'''
trains an Heterogenous GNN to predict reward 
'''

import torch
from gsnn.bayesopt.HGNN import HGNN
import numpy as np 
from gsnn.optim.EarlyStopper import EarlyStopper
from sklearn.metrics import r2_score
import gc
from torch import nn 
from torch.nn import functional as F

class QDWithMSELoss(nn.Module):
    def __init__(self, alpha=0.1, lambda_param=1.0, lambda_mse=1.0, lambda_order=1.0):
        super(QDWithMSELoss, self).__init__()
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.lambda_mse = lambda_mse
        self.lambda_order = lambda_order

    def forward(self, lower, mean, upper, target):
        """
        Args:
            lower (Tensor): Predicted lower bounds (L_i), shape [batch_size]
            mean (Tensor): Predicted means (μ_i), shape [batch_size]
            upper (Tensor): Predicted upper bounds (U_i), shape [batch_size]
            target (Tensor): True target values (y_i), shape [batch_size]
        Returns:
            Tensor: Scalar loss value
        """
        n = target.size(0)  # Number of samples in the batch

        # Enforce ordering of intervals
        lower_ordered = torch.min(lower, upper)
        upper_ordered = torch.max(lower, upper)
        lower = lower_ordered
        upper = upper_ordered

        # Compute captured samples mask
        captured = ((target >= lower) & (target <= upper)).float()  # 1 if captured, 0 otherwise

        # Compute MPIW_capt: Mean Prediction Interval Width for captured samples
        interval_width = upper - lower  # Shape: [batch_size]
        interval_width_capt = interval_width * captured  # Zero width for non-captured samples
        sum_interval_width_capt = interval_width_capt.sum()
        num_captured = captured.sum()
        # Avoid division by zero
        mpiw_capt = sum_interval_width_capt / (num_captured + 1e-8)

        # Compute PICP: Prediction Interval Coverage Probability
        picp = num_captured / n

        # Compute the penalty term
        penalty = torch.max(torch.tensor(0.0, device=target.device), (1 - self.alpha) - picp)
        penalty = penalty ** 2
        penalty = penalty * n / (self.alpha * (1 - self.alpha))

        # QD Loss
        loss_qd = mpiw_capt + self.lambda_param * penalty

        # MSE Loss on the mean prediction
        mse_loss = F.mse_loss(mean, target)

        # Interval Ordering Loss
        ordering_loss = torch.relu(lower - mean).mean() + torch.relu(mean - upper).mean()

        # Total Loss
        total_loss = loss_qd + self.lambda_mse * mse_loss + self.lambda_order * ordering_loss

        return total_loss

def loss_function(predicted_LCB, predicted_mean, predicted_UCB, target, lambda_ordering=1.0):
    # Mean Squared Error for the mean
    loss_mean = torch.nn.functional.mse_loss(predicted_mean.view(-1,1), target.view(-1,1))
    
    # Quantile losses for LCB and UCB
    q_lcb = 0.025  # Lower quantile for 95% confidence interval
    errors_LCB = target.view(-1,1) - predicted_LCB.view(-1,1)
    loss_LCB = torch.max((q_lcb - 1) * errors_LCB, q_lcb * errors_LCB).mean()

    q_ucb = 0.975  # Upper quantile for 95% confidence interval
    errors_UCB = target.view(-1,1) - predicted_UCB.view(-1,1)
    loss_UCB = torch.max((q_ucb - 1) * errors_UCB, q_ucb * errors_UCB).mean()

    # Penalties for ordering
    penalty_LCB_mean = torch.relu(predicted_LCB - predicted_mean).mean()
    penalty_mean_UCB = torch.relu(predicted_mean - predicted_UCB).mean()
    penalty_ordering = penalty_LCB_mean + penalty_mean_UCB

    # Total loss
    total_loss = loss_mean + loss_LCB + loss_UCB + lambda_ordering * penalty_ordering
    return total_loss

class SurrogateEnvironment: 
    ''''''

    def __init__(self, edge_index_dict, x_dict): 

        self.edge_index_dict = edge_index_dict
        self.x_dict = x_dict

    def optimize(self, recorder): 

        gc.collect() 
        torch.cuda.empty_cache()

        if torch.cuda.is_available(): 
            device = 'cuda'
        else: 
            device = 'cpu'

        model = HGNN(num_nodes=self.x_dict['function'].size(0)).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        #crit = torch.nn.GaussianNLLLoss() 
        crit = QDWithMSELoss()

        _n = int(len(recorder)*0.9)
        train_idx = torch.randperm(len(recorder))[:_n]
        val_idx = torch.randperm(len(recorder))[_n:]
 
        best_r = -np.inf
        best_state = model.state_dict()

        early_stopper = EarlyStopper(patience=20, min_delta=0)

        #train_batch = recorder.batch(idxs=train_idx, edge_index_dict=self.edge_index_dict, x_dict=self.x_dict).to(device)
        val_batch = recorder.batch(idxs=val_idx, edge_index_dict=self.edge_index_dict, x_dict=self.x_dict).to(device)
        # TODO: batch training 
        for epoch in range(1000): 
            
            model.train()
            for _idxs in torch.split(train_idx, split_size_or_sections=50): 
                batch = recorder.batch(idxs=_idxs, edge_index_dict=self.edge_index_dict, x_dict=self.x_dict).to(device)
                optim.zero_grad()
                lcb, mu, ucb = model(batch, mask=batch.action)
                loss = crit(lcb, mu, ucb, target=batch.reward.squeeze())
                #loss = loss_function(lcb, mu, ucb, target=batch.reward)
                #loss = crit(rhat_mean, train_batch.reward, rhat_var)
                loss.backward() 
                optim.step()

            # eval
            with torch.no_grad(): 
                model.eval()
                
                lcb, mu, ucb = model(val_batch, mask=val_batch.action)
                #val_loss = crit(mu, val_batch.reward, rhat_var).mean().item()
                r = np.corrcoef(val_batch.reward.detach().cpu().numpy().ravel(), mu.detach().cpu().numpy().ravel())[0,1]

                if r > best_r: 
                    best_r = r
                    best_state = model.state_dict()

            if early_stopper.early_stop(-r): 
                print()
                print(f'stopping training early | epoch={epoch} | best pearson loss: {best_r:.4f}')
                break
            else: 
                print(f'epoch: {epoch}: train loss: {loss.item():.4f} | val pearson: {r:.4f} | best pearson loss: {best_r:.4f}', end='\r')


        model.load_state_dict(best_state)
        for p in model.parameters(): 
            p.requires_grad = False

        # eval r2 
        _, val_rhat, _ = model(val_batch, mask=val_batch.action)
        r2 = r2_score(val_batch.reward.detach().cpu().numpy().ravel(), val_rhat.detach().cpu().numpy().ravel())
        print(f'validation R^2: {r2:.4f}')

        self.model = model
            

    def get_candidate_policy(self, recorder): 

        if torch.cuda.is_available(): 
            device = 'cuda'
        else: 
            device = 'cpu'

        # init candidate - use softmax gumbel 
        n_ = self.x_dict['function'].size(0)
        policy = torch.nn.Parameter(torch.randn((n_, 2), dtype=torch.float32, requires_grad=True, device=device))
        optim = torch.optim.Adam([policy], lr=1e-2)
        early_stopper = EarlyStopper(patience=50, min_delta=0)

        batch = recorder.batch(idxs=[0], edge_index_dict=self.edge_index_dict, x_dict=self.x_dict).to(device)

        for i in range(5000): 
            optim.zero_grad()
            a = torch.nn.functional.gumbel_softmax(policy, tau=1, hard=True, dim=-1)
            lcb, mu, ucb = self.model(batch, a[:, 0])
            loss = -ucb # TODO: use UCB/EI
            loss.backward()
            optim.step()

            if early_stopper.early_stop(loss): 
                print(f'candidate policy chosen...iter: {i}')
                break
            else: 
                print(f'iter: {i} | loss: {loss.item():.4f}', end='\r')

        print(f'candidate policy... predicted mean reward: {mu.item():.3f}, UCB: {-loss.item():.3f}')

        return torch.softmax(policy, dim=-1)[:, 0]

