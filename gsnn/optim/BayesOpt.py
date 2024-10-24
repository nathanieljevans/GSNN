
import torch 
import copy
from gsnn.optim.Environment import Environment
from gsnn.optim.SurrogateEnvironment import SurrogateEnvironment
from gsnn.optim.ExperienceRecord import ExperienceRecord
import itertools
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import numpy as np
import pandas as pd

def get_candidates(num_actions, dtype=torch.float32):
    # Generate all possible binary combinations using itertools.product
    # NOTE: this is not tractable for large action spaces 
    binary_combinations = list(itertools.product([0, 1], repeat=num_actions))
    candidate_actions = torch.tensor(binary_combinations, dtype=dtype)
    
    return candidate_actions

def EI(out, d):
    # NOTE: not differntiable
    # out (samples, batch, outputs) 
    # d best reward
    mu_x = out.mean(dim=(0, 2), keepdims=True).cpu()  
    sigma_x = out.std(dim=(0, 2), keepdims=True).cpu() + 1e-8
    Z = (mu_x - d) / sigma_x
    standard_normal = Normal(0, 1)
    Phi_Z = standard_normal.cdf(Z)
    phi_Z = torch.exp(standard_normal.log_prob(Z))
    obj = (mu_x - d) * Phi_Z + sigma_x * phi_Z
    return obj.mean(dim=(0,2))

def avg(out): 
    return out.mean(dim=[0,2])

def ucb(out, q=0.95): 
    return torch.quantile(out, q=torch.tensor([q]), dim=0).mean(-1).squeeze()

def lcb(out, q=0.05): 
    return torch.quantile(out, q=torch.tensor([q]), dim=0).mean(-1).squeeze()

def pi(out, d): 
    # NOTE: not differentiable
    d = d.mean(-1) # avg across outputs
    out = out.mean(dim=2) # avg across dimensions
    obj = (1.*(out > d)).mean(0) # avg across samples
    return obj


def aggregate_duplicate_actions(actions, rewards):
    """
    Aggregates duplicate actions by averaging their corresponding rewards.

    Parameters:
        actions (torch.Tensor): Tensor of shape (n_samples, n_actions) containing action vectors.
        rewards (torch.Tensor): Tensor of shape (n_samples,) containing rewards for each action.

    Returns:
        torch.Tensor: Unique actions.
        torch.Tensor: Averaged rewards for each unique action.
    """
    actions = actions.type(torch.float32)
    rewards = rewards.type(torch.float32)

    # Step 1: Get unique actions and their indices
    unique_actions, inverse_indices = torch.unique(actions, dim=0, return_inverse=True)

    # Step 2: Initialize tensors to hold sum of rewards and count of each unique action
    reward_sum = torch.zeros(unique_actions.size(0), dtype=torch.float32, device=actions.device)
    count = torch.zeros(unique_actions.size(0), dtype=torch.float32, device=actions.device)

    # Step 3: Use scatter_add to accumulate the rewards and counts for unique actions
    reward_sum = reward_sum.scatter_add(0, inverse_indices, rewards)
    count = count.scatter_add(0, inverse_indices, torch.ones_like(rewards, dtype=torch.float32))

    # Step 4: Calculate the average reward for each unique action
    avg_rewards = reward_sum / count

    # This might get inefficient if we have a large action space
    #action_names = [str(i) for i in range(self.nactions)]
    #df = pd.DataFrame(actions.detach().cpu().numpy() == 1., columns=action_names)
    #df = df.assign(reward = objs.detach().cpu().numpy())
    #df = df.groupby(action_names).mean().reset_index()
    #actions = torch.tensor(df[action_names].values, dtype=torch.float32)
    #objs = torch.tensor(df.reward.values, dtype=torch.float32)

    return unique_actions, avg_rewards


class BayesOpt: 

    def __init__(self, args, data, train_dataset, test_dataset): 
        
        self.args = args 
        self.data = data

        self.model_kwargs = {'edge_index_dict'                : data.edge_index_dict, 
                            'node_names_dict'                 : data.node_names_dict,
                            'channels'                        : args.channels, 
                            'layers'                          : args.layers, 
                            'dropout'                         : args.dropout,
                            'share_layers'                    : args.share_layers,
                            'add_function_self_edges'         : args.add_function_self_edges,
                            'norm'                            : args.norm}

        self.training_kwargs = {'lr'            : args.lr, 
                                'max_epochs'    : args.max_epochs, 
                                'patience'      : args.patience,
                                'min_delta'     : args.min_delta,
                                'batch'         : args.batch,
                                'workers'       : args.workers}

        self.env =  Environment(train_dataset, 
                                test_dataset, 
                                None, 
                                copy.deepcopy(self.model_kwargs), 
                                self.training_kwargs, 
                                metric=args.metric)
        
        assert not args.surr_bias, 'including `surr_bias=True` in the will negate much of the implicit modeling capacity of the surrogate GSNN model'
        
        self.vmodel_kwargs= {'channels'     :args.surr_channels,
                        'norm'              :args.surr_norm,
                        'layers'            :args.surr_layers,
                        'dropout'           :args.surr_dropout,
                        'bias'              :args.surr_bias,
                        'share_layers'      :args.surr_share_layers}
        
        self.vtrain_kwargs= {'batch_size'   :args.surr_batch,
                             'lr'           :args.surr_lr,
                             'wd'           :args.surr_wd, 
                             'epochs'       :args.surr_epochs}
        
        self.venv = SurrogateEnvironment(data                   = data, 
                                         model_kwargs           = self.vmodel_kwargs, 
                                         stochastic_channels    = args.stochastic_channels, 
                                         hnet_width             = args.hnet_width, 
                                         samples                = args.samples)
        
        self.record = ExperienceRecord(args.record_dir)
        self.nactions = data.edge_index_dict['function','to','function'].size(1)

    def collect(self, actions, verbose=True): 
        '''
        run candidate actions
        '''
        _rewards = []
        for i, action in enumerate(actions):
            if verbose: print(f'\t--> evaluating candidate actions... [progress: {i+1}/{len(actions)}]', end='\r')
            rewards = self.env.run(action, action_type='edge', reward_type=self.args.reward_agg, verbose=False)
            self.record.add(action, rewards)
            _rewards.append(rewards.mean().item())

        if verbose: print()
        return _rewards

    def warmup(self, N, verbose=True, p=0.95): 
        '''
        collect initial samples and optimize the surrogate environment
        '''
        #actions = sample_diverse_actions(naction=self.nactions, num_samples=N)
        if N > 0: 
            actions = 1.*(torch.rand(size=(N, self.nactions), dtype=torch.float32) > p)
            self.collect(actions, verbose=verbose)
        if verbose: print()
        self.venv.optimize(self.record, 
                           train_kwargs=self.vtrain_kwargs, 
                           verbose=verbose, 
                           patience=self.args.surr_patience,
                           min_delta=self.args.min_delta)

    def _sample_neighborhood(self, action, N=100, k=1): 
        '''sample actions from the `k`-neighborhood of `action`'''

        neighbors = [] # consider a neighbor as actions with a hamming distance less than or equal to k
        for n in range(N): 
            neighbor = action.clone() 
            idxs = torch.randint(0, self.nactions, size=(k,)) # select binary actions 
            hd = torch.randint(1, k+1, size=(1,)) # select number of bits to flip (hamming distance)
            idxs = idxs[:hd]
            neighbor[idxs] = (~(neighbor[idxs].type(torch.bool))).type(torch.float32)
            neighbors.append(neighbor)
        return torch.stack(neighbors, dim=0)
    
    def get_candidates(self, N, objective, obj_kwargs={}, k=1, batch=100, iters=100, samples=100, verbose=True, lr=1e-2, alpha=0.01): 
        '''use the REINFORCE via the surrogate function to learn a candidate that maximizes the objective function'''

        # Run this N times with random policies 
        best_reward_idx = np.stack([np.array(r) for r in self.record.rewards], axis=0).mean(axis=(-1)).argmax()
        best_reward = (self.record.rewards[best_reward_idx] - self.venv.rewards_mean.item()) / (self.venv.rewards_std.item() + 1e-8)
        best_action = self.record.actions[best_reward_idx]

        # soft constraints on the neighborhood size by calculating an `alpha` value such that
        # E[abs(policy - best_reward)] = k
        #alpha = k/(2*self.nactions)

        # start candidate near the best state
        probs = torch.clip(best_action, alpha, 1-alpha)
        logits = torch.log(probs / (1 - probs))
        policy = torch.nn.Parameter(logits, requires_grad=True)
        optim = torch.optim.Adam([policy], lr=lr)

        crit_map = {'EI':lambda x: EI(x, d=best_reward),
                    'UCB':lambda x: ucb(x, **obj_kwargs),
                    'LCB':lambda x: lcb(x, **obj_kwargs),
                    'PI':lambda x: pi(x, d=best_reward),
                    'mean':lambda x: avg(x, **obj_kwargs)}
        
        crit = crit_map[objective]

        actions = [] 
        objs = []
        for i in range(iters): 
            # REINFORCE to explore the action space for new candidates 
            optim.zero_grad() 
            m = Bernoulli(logits=policy)
            action = m.sample((batch,))
            reward = crit(self.venv.predict(action, samples=samples))
            reward_scaled = (reward - reward.mean())/(reward.std() + 1e-8)
            loss = (-m.log_prob(action)*reward_scaled.unsqueeze(-1)).mean()
            loss.backward()
            optim.step()
            if verbose: print(f'selecting candidates... {i}/{iters} || mean obj ({objective}): {reward.mean().item():.3f} || loss: {loss.item():.4f}', end='\r')
            actions.append(action.detach().cpu())
            objs.append(reward.detach().cpu())

        if verbose: print()

        actions = torch.cat(actions, dim=0)
        objs = torch.cat(objs, -1)

        # aggregate duplicate actions 
        actions, objs = aggregate_duplicate_actions(actions, objs) 

        # TODO: estimate prob_optima based on neighborhood points

        idx = torch.argsort(objs, descending=True)

        if N is not None:
            return actions[idx][:N].detach().cpu(), objs[idx][:N].detach().cpu()
        else: 
            return actions[idx].detach().cpu(), objs[idx].detach().cpu()

    def run_local_search(self, iters, objective, obj_kwargs, neighborhood_size): 
        '''
        local optimization: sample a local neighborhood around the best performing action and select top candidates based on acquisition function 
        Note: choice of local neighborhood size (e.g., hamming distance) become very important for tractable exploration in large search spaces. 
        '''

        best_reward_ = [] 
        for ii in range(iters): 

            best_reward = np.stack([np.array(r) for r in self.record.rewards], axis=0).mean(axis=(-1)).max()
            best_reward_.append(best_reward)
            
            top_candidates, top_objectives = self.get_candidates(N=self.args.bayesopt_batch_size, 
                                                                    objective=objective, 
                                                                    obj_kwargs=obj_kwargs,
                                                                    k=neighborhood_size, 
                                                                    batch=self.args.rl_batch, 
                                                                    iters=self.args.rl_iters, 
                                                                    samples=self.args.rl_samples,
                                                                    lr = self.args.rl_lr, 
                                                                    alpha = self.args.rl_alpha,
                                                                    verbose=True)

            # run env 
            _mean_rewards = self.collect(top_candidates, verbose=True)

            # retrain surrogate 
            self.record.check_and_load() # check to see if there are any new records (if using concurrently with other nodes)
            self.venv.optimize(self.record, 
                               self.vtrain_kwargs, 
                               verbose=True, 
                               ema_sampling=False,  # use this to focus training on newer samples 
                               patience=self.args.surr_patience, 
                               min_delta=self.args.min_delta) # update surrogate model while focusing on newer data

            print(f'[iter: {ii}] best reward: {best_reward.mean():.3f} || exploration improvement: {(max(_mean_rewards) - best_reward.mean()):.3f}')

        return best_reward_
        
    def run(self, iters=10, objective='ucb', obj_kwargs={}, neighborhood_size=2): 
        '''
        '''
        
        # TODO: record acuisition function, actual reward and evaluate performance of surrogate function. 
        # TODO: add obj_kwargs; choice of eps/q etc 
        return self.run_local_search(iters, objective, obj_kwargs, neighborhood_size)



'''
    def _sample_neighborhood(self, N=100, action, k=1): 

        neighbors = [] # consider a neighbor as actions with a hamming distance less than or equal to k
        for n in range(N): 
            neighbor = action.clone() 
            idxs = torch.randint(0, self.nactions, size=(k,)) # select binary actions 
            hd = torch.randint(1, k+1, size=(1,)) # select number of bits to flip (hamming distance)
            idxs = idxs[:hd]
            neighbor[idxs] = (~(neighbor[idxs].type(torch.bool))).type(torch.float32)
            neighbors.append(neighbor)
        return torch.stack(neighbors, dim=0)

        
        
            hamming_dist = (action != best_action).sum(dim=-1).float()
            l1_penalty = torch.abs(action - best_action).sum(dim=-1)
            penalty = torch.where(hamming_dist > k, l1_penalty, torch.zeros_like(l1_penalty)).mean()
            loss = loss + penalty

            # try to enforce exploration outside of current action
            explore_penalty = torch.abs(best_action - action).sum(dim=-1)
            penalty2 = torch.where(hamming_dist == 0, explore_penalty, torch.zeros_like(l1_penalty)).mean()
            loss = loss + penalty2

            


def sample_diverse_actions(naction, num_samples):
    selected_actions = []
    
    # Select the first action randomly
    first_action = (torch.rand(naction) > 0.5).float()
    selected_actions.append(first_action)
    
    # Select the second action as the bitwise flip of the first
    second_action = (~first_action.to(torch.bool)).float()
    selected_actions.append(second_action)
    
    for _ in range(num_samples - 2):
        selected_stack = torch.stack(selected_actions)
        candidates = (torch.rand((100, naction)) > 0.5).float()  # Sample 100 random candidates
        dists = torch.cdist(selected_stack, candidates, p=1)  # Compute Hamming distance (L1 norm)
        min_dists = dists.min(dim=0)[0]  # Find the minimum distance of each candidate to the selected actions
        idx = min_dists.argmax()  # Select the candidate with the maximum of these minimum distances
        selected_actions.append(candidates[idx])
    
    return torch.stack(selected_actions)
'''