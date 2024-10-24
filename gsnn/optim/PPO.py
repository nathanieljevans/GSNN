'''
This is an implementation of proximal policy optimization designed for use with single-state and zero-trajectory environments. 
'''


import numpy as np
import torch
from torch.distributions.bernoulli import Bernoulli

def edw(N, alpha):
        # Generate exponential decay weights
        weights = np.array([(1-alpha) ** i for i in range(N)])
        
        # Normalize weights to sum to 1
        weights /= np.sum(weights)
        
        return weights[::-1]

def clip_loss(old_policy, new_policy, actions, advantages, clip_param):
        """
        Compute the clipped PPO loss.

        Args: 
            actions         tensor          (ppo_batch, n_actions)
            advantages      tensor          (ppo_batch, 1)
        """

        old_log_prob = old_policy.log_prob(actions)#.sum(dim=1, keepdim=True).detach()   # (ppo_batch, n_actions)
        new_log_prob = new_policy.log_prob(actions)#.sum(dim=1, keepdim=True)            # (ppo_batch, n_actions)

        # Calculate probability ratios
        prob_ratio = torch.exp(new_log_prob - old_log_prob) # (ppo_batch, n_actions) 

        unclipped_obj = prob_ratio * advantages # (ppo_batch, n_actions) 

        # Calculate the clipped objective
        clipped_obj = torch.clamp(prob_ratio, 1 - clip_param, 1 + clip_param) * advantages

        # Compute the final loss as the minimum of the unclipped and clipped objectives
        clip_loss = -torch.minimum(unclipped_obj, clipped_obj).mean()

        return clip_loss


class PPO(): 
    def __init__(self, args, clip=10, eps=1e-3, warmup=3, verbose=True): 

        self.args = args 
        self.entropy = args.init_entropy
        self.clip = clip 
        self.eps = eps 
        self.warmup = warmup
        self.rewards = []

        self.iteration = 0
        self.verbose = verbose

    def update(self, rewards):
        
        #if np.isinf(reward) or np.isnan(reward): reward = -1
        self.rewards.append(rewards)

        self.iteration += 1

        # Check if it's time to decay the entropy
        if self.iteration % self.args.entropy_schedule == 0:
            if self.verbose: 
                print()
                print(f'decaying entropy: {self.entropy:5f}->{max(self.entropy * self.args.entropy_decay, self.args.min_entropy):.5f}')
            self.entropy = float(max(self.entropy * self.args.entropy_decay, self.args.min_entropy))

    def get_reward_params(self):
        if len(self.rewards) <= self.warmup:
            return 0,1
        else: 
            rewards_ = np.stack(self.rewards, axis=0) # (n_rewards, n_outputs)
            weights = edw(rewards_.shape[0], alpha=self.args.alpha)
            mus = [] ; stds = [] 
            for i in range(len(self.rewards[0])): 
                r = rewards_[:, i]
                mu = np.average(r, weights=weights)
                squared_diffs = (r - mu) ** 2
                std = np.sqrt(np.sum(weights * squared_diffs) / np.sum(weights))
                mus.append(mu) ; stds.append(std)
            return np.array(mus), np.array(stds)
       
    def scale(self, rewards):
        # rewards shape (n_outputs)
        mu, std = self.get_reward_params()
        return (rewards - mu) / (std + self.eps)

    def train_actor(self, logits, actions, rewards, x, actor, optim):
        '''
        
        Args: 
            logits       tensor         policy logits 
            actions      list<array>    batch of actions 
            rewards      list<array>    batch of unnormalized rewards
        '''

        actions    = torch.stack(actions, dim=0)

        # should we really be normalizing within batch? this means that there will always be a positive obs, even if it's less than the baseline
        #advantages = [self.scale(r) for r in rewards]  # normalize by baseline using ema ; size (ppo_batch, num_rewards)
        advantages = np.array(rewards) # normalize by baseline using ema ; size (ppo_batch, num_rewards)
        advantages = (advantages - np.mean(advantages, 0, keepdims=True)) / (np.std(advantages, 0, keepdims=True) + 1e-6) # normalize within batch ; shape: (ppo_batch, num_rewards)
        advantages = np.clip(advantages, -self.clip, self.clip).mean(-1) # shape: (ppo_batch,)
        advantages = torch.tensor(advantages, dtype=torch.float32).view(-1,1)

        old_policy = Bernoulli(logits=logits.view(1,-1).detach())

        for ii in range(self.args.ppo_iters):

            optim.zero_grad()
            new_logits = actor(x).view(1,-1)
            new_policy = Bernoulli(logits=new_logits)

            # early stopping reccommended by openai spinning up: https://spinningup.openai.com/en/latest/algorithms/ppo.html 
            if torch.distributions.kl.kl_divergence(new_policy, old_policy).mean() > self.args.target_kl: # should maybe be a sum 
                #print('reached kl target', ii)
                break
            
            loss = clip_loss(old_policy, new_policy, actions.squeeze(), advantages, clip_param=self.args.clip_param) \
                        - self.entropy*(new_policy.entropy()).mean()

            loss.backward()
            optim.step()

        return actor 