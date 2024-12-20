
import numpy as np
import torch
from torch.distributions.bernoulli import Bernoulli
from sklearn.metrics import roc_auc_score

class REINFORCE(torch.nn.Module): 
    def __init__(self, env, n_actions, action_labels=None, clip=10, eps=1e-5, warmup=3, verbose=True, 
                        entropy=0., entropy_decay=0.99, min_entropy=0.01, window=10,
                        init_prob=0.9, lr=1e-2, policy_decay=0.): 
        super().__init__()

        self.action_labels = action_labels
        self.env = env
        self.n_actions = n_actions
        self.entropy = entropy
        self.entropy_decay = entropy_decay
        self.min_entropy = min_entropy
        self.clip = clip 
        self.eps = eps 
        self.warmup = warmup
        self.rewards = []
        self.iteration = 0
        self.verbose = verbose
        self.window=window
        self.policy_decay = policy_decay

        # need to convert init prob to logit value 
        init_logit = np.log(init_prob / (1 - init_prob))
        self.logits = torch.nn.Parameter(torch.ones((1, self.n_actions), dtype=torch.float32) * init_logit)

        self.optim = torch.optim.Adam([self.logits], lr=lr)

        self.best_reward = None
        self.best_action = None

        self.iteration = 0

    def sample(self):

        policy = Bernoulli(logits=self.logits)
        action = policy.sample()
        return action

    def update(self, rewards):
        
        self.rewards.append(rewards)
        self.iteration += 1

        self.entropy = float(max(self.entropy * self.entropy_decay, self.min_entropy))
        if self.verbose: print(f'entropy value -> {self.entropy:.3f}', end='\r')

    def get_reward_params(self):
        if len(self.rewards) < self.warmup:
            return 0,1
        else: 
            rewards_ = np.stack(self.rewards[-self.window:], axis=0)
            return rewards_.mean(0), rewards_.std(0)
       
    def scale(self, rewards):
        # rewards shape (n_outputs)
        mu, std = self.get_reward_params()
        rewards = (rewards - mu) / (std + self.eps)
        rewards = np.clip(rewards, -self.clip, self.clip)
        return rewards
    
    def prob_of(self, action): 
        policy = Bernoulli(logits=self.logits)
        return torch.exp(policy.log_prob(action).sum())
    
    def get_edge_probs(self):
        
        return torch.sigmoid(self.logits).detach().numpy()
    
    def print_progress_(self): 

        if self.action_labels is not None: 
            edge_probs = self.logits.squeeze().sigmoid().detach().cpu().numpy()
            true_action = self.action_labels
            auroc = roc_auc_score(true_action, edge_probs)
            acc = ((edge_probs > 0.5) == true_action).mean()
            prob_true = self.prob_of(torch.from_numpy(true_action))
            print(f'\t\t\t --> iter: {self.iteration} || auroc {auroc:0.3f} || acc: {acc:.3f} || prob(true_action): {prob_true:.3f} || last reward: {self.rewards[-1].mean():.3f}')
        else: 
            print(f'\t\t\t --> iter: {self.iteration} || last reward: {self.rewards[-1].mean():.3f}')

    def step(self):
        '''
        
        Args: 
            logits       tensor         policy logits (n_actions,)
            actions      list<array>    batch of actions (n_actions,)
            rewards      list<array>    batch of unnormalized rewards (n_outputs,)
        '''
        self.optim.zero_grad()
        policy = Bernoulli(logits=self.logits)
        action = policy.sample()
        rewards = self.env.run(action)

        advantages = self.scale(rewards).mean()

        if len(self.rewards) >= self.warmup: 
            loss = -(policy.log_prob(action) * advantages).sum() + self.entropy * policy.entropy().sum() + self.policy_decay*self.logits.sigmoid().mean()
            loss.backward()
            self.optim.step()
            
        self.update(rewards)

        # log best reward 
        if (self.best_reward is None) or (rewards.mean() > self.best_reward.mean()): 
            self.best_reward = rewards
            self.best_action = action

        self.iteration += 1

        self.print_progress_()



