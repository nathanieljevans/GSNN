import numpy as np 

def edw(N, alpha):
    # Generate exponential decay weights
    weights = np.array([(1-alpha) ** i for i in range(N)])
    
    # Normalize weights to sum to 1
    weights /= np.sum(weights)
    
    return weights[::-1]

class RewardScaler:
    def __init__(self, clip=5, eps=1e-3, alpha=0.04, warmup=3):
        """
        Reward scaling with exponential moving average statistics.

        Scales rewards using a running mean and standard deviation computed with exponential decay weights.
        This helps stabilize training by normalizing the reward distribution.

        Args:
            clip (float, optional): Maximum absolute value for scaled rewards. Default: 5
            eps (float, optional): Small constant for numerical stability. Default: 1e-3
            alpha (float, optional): Decay rate for exponential moving average. Default: 0.04
            warmup (int, optional): Number of updates before using computed statistics. Default: 3

        Example:
            >>> scaler = RewardScaler(clip=5, alpha=0.1)
            >>> for i in range(100):
            ...     reward = get_reward()
            ...     scaler.update(reward)
            ...     scaled_reward = scaler.scale(reward)
        """

        self.rewards = []
        self.eps = eps
        self.clip = clip
        self.alpha = alpha
        self.warmup = warmup

    def update(self, reward):
        
        if np.isinf(reward) or np.isnan(reward): reward = -1
        self.rewards.append(reward)

    def get_params(self):
        
        if len(self.rewards) <= self.warmup:
            return 0, 1
        else: 
            weights = edw(len(self.rewards), alpha=self.alpha)
            mu = np.average(self.rewards, weights=weights)

            # Bias-corrected weighted variance
            squared_diffs = (self.rewards - mu) ** 2
            
            std = np.sqrt(np.sum(weights * squared_diffs) / np.sum(weights))

            return mu, std
       
    def scale(self, reward):

        mu, std = self.get_params()
        
        return np.clip((reward - mu) / (std + self.eps), -self.clip, self.clip)