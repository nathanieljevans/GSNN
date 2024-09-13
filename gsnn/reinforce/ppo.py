
import torch
from torch.distributions.bernoulli import Bernoulli

def clip_loss(old_policy, new_policy, action, reward_scaled, clip_param=0.2):
    """
    Compute the clipped PPO loss.

    Parameters:
    - old_logits: logits from the policy before the update (old policy).
    - new_logits: logits from the current policy (new policy).
    - actions: the actions taken (sampled).
    - reward_scaled: the advantage (scaled reward in your case).
    - clip_param: the clipping parameter (epsilon).

    Returns:
    - clip_loss: the loss with clipping applied.
    """

    old_log_prob = old_policy.log_prob(action)
    new_log_prob = new_policy.log_prob(action)

    # Calculate probability ratios
    prob_ratio = torch.exp(new_log_prob - old_log_prob)

    # Calculate the unclipped objective
    unclipped_obj = prob_ratio * reward_scaled

    # Calculate the clipped objective
    clipped_obj = torch.clamp(prob_ratio, 1 - clip_param, 1 + clip_param) * reward_scaled

    # Compute the final loss as the minimum of the unclipped and clipped objectives
    clip_loss = -torch.minimum(unclipped_obj, clipped_obj).mean()

    return clip_loss

def update_actor(args, actor, x, reward_scaled, old_logits, action, optim, iter, iters=4, clip_param=0.2, target_kl=0.01):
    """
    Train the actor using PPO with clipping.

    Parameters:
    - NS: The neural network representing the actor (node selector).
    - x: The input node features.
    - action: The actions sampled from the old policy.
    - reward_scaled: The scaled reward (used as the advantage).
    - old_logits: Logits from the old policy (before update).
    - optimizer: Optimizer for the actor.
    - ppo_epochs: Number of training steps (epochs) to update the actor per reward.
    - clip_param: Clipping parameter (epsilon) for PPO.

    Returns:
    - None
    """
    old_policy = Bernoulli(logits=old_logits.detach())

    entropy_decay = args.entropy/(args.outer_iters/2) # decay entropy linearly to zero in half the total iters 
    for ii in range(iters):
        optim.zero_grad()

        # 1. Recompute new logits from the current policy (after updates)
        new_logits = actor(x).squeeze()
        new_policy = Bernoulli(logits=new_logits)

        # early stopping reccommended by openai spinning up: https://spinningup.openai.com/en/latest/algorithms/ppo.html 
        if torch.distributions.kl.kl_divergence(new_policy, old_policy).mean() > target_kl: 
            #print('early stopping at ppo iter: ', ii)
            break

        entropy_coef = max((args.entropy - iter*entropy_decay), 0)

        # 2. Compute the clipped PPO loss
        loss = clip_loss(old_policy, new_policy, action, reward_scaled, clip_param=clip_param) \
                    + args.l1_penalty*new_logits.mean() \
                    - entropy_coef*new_policy.entropy().mean()

        # 3. Backpropagate the loss and update the policy
        loss.backward()
        optim.step()

    return actor 