"""PPO math: GAE advantage estimation and the clipped surrogate loss."""

import torch


def compute_gae(rewards, values, bootstrap_value, gamma, lam):
    """GAE advantages and returns for one episode's T real steps.

    `values[t]` is V(s_t), the critic's estimate at the state action t was
    chosen from. `bootstrap_value` is V(s_T) for the state after the last
    action: 0 if the episode terminated (reached the target -- no future to
    bootstrap), otherwise the critic's estimate of the truncated episode's
    final state (see network.ActorCritic.get_value).

    Returns (advantages, returns), each a length-T list.
    """
    T = len(rewards)
    values_ext = values + [bootstrap_value]
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [advantages[t] + values[t] for t in range(T)]
    return advantages, returns


def ppo_loss(model, obs_batch, action_batch, old_log_prob_batch, advantage_batch, return_batch,
             clip_eps, value_coef, entropy_coef):
    """The clipped surrogate PPO objective plus value and entropy terms.

    `model.evaluate_actions` recomputes log-probs/values/entropy under the
    *current* parameters (with gradients); `old_log_prob_batch` is what the
    acting policy produced at rollout-collection time, so the ratio
    r_t(theta) = exp(new_log_prob - old_log_prob) measures how far the
    policy has moved since collecting this batch.
    """
    new_log_prob, value, entropy = model.evaluate_actions(obs_batch, action_batch)
    ratio = (new_log_prob - old_log_prob_batch).exp()

    surr1 = ratio * advantage_batch
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage_batch
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = (value - return_batch).pow(2).mean()
    entropy_bonus = entropy.mean()

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

    stats = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy_bonus.item(),
        "approx_kl": (old_log_prob_batch - new_log_prob).mean().item(),
    }
    return loss, stats
