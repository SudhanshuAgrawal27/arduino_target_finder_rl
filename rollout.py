"""Rollout collection: run episodes serially under the current policy and
flatten them into one PPO training batch."""

from contextlib import nullcontext

import torch
from tqdm import tqdm

from network import obs_to_tensor
from ppo import compute_gae
from simulator import ACTIONS, GridEnvironment, run_simulation, temporary_seed


def collect_rollouts(env_kwargs, model, n_episodes, gamma, lam, episode_seeds=None, desc="rollout"):
    """Run n_episodes full episodes serially (one GridEnvironment at a
    time, engine="mlp_network") under `model`, and turn them into a flat
    training batch.

    `env_kwargs` configures each GridEnvironment (grid_size, subgrid_size,
    score_radius, max_steps, history_length); a fresh instance is
    constructed per episode.

    `episode_seeds=None` (the default, used for actual training rollouts):
    episode-to-episode variety comes from the global random stream (see
    simulator.set_global_seed) continuing to advance across the whole call.
    Pass a list of `n_episodes` seeds (e.g. from simulator.derive_episode_seeds)
    to instead wrap each episode in `temporary_seed(episode_seeds[i])` --
    fixing the exact set of problem instances played and leaving the
    ambient global stream undisturbed by the call, e.g. for a
    validation-loss batch that must reuse eval's episode set.

    `model` must expose `.act`, `.get_value` (network.ActorCritic does) --
    pass the unwrapped model if it's been through accelerator.prepare().

    Returns (batch, stats): batch is a dict of stacked tensors
    (observations, actions, old_log_probs, advantages, returns), all still
    on CPU; stats holds per-episode lists (episode_returns, episode_lengths,
    successes) for logging.
    """
    obs_list, action_list, log_prob_list, advantage_list, return_list = [], [], [], [], []
    episode_returns, episode_lengths, successes = [], [], []

    for i in tqdm(range(n_episodes), desc=desc, leave=False):
        seed_ctx = temporary_seed(episode_seeds[i]) if episode_seeds is not None else nullcontext()
        with seed_ctx:
            env = GridEnvironment(**env_kwargs)
            trajectory = run_simulation(env=env, engine="mlp_network", network=model)

        n_steps = len(trajectory) - 1
        rewards = [trajectory[t + 1]["reward"] for t in range(n_steps)]
        values = [trajectory[t + 1]["value"] for t in range(n_steps)]

        # terminated: no future, bootstrap with 0. truncated: the episode
        # was cut off, not concluded, so bootstrap with the critic's own
        # estimate of the final (never-acted-on) state.
        bootstrap_value = 0.0 if env.terminated else model.get_value(trajectory[-1]["observation"])
        advantages, returns = compute_gae(rewards, values, bootstrap_value, gamma, lam)

        for t in range(n_steps):
            obs_list.append(obs_to_tensor(trajectory[t]["observation"], subgrid_size=env.subgrid_size))
            action_list.append(ACTIONS.index(trajectory[t + 1]["action"]))
            log_prob_list.append(trajectory[t + 1]["log_prob"])
        advantage_list.extend(advantages)
        return_list.extend(returns)

        episode_returns.append(sum(rewards))
        episode_lengths.append(n_steps)
        successes.append(env.terminated)

    batch = {
        "observations": torch.stack(obs_list),
        "actions": torch.tensor(action_list, dtype=torch.long),
        "old_log_probs": torch.tensor(log_prob_list, dtype=torch.float32),
        "advantages": torch.tensor(advantage_list, dtype=torch.float32),
        "returns": torch.tensor(return_list, dtype=torch.float32),
    }
    stats = {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "successes": successes,
    }
    return batch, stats
