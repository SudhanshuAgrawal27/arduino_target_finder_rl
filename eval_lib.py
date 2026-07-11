"""Evaluation: run a fixed, seeded set of episodes under a given policy
engine and report summary stats. Used for train.py's per-epoch eval, its
random-policy baseline, and the standalone eval_random.py / eval_fixed_dataset.py
scripts."""

from collections import defaultdict

from tqdm import tqdm

from simulator import GridEnvironment, derive_episode_seeds, run_simulation, temporary_seed


class DeterministicPolicy:
    """Adapts ActorCritic.act_deterministic into the `.act(observation) ->
    (action_index, log_prob, value, entropy)` shape run_simulation's
    mlp_network engine expects. log_prob/value/entropy are unused so just
    None."""

    def __init__(self, model):
        self.model = model

    def act(self, observation):
        return self.model.act_deterministic(observation), None, None, None


def run_eval(env_kwargs, n_episodes, seed, engine="mlp_network", model=None):
    """Run n_episodes under `engine` and return summary stats.

    Each episode is deterministically derived from `seed` via
    temporary_seed(episode_seeds[i]), so calling this again with the same
    (seed, n_episodes) reproduces identical episodes -- and a different
    `engine` with the same (seed, n_episodes) plays the same problem
    instances, making eval and a random-policy baseline directly comparable.

    engine="mlp_network" requires `model` (pass the unwrapped model if it's
    been through accelerator.prepare()) and acts greedily via
    DeterministicPolicy. engine="random" needs no model.
    """
    network = DeterministicPolicy(model) if engine == "mlp_network" else None

    episode_returns, episode_lengths, successes = [], [], []
    for episode_seed in tqdm(derive_episode_seeds(seed, n_episodes), desc=f"eval:{engine}", leave=False):
        with temporary_seed(episode_seed):
            env = GridEnvironment(**env_kwargs)
            trajectory = run_simulation(env=env, engine=engine, network=network)
        rewards = [step["reward"] for step in trajectory[1:]]
        episode_returns.append(sum(rewards))
        episode_lengths.append(len(rewards))
        successes.append(env.terminated)

    n = len(episode_returns)
    return {
        "avg_return": sum(episode_returns) / n,
        "success_rate": sum(successes) / n,
        "avg_length": sum(episode_lengths) / n,
    }


def run_eval_fixed(env_kwargs, dataset, engine="mlp_network", model=None):
    """Like run_eval, but replays a fixed, pre-generated, categorized set of
    problem instances (see build_eval_fixed_dataset.py / eval_fixed_dataset.json)
    instead of episodes derived from a seed range. Returns overall stats plus
    a per-category breakdown.

    `dataset` is the parsed eval_fixed_dataset.json dict. The seed -> instance
    mapping depends only on grid_size/subgrid_size/score_radius, so those
    three are asserted to match what the dataset was generated with.

    Each summary dict reports avg_return and success_rate over all n episodes
    in the bucket, but avg_length only over the n_solved episodes that reached
    the target (None if none did) -- so avg_length measures path efficiency on
    solved instances, not a mix of that and the max_steps cutoff of failures.
    """
    for key in ("grid_size", "subgrid_size", "score_radius"):
        if dataset[key] != env_kwargs[key]:
            raise ValueError(
                f"env_kwargs[{key!r}]={env_kwargs[key]!r} does not match the "
                f"fixed dataset's {key}={dataset[key]!r} -- the recorded seeds "
                f"would reproduce different problem instances than the ones "
                f"the dataset was categorized for."
            )

    network = DeterministicPolicy(model) if engine == "mlp_network" else None

    overall = {"returns": [], "lengths": [], "successes": []}
    per_category = defaultdict(lambda: {"returns": [], "lengths": [], "successes": []})

    for entry in tqdm(dataset["entries"], desc=f"eval_fixed:{engine}", leave=False):
        with temporary_seed(entry["seed"]):
            env = GridEnvironment(**env_kwargs)
        trajectory = run_simulation(env=env, engine=engine, network=network)
        rewards = [step["reward"] for step in trajectory[1:]]

        category = f"{entry['distance_tier']}/{entry['target_locality']}"
        for bucket in (overall, per_category[category]):
            bucket["returns"].append(sum(rewards))
            bucket["lengths"].append(len(rewards))
            bucket["successes"].append(env.terminated)

    return {
        "overall": summarize_bucket(overall),
        "by_category": {category: summarize_bucket(bucket) for category, bucket in sorted(per_category.items())},
    }


def summarize_bucket(bucket):
    """Summary stats for one bucket of parallel per-episode lists (returns,
    lengths, successes). avg_return and success_rate are over all n episodes,
    but avg_length is over the n_solved that reached the target (None if none
    did) -- so it measures path efficiency on solved instances rather than a
    mix of that and the max_steps cutoff a failed episode always hits."""
    n = len(bucket["returns"])
    solved_lengths = [length for length, ok in zip(bucket["lengths"], bucket["successes"]) if ok]
    n_solved = len(solved_lengths)
    return {
        "avg_return": sum(bucket["returns"]) / n,
        "success_rate": sum(bucket["successes"]) / n,
        "avg_length": (sum(solved_lengths) / n_solved) if n_solved else None,
        "n": n,
        "n_solved": n_solved,
    }
