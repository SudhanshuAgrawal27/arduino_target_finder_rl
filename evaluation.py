"""Evaluation: run a fixed, seeded set of episodes under a given policy
engine and report summary stats. Used for train.py's per-epoch eval, its
random-policy baseline, and the standalone eval.py script."""

from collections import defaultdict

from tqdm import tqdm

from simulator import GridEnvironment, derive_episode_seeds, run_simulation, temporary_seed


class DeterministicPolicy:
    """Adapts ActorCritic.act_deterministic into the `.act(observation) ->
    (action_index, log_prob, value, entropy)` shape run_simulation's
    mlp_network engine expects. log_prob/value/entropy aren't needed for
    eval, so they're just None. Shared with viewer.py for gameplay
    rendering, not just run_eval below."""

    def __init__(self, model):
        self.model = model

    def act(self, observation):
        return self.model.act_deterministic(observation), None, None, None


def run_eval(env_kwargs, n_episodes, seed, engine="mlp_network", model=None):
    """Run n_episodes under `engine` and return summary stats.

    Each episode's problem instance (and, for engine="random", its action
    sequence too) is deterministically derived from `seed`: episode i's
    GridEnvironment and run_simulation call are wrapped in
    `temporary_seed(episode_seeds[i])`, where episode_seeds is itself a
    deterministic function of `seed` and `n_episodes`. So calling this
    again with the same (seed, n_episodes) reproduces identical episodes --
    and calling it with a *different* engine but the *same* (seed,
    n_episodes) plays the exact same set of problem instances, which is
    what makes an eval run and a random-policy baseline run directly
    comparable.

    engine="mlp_network" requires `model` (network.ActorCritic; pass the
    unwrapped model if it's been through accelerator.prepare()) and always
    acts greedily via `DeterministicPolicy`. engine="random" needs no
    model.
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
    problem instances (see build_fixed_eval_set.py / fixed_eval_set.json)
    instead of episodes freshly derived from a seed range. Returns overall
    stats plus a per-category breakdown, so a checkpoint's performance on
    e.g. "long distance / boundary target" can be compared directly against
    "short / central".

    `dataset` is the parsed fixed_eval_set.json dict. Each entry's seed is
    replayed via temporary_seed(seed) + GridEnvironment(**env_kwargs), which
    reproduces that exact problem instance -- but only because the seed ->
    instance mapping depends solely on grid_size/subgrid_size/score_radius
    (the only env params that consume randomness during reset()), so those
    three are asserted to match what the dataset was generated with.
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

    def summarize(bucket):
        n = len(bucket["returns"])
        return {
            "avg_return": sum(bucket["returns"]) / n,
            "success_rate": sum(bucket["successes"]) / n,
            "avg_length": sum(bucket["lengths"]) / n,
            "n": n,
        }

    return {
        "overall": summarize(overall),
        "by_category": {category: summarize(bucket) for category, bucket in sorted(per_category.items())},
    }
