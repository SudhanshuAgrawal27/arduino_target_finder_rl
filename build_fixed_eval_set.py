"""Generate a fixed, categorized evaluation dataset of 100 GridEnvironment
problem instances, for testing any checkpoint's performance broken down by
task difficulty (see conf notes / README for how eval_fixed.py consumes it).

Categories cross two axes:
  - distance tier: short (1-4) / medium (5-8) / long (9-14) -- the Manhattan
    distance from start to target, i.e. the optimal number of steps.
  - target locality: central (the target sits >= score_radius from every
    subgrid edge, so its score_radius "warmth halo" is fully unclipped) vs
    boundary (the halo is clipped by a subgrid edge on at least one side).

Reuses the existing seed-reproducibility idiom (temporary_seed +
GridEnvironment) instead of adding an explicit-instance constructor to
GridEnvironment: each manifest entry is a seed that, replayed through
temporary_seed(seed) + GridEnvironment(**same grid/subgrid/radius),
deterministically reproduces one specific problem instance -- see
evaluation.run_eval for the same pattern.
"""

import json
import random

from simulator import GRID_SIZE, SCORE_RADIUS, SUBGRID_SIZE, GridEnvironment, temporary_seed

DISTANCE_TIERS = [
    ("short", 1, 4),
    ("medium", 5, 8),
    ("long", 9, 2 * (SUBGRID_SIZE - 1)),
]

CATEGORY_COUNTS = {
    ("short", "central"): 17,
    ("short", "boundary"): 17,
    ("medium", "central"): 17,
    ("medium", "boundary"): 17,
    ("long", "central"): 16,
    ("long", "boundary"): 16,
}
assert sum(CATEGORY_COUNTS.values()) == 100

GENERATION_SEED = 20260709  # fixed seed for the search itself, so the manifest is reproducible
OUTPUT_PATH = "fixed_eval_set.json"


def target_locality(target_local, subgrid_size=SUBGRID_SIZE, score_radius=SCORE_RADIUS):
    x, y = target_local
    edge_dist = min(x, subgrid_size - 1 - x, y, subgrid_size - 1 - y)
    return "central" if edge_dist >= score_radius else "boundary"


def distance_tier(distance):
    for name, lo, hi in DISTANCE_TIERS:
        if lo <= distance <= hi:
            return name
    return None


def main():
    rng = random.Random(GENERATION_SEED)
    remaining = dict(CATEGORY_COUNTS)
    entries = []
    seen_seeds = set()
    seen_instances = set()  # (origin, start_local): the state space is small enough (~64x63)
                             # that distinct seeds regularly land on the same instance -- dedupe
                             # on the actual instance, not just the seed.

    while remaining:
        seed = rng.randint(0, 2**31 - 1)
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)

        with temporary_seed(seed):
            env = GridEnvironment(grid_size=GRID_SIZE, subgrid_size=SUBGRID_SIZE, score_radius=SCORE_RADIUS)

        instance_key = (env.origin, env.start_local)
        if instance_key in seen_instances:
            continue

        distance = abs(env.start_local[0] - env.target_local[0]) + abs(env.start_local[1] - env.target_local[1])
        key = (distance_tier(distance), target_locality(env.target_local))
        if key not in remaining:
            continue

        seen_instances.add(instance_key)
        entries.append({
            "id": len(entries),
            "seed": seed,
            "distance_tier": key[0],
            "target_locality": key[1],
            "distance": distance,
            "origin": env.origin,
            "target_local": env.target_local,
            "start_local": env.start_local,
        })
        remaining[key] -= 1
        if remaining[key] == 0:
            del remaining[key]

    manifest = {
        "grid_size": GRID_SIZE,
        "subgrid_size": SUBGRID_SIZE,
        "score_radius": SCORE_RADIUS,
        "generation_seed": GENERATION_SEED,
        "category_counts": {f"{tier}/{locality}": n for (tier, locality), n in CATEGORY_COUNTS.items()},
        "distance_tiers": {name: [lo, hi] for name, lo, hi in DISTANCE_TIERS},
        "target_locality_definitions": {
            "central": f"target >= {SCORE_RADIUS} cells from every subgrid edge (score_radius halo fully unclipped)",
            "boundary": f"target within {SCORE_RADIUS} cells of a subgrid edge (score_radius halo clipped on at least one side)",
        },
        "entries": entries,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(entries)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
