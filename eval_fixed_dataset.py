"""Standalone evaluation against the fixed, categorized eval dataset
(eval_fixed_dataset.json, built by build_eval_fixed_dataset.py). Unlike
eval_random.py, which draws fresh episodes from a seed range, this replays
the exact same 100 problem instances every time and reports a breakdown by
difficulty category alongside the overall stats.

Usage:
    python3 eval_fixed_dataset.py checkpoint_dir=trained_models/2026-07-06_14-30-05/epoch_1
    python3 eval_fixed_dataset.py checkpoint_dir=... dataset_path=eval_fixed_dataset.json
"""

import json
from pathlib import Path

import hydra
from accelerate import Accelerator
from omegaconf import OmegaConf

from eval_lib import run_eval_fixed
from network import ActorCritic


@hydra.main(version_base=None, config_path="conf", config_name="eval_fixed_dataset_config")
def main(cfg):
    checkpoint_dir = Path(cfg.checkpoint_dir)
    run_dir = checkpoint_dir.parent
    train_cfg = OmegaConf.load(run_dir / "config.yaml")

    model = ActorCritic(
        history_length=train_cfg.env.history_length,
        hidden_dim=train_cfg.network.hidden_dim,
        subgrid_size=train_cfg.env.subgrid_size,
    )
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    accelerator.load_state(str(checkpoint_dir))
    model = accelerator.unwrap_model(model)
    model.eval()

    env_kwargs = dict(
        grid_size=train_cfg.env.grid_size,
        subgrid_size=train_cfg.env.subgrid_size,
        score_radius=train_cfg.env.score_radius,
        max_steps=train_cfg.env.max_steps,
        history_length=train_cfg.env.history_length,
        step_penalty=train_cfg.env.step_penalty,
        wall_penalty=train_cfg.env.get("wall_penalty", 0.0),  # older checkpoints predate this field
        success_bonus=train_cfg.env.success_bonus,
    )

    with open(cfg.dataset_path) as f:
        dataset = json.load(f)

    stats = run_eval_fixed(env_kwargs, dataset, engine="mlp_network", model=model)

    accelerator.print(f"Checkpoint: {checkpoint_dir}\n")
    print_legend(dataset, accelerator.print)
    print_report(stats, accelerator.print)


def print_legend(dataset, printer=print):
    """Explain what each category label means, sourced from the dataset
    manifest itself (see build_eval_fixed_dataset.py) so it can't drift out
    of sync with however the categories were actually generated."""
    printer("Category legend:")
    printer("  distance tier (start -> target, Manhattan steps):")
    for name, (lo, hi) in dataset["distance_tiers"].items():
        printer(f"    {name:8s} = {lo}-{hi} steps")
    printer("  target locality:")
    for name, description in dataset["target_locality_definitions"].items():
        printer(f"    {name:8s} = {description}")
    printer("")


def print_report(stats, printer=print):
    """Overall + per-category stats as an aligned table (rather than raw
    key=value dumps) so rows are easy to scan and compare."""
    columns = ["Category", "Avg Return", "Success Rate", "Avg Length", "N"]
    printer(f"{columns[0]:<20}{columns[1]:>12}{columns[2]:>14}{columns[3]:>12}{columns[4]:>6}")

    def row(name, s):
        printer(f"{name:<20}{s['avg_return']:>12.3f}{s['success_rate']:>14.3f}{s['avg_length']:>12.1f}{s['n']:>6d}")

    row("OVERALL", stats["overall"])
    for category, s in stats["by_category"].items():
        row(category.replace("/", " / "), s)


if __name__ == "__main__":
    main()
