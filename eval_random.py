"""Standalone evaluation: load a saved training checkpoint and run
deterministic evaluation episodes independently of training, drawing fresh
random episodes from a seed each time (see eval_fixed_dataset.py for
replaying a fixed, categorized set of problem instances instead).

Usage:
    python3 eval_random.py checkpoint_dir=trained_models/2026-07-06_14-30-05/epoch_1
    python3 eval_random.py checkpoint_dir=... episodes=200 seed=7
"""

from pathlib import Path

import hydra
from accelerate import Accelerator
from omegaconf import OmegaConf

from eval_lib import run_eval
from network import ActorCritic


@hydra.main(version_base=None, config_path="conf", config_name="eval_random_config")
def main(cfg):
    checkpoint_dir = Path(cfg.checkpoint_dir)
    run_dir = checkpoint_dir.parent
    train_cfg = OmegaConf.load(run_dir / "config.yaml")

    model = ActorCritic(
        # older checkpoints predate these fields; they used the full history and 2 layers
        window_length=train_cfg.network.get("window_length", train_cfg.env.history_length),
        hidden_dim=train_cfg.network.hidden_dim,
        num_layers=train_cfg.network.get("num_layers", 2),
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

    stats = run_eval(env_kwargs, cfg.episodes, seed=cfg.seed, engine="mlp_network", model=model)
    accelerator.print(f"Checkpoint: {checkpoint_dir}")
    accelerator.print(
        f"avg_return={stats['avg_return']:.3f} success_rate={stats['success_rate']:.3f} "
        f"avg_length={stats['avg_length']:.1f} (over {cfg.episodes} episodes, seed={cfg.seed})"
    )


if __name__ == "__main__":
    main()
