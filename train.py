"""PPO training loop for the grid-search environment.

Trains one model per seed in cfg.seeds (default [42, 43, 44]), each into its
own run directory, so a single invocation produces a multi-seed ablation
point. Override to a single seed for parallel per-(config, seed) launching:

    python3 train.py                                  # all of cfg.seeds
    python3 train.py --config-name config_train_h96_l2_hist4
    python3 train.py --config-name config_train_h96_l2_hist4 seeds=[43]
    python3 train.py training.num_epochs=5 ppo.learning_rate=1e-4
"""

import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm

from eval_lib import run_eval
from network import ActorCritic
from ppo import ppo_loss
from rollout import collect_rollouts
from simulator import derive_episode_seeds
from simulator import set_global_seed as set_env_seed


def _load_dotenv(path=".env"):
    """Load KEY=VALUE lines from a .env file into os.environ, without
    overriding anything already set in the environment."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if value:
                os.environ.setdefault(key, value)


def _env_kwargs(cfg):
    return dict(
        grid_size=cfg.env.grid_size,
        subgrid_size=cfg.env.subgrid_size,
        score_radius=cfg.env.score_radius,
        max_steps=cfg.env.max_steps,
        history_length=cfg.env.history_length,
        step_penalty=cfg.env.step_penalty,
        wall_penalty=cfg.env.wall_penalty,
        success_bonus=cfg.env.success_bonus,
    )


def _run_name(cfg, timestamp, seed):
    """<timestamp>_<run_name>_seed<seed>, with each piece optional. The
    timestamp is dropped when output.timestamp is false, giving a
    deterministic per-(config, seed) directory the launcher can skip if it's
    already complete."""
    parts = []
    if cfg.output.get("timestamp", True):
        parts.append(timestamp)
    if cfg.output.run_name:
        parts.append(str(cfg.output.run_name))
    parts.append(f"seed{seed}")
    return "_".join(parts)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    _load_dotenv()

    if cfg.network.window_length > cfg.env.history_length:
        raise ValueError(
            f"network.window_length ({cfg.network.window_length}) cannot exceed "
            f"env.history_length ({cfg.env.history_length}) -- the network can't see "
            f"more history than the environment produces."
        )

    seeds = list(cfg.seeds) if OmegaConf.is_list(cfg.seeds) else [cfg.seeds]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for seed in seeds:
        train_one_seed(cfg, seed, timestamp)


def train_one_seed(cfg, seed, timestamp):
    """Train a single model at `seed` and checkpoint every epoch into its own
    run directory."""
    run_name = _run_name(cfg, timestamp, seed)
    run_dir = Path(cfg.output.base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the fully-resolved config plus the exact seed this run used, so
    # eval scripts can reconstruct the architecture and the run is traceable.
    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    run_cfg.seed_used = seed
    OmegaConf.save(run_cfg, run_dir / "config.yaml")

    # mode="online" requires `wandb login` to have already been run once.
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        dir=str(run_dir),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(run_cfg, resolve=True),
        reinit=True,
    )
    print(f"wandb run: {run.url}")
    workspace_url_file = Path(__file__).parent / "conf" / "wandb_workspace_url.txt"
    if workspace_url_file.exists():
        print(f"wandb combined train/eval/baseline view: {workspace_url_file.read_text().strip()}")

    # accelerate's set_seed covers torch/numpy/random; set_env_seed re-seeds
    # the env's `random` stream so episode geometry is reproducible from seed.
    set_seed(seed)
    set_env_seed(seed)

    accelerator = Accelerator()

    model = ActorCritic(
        window_length=cfg.network.window_length,
        hidden_dim=cfg.network.hidden_dim,
        num_layers=cfg.network.num_layers,
        subgrid_size=cfg.env.subgrid_size,
    )
    # Isolate model-init RNG: orthogonal init consumes a width/depth-dependent
    # number of torch draws, which would otherwise leave the downstream
    # exploration-sampling and minibatch-shuffle stream at an
    # architecture-dependent offset. Re-seeding torch here makes that stream
    # depend only on `seed`, so two architectures at the same seed see
    # identical exploration/shuffle noise (a controlled comparison) while
    # different seeds still vary the whole run (init included).
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.learning_rate)
    model, optimizer = accelerator.prepare(model, optimizer)
    # Linear LR decay to 10% of the initial rate, stepped once per epoch, to
    # damp late-training oscillation. No warmup: PPO's ratio/grad-norm
    # clipping already bounds how far one update can move the policy.
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.training.num_epochs
    )
    # Rollout/eval call .act/.get_value/.act_deterministic, which only exist
    # on the unwrapped module.
    raw_model = accelerator.unwrap_model(model)

    env_kwargs = _env_kwargs(cfg)

    accelerator.print(f"Run directory: {run_dir} (seed {seed})")

    for epoch in range(1, cfg.training.num_epochs + 1):
        model.train()
        epoch_lr = optimizer.param_groups[0]["lr"]
        batch, rollout_stats = collect_rollouts(
            env_kwargs, raw_model, cfg.training.episodes_per_epoch, cfg.ppo.gamma, cfg.ppo.lam,
            desc=f"epoch {epoch} train rollout",
        )

        # Collected pre-update, scored against ppo_loss after this epoch's
        # gradient updates below -- gives a real ratio/KL against the
        # updated policy rather than a model compared to itself.
        val_batch, _ = collect_rollouts(
            env_kwargs, raw_model, cfg.training.eval_episodes, cfg.ppo.gamma, cfg.ppo.lam,
            episode_seeds=derive_episode_seeds(cfg.training.eval_seed, cfg.training.eval_episodes),
            desc=f"epoch {epoch} val rollout",
        )

        device = accelerator.device
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        old_log_probs = batch["old_log_probs"].to(device)
        advantages = batch["advantages"].to(device)
        returns = batch["returns"].to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = observations.shape[0]
        minibatch_stats = []
        n_minibatches = -(-n // cfg.ppo.minibatch_size)  # ceil div
        with tqdm(total=cfg.ppo.update_epochs * n_minibatches, desc=f"epoch {epoch} update", leave=False) as pbar:
            for _ in range(cfg.ppo.update_epochs):
                perm = torch.randperm(n, device=device)
                for start in range(0, n, cfg.ppo.minibatch_size):
                    idx = perm[start:start + cfg.ppo.minibatch_size]
                    loss, stats = ppo_loss(
                        model,
                        observations[idx], actions[idx], old_log_probs[idx],
                        advantages[idx], returns[idx],
                        cfg.ppo.clip_eps, cfg.ppo.value_coef, cfg.ppo.entropy_coef,
                    )
                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), cfg.ppo.max_grad_norm)
                    optimizer.step()
                    minibatch_stats.append(stats)
                    pbar.update(1)
        lr_scheduler.step()

        avg_loss_stats = {
            key: sum(s[key] for s in minibatch_stats) / len(minibatch_stats)
            for key in minibatch_stats[0]
        }

        n_ep = len(rollout_stats["episode_returns"])
        train_avg_return = sum(rollout_stats["episode_returns"]) / n_ep
        train_success_rate = sum(rollout_stats["successes"]) / n_ep
        train_avg_length = sum(rollout_stats["episode_lengths"]) / n_ep
        accelerator.print(
            f"[epoch {epoch}] train: avg_return={train_avg_return:.3f} "
            f"success_rate={train_success_rate:.3f} avg_length={train_avg_length:.1f} "
            f"loss={avg_loss_stats['loss']:.4f} "
            f"policy_loss={avg_loss_stats['policy_loss']:.4f} "
            f"value_loss={avg_loss_stats['value_loss']:.4f} "
            f"entropy={avg_loss_stats['entropy']:.4f} "
            f"approx_kl={avg_loss_stats['approx_kl']:.4f} "
            f"lr={epoch_lr:.2e}"
        )

        model.eval()
        # Same eval_seed every epoch, for both eval and baseline: makes
        # eval_stats comparable epoch-to-epoch and eval-vs-baseline a fair,
        # matched comparison.
        eval_stats = run_eval(
            env_kwargs, cfg.training.eval_episodes, seed=cfg.training.eval_seed,
            engine="mlp_network", model=raw_model,
        )
        baseline_stats = run_eval(
            env_kwargs, cfg.training.eval_episodes, seed=cfg.training.eval_seed,
            engine="random",
        )
        accelerator.print(
            f"[epoch {epoch}] eval:     avg_return={eval_stats['avg_return']:.3f} "
            f"success_rate={eval_stats['success_rate']:.3f} avg_length={eval_stats['avg_length']:.1f}"
        )
        accelerator.print(
            f"[epoch {epoch}] baseline: avg_return={baseline_stats['avg_return']:.3f} "
            f"success_rate={baseline_stats['success_rate']:.3f} avg_length={baseline_stats['avg_length']:.1f}"
        )

        log_dict = {
            "epoch": epoch,
            "train/learning_rate": epoch_lr,
            "train/avg_return": train_avg_return,
            "train/success_rate": train_success_rate,
            "train/avg_length": train_avg_length,
            "train/loss": avg_loss_stats["loss"],
            "train/policy_loss": avg_loss_stats["policy_loss"],
            "train/value_loss": avg_loss_stats["value_loss"],
            "train/entropy": avg_loss_stats["entropy"],
            "train/approx_kl": avg_loss_stats["approx_kl"],
            "eval/avg_return": eval_stats["avg_return"],
            "eval/success_rate": eval_stats["success_rate"],
            "eval/avg_length": eval_stats["avg_length"],
            "baseline/avg_return": baseline_stats["avg_return"],
            "baseline/success_rate": baseline_stats["success_rate"],
            "baseline/avg_length": baseline_stats["avg_length"],
        }

        # Validation loss: PPO objective terms on the fixed eval-seed set,
        # under this epoch's update -- checks whether the loss generalizes
        # past the batch collect_rollouts drew this epoch.
        val_advantages = val_batch["advantages"].to(device)
        val_advantages = (val_advantages - val_advantages.mean()) / (val_advantages.std() + 1e-8)
        with torch.no_grad():
            _, val_stats = ppo_loss(
                raw_model,
                val_batch["observations"].to(device), val_batch["actions"].to(device),
                val_batch["old_log_probs"].to(device), val_advantages, val_batch["returns"].to(device),
                cfg.ppo.clip_eps, cfg.ppo.value_coef, cfg.ppo.entropy_coef,
            )
        accelerator.print(
            f"[epoch {epoch}] val:      loss={val_stats['loss']:.4f} "
            f"policy_loss={val_stats['policy_loss']:.4f} "
            f"value_loss={val_stats['value_loss']:.4f} "
            f"entropy={val_stats['entropy']:.4f} "
            f"approx_kl={val_stats['approx_kl']:.4f}"
        )
        log_dict.update({
            "val/loss": val_stats["loss"],
            "val/policy_loss": val_stats["policy_loss"],
            "val/value_loss": val_stats["value_loss"],
            "val/entropy": val_stats["entropy"],
            "val/approx_kl": val_stats["approx_kl"],
        })

        wandb.log(log_dict, step=epoch)

        checkpoint_dir = run_dir / f"epoch_{epoch}"
        accelerator.save_state(str(checkpoint_dir))
        accelerator.print(f"[epoch {epoch}] saved checkpoint to {checkpoint_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
