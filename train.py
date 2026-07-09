"""PPO training loop for the grid-search environment.

Usage:
    python3 train.py
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

from evaluation import run_eval
from network import ActorCritic
from ppo import ppo_loss
from rollout import collect_rollouts
from simulator import derive_episode_seeds
from simulator import set_global_seed as set_env_seed


def _load_dotenv(path=".env"):
    """Load KEY=VALUE lines from a .env file into os.environ, without
    overriding anything already set in the environment (e.g. an explicit
    `export WANDB_API_KEY=...` in the shell wins). Pure stdlib -- avoids
    pulling in python-dotenv for one small file. Used so WANDB_API_KEY
    persists in .env (see .gitignore) across container recreation, without
    needing `wandb login` re-run every session."""
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
        success_bonus=cfg.env.success_bonus,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    _load_dotenv()

    run_name = cfg.output.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg.output.base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # mode="online" uploads live to your wandb.ai account -- requires
    # `wandb login` (an API key) to have already been run once on this
    # machine; nothing here can do that login for you. Run files still also
    # land in run_dir for local reference regardless of mode.
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        dir=str(run_dir),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"wandb run: {run.url}")
    workspace_url_file = Path(__file__).parent / "conf" / "wandb_workspace_url.txt"
    if workspace_url_file.exists():
        print(f"wandb combined train/eval/baseline view: {workspace_url_file.read_text().strip()}")

    # accelerate's set_seed covers torch/numpy/random (and so, transitively,
    # the environment -- simulator.py's randomness is plain `random` under
    # the hood). Calling simulator's own seed function too is redundant but
    # keeps the environment's reproducibility contract explicit here rather
    # than relying on that implementation detail.
    set_seed(cfg.seed)
    set_env_seed(cfg.seed)

    accelerator = Accelerator()

    model = ActorCritic(
        history_length=cfg.env.history_length,
        hidden_dim=cfg.network.hidden_dim,
        subgrid_size=cfg.env.subgrid_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.learning_rate)
    model, optimizer = accelerator.prepare(model, optimizer)
    # Standard PPO trick: anneal LR linearly over training so early updates
    # can move fast while late updates (near convergence) don't keep
    # churning the policy -- a constant LR never fully stops oscillating
    # (SGD/Adam settle into a noise ball whose size scales with LR), so
    # decay is what actually damps that out over time. Floored at 10% of
    # the initial LR rather than 0, so the policy still has room to keep
    # adapting late in training if it hasn't fully converged by the last
    # epoch. Stepped once per epoch, so total_iters is in epochs, not
    # minibatch updates. No warmup: PPO's ratio clipping (clip_eps) and
    # grad-norm clipping already bound how far a single update can move
    # the policy, which is what warmup would otherwise protect against.
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.training.num_epochs
    )
    # Rollout collection and eval call custom methods (.act, .get_value,
    # .act_deterministic) that only exist on the unwrapped module -- and
    # since a single stateful environment can't be meaningfully sharded
    # across processes anyway, collection always uses the unwrapped model.
    # (This project targets single-process training; evaluate_actions()
    # calling self.forward() directly means multi-process DDP gradient sync
    # is not handled here.)
    raw_model = accelerator.unwrap_model(model)

    env_kwargs = _env_kwargs(cfg)

    accelerator.print(f"Run directory: {run_dir}")

    for epoch in range(1, cfg.training.num_epochs + 1):
        model.train()
        epoch_lr = optimizer.param_groups[0]["lr"]
        batch, rollout_stats = collect_rollouts(
            env_kwargs, raw_model, cfg.training.episodes_per_epoch, cfg.ppo.gamma, cfg.ppo.lam,
            desc=f"epoch {epoch} train rollout",
        )

        # Collected now (pre-update, same as `batch` above) and only scored
        # against ppo_loss after this epoch's gradient updates below -- so
        # its old_log_probs/returns reflect the policy *before* this
        # epoch's update and new_log_prob/value reflect the policy *after*,
        # giving a real ratio/KL rather than a model compared to itself.
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
        # Same eval_seed every epoch -> same set of games each time, so
        # eval_stats is directly comparable epoch-to-epoch. Same seed for
        # the random baseline too -> both play the identical episodes this
        # epoch, so eval vs baseline is a fair, matched comparison.
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

        # Validation loss: the actual PPO objective terms (not gameplay
        # stats) on the same fixed eval-seed episode set, under this
        # epoch's update -- reveals whether the loss the optimizer is
        # minimizing generalizes past the specific batch collect_rollouts
        # drew this epoch, which train/eval/baseline above (gameplay
        # outcomes only) can't show. Runs every epoch, same cadence as eval.
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