"""Benchmarks a trained checkpoint's simulated performance against its real,
LDR-driven performance across the whole fixed evaluation dataset
(eval_fixed_dataset.json) -- unlike eval_demo_16-16-ldr-feedback.py, which
runs the same side-by-side comparison for just one seed at a time.

For every entry in the dataset, runs the same two passes
eval_demo_16-16-ldr-feedback.py runs for one seed: a noiseless "perfect
world" simulation, then a real LDR-driven episode with the identical
subgrid/target/start. This runs headless -- no board display, no target
preview, no "thinking" overlay, no per-step delay -- since it's meant to run
unattended over ~100 episodes rather than for a person to watch. The one
physical step that *is* required to take each LDR reading (lighting the
probe LED at the candidate position before reading the sensor) still
happens, via build_ldr_proximity_fn imported directly from
eval_demo_16-16-ldr-feedback.py.

Requires a board connected and `python3 eval_ldr_sweep.py --calibrate`
already run (same prerequisite as eval_demo_16-16-ldr-feedback.py).

Usage:
    python3 eval_ldr_benchmark.py checkpoint_dir=trained_models/h64_l3_hist4_ep150_seed43/epoch_150
"""

import importlib.util
import json
import time
from collections import deque
from pathlib import Path

import hydra
from accelerate import Accelerator
from omegaconf import OmegaConf
from tqdm import tqdm

from arduino.led_board_controller.led_board_client import clear, connect, read_ldr, set_episode_layer
from eval_lib import DeterministicPolicy
from network import ActorCritic
from simulator import GridEnvironment, State, run_simulation, temporary_seed

# eval_demo_16-16-ldr-feedback.py's filename has dashes, so it can't be
# `import`ed by name -- load it by path instead, to reuse its
# build_ldr_proximity_fn/compute_boundary_points/NO_TARGET rather than
# duplicating them here (see that file's own note on the same problem with
# eval_demo_16-16.py).
_spec = importlib.util.spec_from_file_location(
    "eval_demo_16_16_ldr_feedback", Path(__file__).parent / "eval_demo_16-16-ldr-feedback.py"
)
_ldr_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ldr_demo)


def run_pair(env_kwargs, entry_seed, network, ser, calibration_levels, ldr_linger_seconds, boundary_margin):
    """Runs the perfect-world sim pass and the real LDR pass for one fixed-
    dataset entry -- same seed for both, so identical subgrid/target/start
    -- mirroring eval_demo_16-16-ldr-feedback.py's main() but headless.
    Returns (sim_steps, sim_success, real_steps, real_success)."""
    with temporary_seed(entry_seed):
        sim_env = GridEnvironment(**env_kwargs)
    sim_trajectory = run_simulation(env=sim_env, engine="mlp_network", network=network)
    sim_steps = len(sim_trajectory) - 1

    with temporary_seed(entry_seed):
        real_env = GridEnvironment(**env_kwargs)

    reply = clear(ser)
    if reply != "OK":
        raise RuntimeError(f"LED board rejected clear: {reply!r}")

    # Baseline must be measured under the same conditions
    # eval_demo_16-16-ldr-feedback.py uses (boundary drawn, target hidden)
    # so it's comparable to the calibration curve.
    boundary = _ldr_demo.compute_boundary_points(
        real_env.origin, real_env.subgrid_size, real_env.grid_size, boundary_margin
    )
    reply = set_episode_layer(ser, boundary, _ldr_demo.NO_TARGET)
    if reply != "OK":
        raise RuntimeError(f"LED board rejected episode layer: {reply!r}")
    time.sleep(ldr_linger_seconds)
    baseline = read_ldr(ser)

    real_env._proximity = _ldr_demo.build_ldr_proximity_fn(
        ser, real_env, baseline, calibration_levels, ldr_linger_seconds
    )
    # Redo the initial history/score now that the LDR-backed proximity is in
    # place, exactly matching what GridEnvironment.reset() does internally
    # (see eval_demo_16-16-ldr-feedback.py's main() for the same fixup).
    initial_score = real_env._proximity(real_env.agent_local)
    real_env._current_score = initial_score
    real_env._history = deque(
        [State(x=real_env.agent_local[0], y=real_env.agent_local[1], score=initial_score)] * real_env.history_length,
        maxlen=real_env.history_length,
    )

    real_trajectory = run_simulation(env=real_env, engine="mlp_network", network=network)
    real_steps = len(real_trajectory) - 1

    return sim_steps, sim_env.terminated, real_steps, real_env.terminated


@hydra.main(version_base=None, config_path="conf", config_name="eval_ldr_benchmark_config")
def main(cfg):
    checkpoint_dir = Path(cfg.checkpoint_dir)
    run_dir = checkpoint_dir.parent
    train_cfg = OmegaConf.load(run_dir / "config.yaml")

    model = ActorCritic(
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
    network = DeterministicPolicy(model)

    env_kwargs = dict(
        grid_size=train_cfg.env.grid_size,
        subgrid_size=train_cfg.env.subgrid_size,
        score_radius=train_cfg.env.score_radius,
        max_steps=train_cfg.env.max_steps,
        history_length=train_cfg.env.history_length,
        step_penalty=train_cfg.env.step_penalty,
        wall_penalty=train_cfg.env.get("wall_penalty", 0.0),
        success_bonus=train_cfg.env.success_bonus,
    )

    with open(cfg.dataset_path) as f:
        dataset = json.load(f)
    for key in ("grid_size", "subgrid_size", "score_radius"):
        if dataset[key] != env_kwargs[key]:
            raise ValueError(
                f"env_kwargs[{key!r}]={env_kwargs[key]!r} does not match the fixed "
                f"dataset's {key}={dataset[key]!r} -- the recorded seeds would "
                f"reproduce different problem instances than the ones the dataset "
                f"was categorized for."
            )

    with open(cfg.ldr_calibration_file) as f:
        calibration_levels = json.load(f)["levels"]

    ser = connect(cfg.led_port, cfg.led_baud)

    sim_steps_all, real_steps_all = [], []
    sim_success_all, real_success_all = [], []
    real_wins = sim_wins = ties = 0

    for entry in tqdm(dataset["entries"], desc="benchmark"):
        sim_steps, sim_ok, real_steps, real_ok = run_pair(
            env_kwargs, entry["seed"], network, ser, calibration_levels,
            cfg.ldr_linger_seconds, cfg.boundary_margin,
        )
        sim_steps_all.append(sim_steps)
        real_steps_all.append(real_steps)
        sim_success_all.append(sim_ok)
        real_success_all.append(real_ok)
        if real_steps < sim_steps:
            real_wins += 1
        elif sim_steps < real_steps:
            sim_wins += 1
        else:
            ties += 1

    ser.close()

    n = len(dataset["entries"])
    accelerator.print(f"Checkpoint: {checkpoint_dir}")
    accelerator.print(f"Benchmarked {n} fixed-dataset instances\n")
    accelerator.print(f"{'':14s}{'Avg Steps':>12}{'Success Rate':>14}")
    accelerator.print(f"{'Simulation':14s}{sum(sim_steps_all) / n:>12.2f}{sum(sim_success_all) / n:>14.3f}")
    accelerator.print(f"{'Real (LDR)':14s}{sum(real_steps_all) / n:>12.2f}{sum(real_success_all) / n:>14.3f}")
    accelerator.print("")
    accelerator.print(f"Real won (fewer steps): {real_wins}/{n}")
    accelerator.print(f"Sim won (fewer steps):  {sim_wins}/{n}")
    accelerator.print(f"Ties:                   {ties}/{n}")


if __name__ == "__main__":
    main()
