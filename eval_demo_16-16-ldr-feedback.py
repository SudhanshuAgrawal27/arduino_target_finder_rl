"""Runs a demo episode from a chosen checkpoint exactly like
eval_demo_16-16.py, except the policy's proximity observation comes from a
real photoresistor (LDR) reading instead of the environment's geometric
distance formula. The underlying game -- movement legality, and reaching
the target -- is untouched; only the "how warm am I" signal fed to the
network is replaced.

Pipeline:
  1. Offline, once: python3 eval_ldr_sweep.py --calibrate
     Sweeps the 8x8 window around the target, measuring how much the LDR's
     reading changes (delta from a single ambient baseline) as a function
     of Manhattan distance from the target, and saves that curve to
     ldr_calibration.json (see simulator.py's proximity_score for the exact
     discrete levels the trained policy expects: 1.0 at distance 0, down to
     0.0 at/past score_radius+1).
  2. This script, per run:
     - Draws the episode's subgrid boundary + target (no agent yet) and
       takes ONE ambient baseline reading -- this is the baseline for the
       whole game, not re-measured per step.
     - Loads ldr_calibration.json and, on every step, lights the agent
       alone (matching how calibration measured its reference deltas),
       reads the LDR, and nearest-neighbor-classifies the resulting delta
       against the calibrated levels to get a score in the same
       {1.0, ..., 0.0} set the policy was trained on.
     - Runs to termination (agent reaches the target) or truncation
       (max_steps), same as any other eval_demo, and reports steps/return/
       success plus the LED power estimate.
     - Immediately after, runs a second "perfect world" episode -- the same
       seed, so the same subgrid/start/target, but with the environment's
       normal noiseless proximity -- purely to report how many steps the
       noiseless optimum would have taken. Not shown on the board unless
       display_sim_run=true, and never touches the LDR.

Usage:
    python3 eval_ldr_sweep.py --calibrate       # once, before the first run
    python3 eval_demo_16-16-ldr-feedback.py checkpoint_dir=... seed=7
    python3 eval_demo_16-16-ldr-feedback.py display_sim_run=true
"""

import json
import time
from collections import deque
from pathlib import Path

import hydra
from accelerate import Accelerator
from omegaconf import OmegaConf

from arduino.led_board_controller.led_board_client import clear, connect, read_ldr, set_dynamic_layer, set_episode_layer
from arduino.led_board_controller.power_model import SUPPLY_VOLTAGE, frame_power_stats
from eval_lib import DeterministicPolicy
from network import ActorCritic
from simulator import GridEnvironment, State, run_simulation, temporary_seed


# Duplicated from eval_demo_16-16.py: Python can't `import` a module whose
# filename contains dashes, so compute_boundary_points/LedGridDisplay16x16
# can't be shared directly with that file -- keep both copies in sync by
# hand if either changes. LedGridDisplay16x16 is trimmed here (no dry_run
# parameter): this script's real run always has a board attached (it's
# reading the LDR from it), and the dummy run either skips display entirely
# (on_step=None) or uses a real connection -- there's no board-less case.
def compute_boundary_points(origin, subgrid_size, grid_size, margin):
    """The outline of the rectangle `margin` cells outside the subgrid
    [origin, origin + subgrid_size), clipped to the global [0, grid_size)
    board -- so it reads as a frame just around the playable area without
    ever overlapping it or wrapping off the physical board."""
    ox, oy = origin
    x_min, x_max = ox - margin, ox + subgrid_size - 1 + margin
    y_min, y_max = oy - margin, oy + subgrid_size - 1 + margin

    points = set()
    for x in range(x_min, x_max + 1):
        points.add((x, y_min))
        points.add((x, y_max))
    for y in range(y_min, y_max + 1):
        points.add((x_min, y))
        points.add((x_max, y))

    return [(x, y) for x, y in points if 0 <= x < grid_size and 0 <= y < grid_size]


class LedGridDisplay16x16:
    """Mirrors env state onto the 16x16 WS2812B panel in global coordinates,
    and tracks a per-step LED count/current estimate for power_summary().
    See eval_demo_16-16.py's version of this class for the full rationale
    -- this copy always has `ser` connected (no dry_run)."""

    def __init__(self, ser, trail_length, boundary_margin, step_delay_seconds=0.0):
        self.ser = ser
        self.trail = deque(maxlen=trail_length)
        self.boundary_margin = boundary_margin
        self.step_delay_seconds = step_delay_seconds
        self._boundary_count = 0
        self.step_power_stats = []

    def update(self, env):
        if env.steps == 0:
            boundary = compute_boundary_points(
                env.origin, env.subgrid_size, env.grid_size, self.boundary_margin
            )
            reply = set_episode_layer(self.ser, boundary, env.target_global)
            if reply != "OK":
                raise RuntimeError(f"LED board rejected episode layer: {reply!r}")
            self._boundary_count = len(boundary)
            self.trail.clear()
            self.step_power_stats = []

        agent_global = env.local_to_global(*env.agent_local)
        self.trail.append(agent_global)
        trail_points = (
            [] if env.terminated else [p for p in list(self.trail)[:-1] if p != agent_global]
        )

        reply = set_dynamic_layer(self.ser, agent_global, trail_points)
        if reply != "OK":
            raise RuntimeError(f"LED board rejected dynamic layer: {reply!r}")
        self.step_power_stats.append(
            frame_power_stats(self._boundary_count, len(trail_points), has_target=True, has_agent=True)
        )
        if self.step_delay_seconds:
            time.sleep(self.step_delay_seconds)

    def power_summary(self):
        if not self.step_power_stats:
            return None
        peak = max(self.step_power_stats, key=lambda s: s["ma_lit_phase"])
        n = len(self.step_power_stats)
        avg_leds = sum(s["leds_avg"] for s in self.step_power_stats) / n
        avg_ma = sum(s["ma_avg"] for s in self.step_power_stats) / n
        return {
            "peak_leds": peak["leds_lit_phase"],
            "peak_ma": peak["ma_lit_phase"],
            "avg_leds": avg_leds,
            "avg_ma": avg_ma,
        }


def build_ldr_proximity_fn(ser, env, baseline, calibration_levels, linger_seconds):
    """Replaces GridEnvironment._proximity's geometric distance formula with
    a real LDR reading. Lights the agent alone (empty trail) at `pos` --
    matching how eval_ldr_sweep.py --calibrate measured its reference
    deltas, so the live reading is comparable to the calibration curve --
    waits for the sensor to settle, then nearest-neighbor-classifies
    (reading - baseline) against the calibrated levels and returns that
    level's score. True game mechanics (movement legality, termination) are
    untouched; this is the sole point where LDR data enters the picture,
    called from inside GridEnvironment.perform_action."""
    def ldr_proximity(pos):
        global_pos = env.local_to_global(*pos)
        reply = set_dynamic_layer(ser, global_pos, [])
        if reply != "OK":
            raise RuntimeError(f"LED board rejected dynamic layer: {reply!r}")
        time.sleep(linger_seconds)
        delta = read_ldr(ser) - baseline
        nearest = min(calibration_levels.values(), key=lambda level: abs(level["avg_delta"] - delta))
        return nearest["score"]
    return ldr_proximity


def print_outcome(accelerator, label, trajectory, terminated):
    rewards = [step["reward"] for step in trajectory[1:]]
    steps = len(rewards)
    accelerator.print(f"[{label}] steps={steps} return={sum(rewards):.3f} success={terminated}")
    return steps


def print_power_summary(accelerator, display):
    power = display.power_summary()
    if power is None:
        return
    avg_w = power["avg_ma"] / 1000.0 * SUPPLY_VOLTAGE
    peak_w = power["peak_ma"] / 1000.0 * SUPPLY_VOLTAGE
    accelerator.print(
        f"LEDs: peak {power['peak_leds']} lit at once (~{power['peak_ma']:.0f}mA, ~{peak_w:.2f}W); "
        f"time-averaged ~{power['avg_leds']:.1f} effectively-on "
        f"(~{power['avg_ma']:.0f}mA, ~{avg_w:.2f}W) accounting for the blink duty cycle"
    )


@hydra.main(version_base=None, config_path="conf", config_name="eval_demo_config")
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

    with open(cfg.ldr_calibration_file) as f:
        calibration = json.load(f)
    calibration_levels = calibration["levels"]

    ser = connect(cfg.led_port, cfg.led_baud)
    reply = clear(ser)
    if reply != "OK":
        raise RuntimeError(f"LED board rejected clear: {reply!r}")

    # --- Real run: proximity comes from the LDR ---
    with temporary_seed(cfg.seed):
        env = GridEnvironment(**env_kwargs)

    # Draw the boundary + target with no agent yet, then take the single
    # ambient-baseline reading this whole game will be measured against.
    boundary = compute_boundary_points(env.origin, env.subgrid_size, env.grid_size, cfg.boundary_margin)
    reply = set_episode_layer(ser, boundary, env.target_global)
    if reply != "OK":
        raise RuntimeError(f"LED board rejected episode layer: {reply!r}")
    time.sleep(cfg.ldr_linger_seconds)
    baseline = read_ldr(ser)
    accelerator.print(f"LDR baseline for this game: {baseline}")

    env._proximity = build_ldr_proximity_fn(ser, env, baseline, calibration_levels, cfg.ldr_linger_seconds)

    # __init__'s own reset() already ran (with the TRUE simulated proximity)
    # before env._proximity could be patched -- redo just the initial
    # history/score now that the LDR-backed version is in place, matching
    # exactly what reset() does internally.
    initial_score = env._proximity(env.agent_local)
    env._current_score = initial_score
    env._history = deque(
        [State(x=env.agent_local[0], y=env.agent_local[1], score=initial_score)] * env.history_length,
        maxlen=env.history_length,
    )

    # step_delay_seconds=0: ldr_proximity's own settle wait already paces
    # every step -- an extra delay here would just double up the pause.
    real_display = LedGridDisplay16x16(ser, trail_length=env.history_length, boundary_margin=cfg.boundary_margin)
    real_trajectory = run_simulation(env=env, engine="mlp_network", network=network, on_step=real_display.update)

    accelerator.print(f"Checkpoint: {checkpoint_dir}")
    real_steps = print_outcome(accelerator, "LDR-feedback run", real_trajectory, env.terminated)
    print_power_summary(accelerator, real_display)

    # --- Dummy "perfect world" run: same episode, noiseless proximity ---
    with temporary_seed(cfg.seed):
        dummy_env = GridEnvironment(**env_kwargs)

    display_sim_run = cfg.get("display_sim_run", False)
    on_step = None
    dummy_display = None
    if display_sim_run:
        reply = clear(ser)
        if reply != "OK":
            raise RuntimeError(f"LED board rejected clear: {reply!r}")
        dummy_display = LedGridDisplay16x16(
            ser, trail_length=dummy_env.history_length, boundary_margin=cfg.boundary_margin,
            step_delay_seconds=cfg.step_delay_seconds,
        )
        on_step = dummy_display.update

    dummy_trajectory = run_simulation(env=dummy_env, engine="mlp_network", network=network, on_step=on_step)
    dummy_steps = print_outcome(accelerator, "perfect-world dummy run", dummy_trajectory, dummy_env.terminated)
    if dummy_display is not None:
        print_power_summary(accelerator, dummy_display)

    accelerator.print(f"LDR feedback cost {real_steps - dummy_steps} extra step(s) vs. the noiseless optimum")

    ser.close()


if __name__ == "__main__":
    main()
