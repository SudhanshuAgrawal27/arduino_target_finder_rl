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
  2. This script, per run, ALWAYS runs two episodes back to back, same seed
     (so the same subgrid/start/target) for both:
     a. A "perfect world" episode first, with the environment's normal
        noiseless proximity -- shown on the board with the target visibly
        lit (dim blue), since nothing about this pass depends on hiding it.
        Purely a reference for how many steps the noiseless optimum takes;
        never touches the LDR. Simulation-only extra: right before each
        move, briefly lights the 4 candidate next cells (one per action) in
        shades of yellow scaled by that action's policy probability, so a
        viewer can see what the agent is "thinking" before it commits to a
        direction -- purely cosmetic, and never shown during the real
        LDR-driven episode below (which already hides the target for the
        same reason: not giving away information the agent can't itself
        sense).
     b. The real LDR-driven episode second -- proximity now comes from an
        actual photoresistor reading instead of geometric distance. Draws
        the boundary and takes one ambient baseline reading up front (the
        baseline for the whole game, not re-measured per step) with the
        target unlit, then briefly pulses the target blue a few times --
        purely so a human watching the board can see where it is -- before
        it disappears for good and the game actually starts. During play
        the target stays deliberately UNLIT: showing it would give away
        visually what the agent is supposed to be sensing for itself. On
        every step, lights the agent alone (matching how calibration
        measured its reference deltas), reads the LDR, and nearest-
        neighbor-classifies the resulting delta against ldr_calibration.json
        to get a score in the same {1.0, ..., 0.0} set the policy was
        trained on. True game mechanics (movement, termination on reaching
        the target) are untouched -- only this one signal is replaced.
     Once both finish, prints steps/return/success/power for both and the
     step-count delta between them.

Usage:
    python3 eval_ldr_sweep.py --calibrate       # once, before the first run
    python3 eval_demo_16-16-ldr-feedback.py checkpoint_dir=... seed=7
"""

import json
import time
from collections import deque
from pathlib import Path

import hydra
from accelerate import Accelerator
from omegaconf import OmegaConf

from arduino.led_board_controller.led_board_client import (
    clear, connect, read_ldr, set_dynamic_layer, set_episode_layer, set_thinking_layer,
)
from arduino.led_board_controller.power_model import SUPPLY_VOLTAGE, frame_power_stats
from eval_lib import DeterministicPolicy
from network import ActorCritic
from simulator import ACTION_DELTAS, GridEnvironment, State, run_simulation, temporary_seed


# Duplicated from eval_demo_16-16.py: Python can't `import` a module whose
# filename contains dashes, so compute_boundary_points/LedGridDisplay16x16
# can't be shared directly with that file -- keep both copies in sync by
# hand if either changes. LedGridDisplay16x16 is trimmed here (no dry_run
# parameter): both episodes this script runs always have a board attached.
NO_TARGET = (-1, -1)  # see LedGridDisplay16x16's show_target docstring

# led_board_client's own functions already retry internally (see its
# _MAX_ATTEMPTS) but real hardware runs have shown that budget can still be
# exhausted -- around 10% of the time per command, per an eval_ldr_sweep.py
# --calibrate run (see arduino/README.md's "Known open issue"). Wrapping
# calls here with a few more full attempts adds extra margin on top of
# that; how a caller reacts to still-persistent failure after that is a
# per-call-site decision -- see the two different uses below.
_EXTRA_RETRY_ATTEMPTS = 3

# How the one-time "here's where the target is" preview blinks before the
# real LDR-driven run starts: on/off in equal halves, repeated this many
# times, purely a visual cue for a human watching the board -- the RL
# episode itself never sees the target lit (show_target=False below).
_TARGET_PREVIEW_CYCLES = 3
_TARGET_PREVIEW_ON_SECONDS = 0.5
_TARGET_PREVIEW_OFF_SECONDS = 0.5

# How long the simulation-only "thinking" preview (see show_thinking) stays
# lit before the move it previews actually happens.
_THINKING_ON_SECONDS = 0.4


def _retry_send(fn, *args, label):
    """Calls a led_board_client function up to _EXTRA_RETRY_ATTEMPTS times
    if it keeps coming back non-"OK", printing a warning on each failed
    attempt. Returns the final reply either way -- it's up to the caller to
    decide whether a still-persistent failure is fatal (see call sites)."""
    reply = "ERR"
    for attempt in range(_EXTRA_RETRY_ATTEMPTS):
        reply = fn(*args)
        if reply == "OK":
            return reply
        print(f"warning: LED board rejected {label} ({reply!r}), retry {attempt + 1}/{_EXTRA_RETRY_ATTEMPTS}")
    return reply


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


def preview_target(ser, boundary, target_point):
    """One-time, human-facing cue: blinks the target blue a few slow on/off
    cycles, then leaves it off. Called once, right before the real LDR-
    driven run starts, so someone watching the board can see where the
    target is before the run begins hiding it (see show_target=False on
    that run's LedGridDisplay16x16) -- it never affects what the policy
    itself observes, since this happens before env._proximity is patched
    and before run_simulation starts stepping the episode.

    Reuses set_episode_layer's own dim-blue target color, toggling it
    against the sentinel used elsewhere to hide the target (NO_TARGET) --
    the same on-board mechanism LedGridDisplay16x16 uses, just driven
    manually here at a much slower, human-visible cadence. Cosmetic only
    (like LedGridDisplay16x16.update()): a persistent failure warns and
    moves on rather than aborting the run over a preview blink."""
    for cycle in range(_TARGET_PREVIEW_CYCLES):
        reply = _retry_send(set_episode_layer, ser, boundary, target_point, label="episode layer (target preview on)")
        if reply != "OK":
            print(f"warning: giving up on target preview on-blink {cycle + 1} after retries ({reply!r})")
        time.sleep(_TARGET_PREVIEW_ON_SECONDS)

        reply = _retry_send(set_episode_layer, ser, boundary, NO_TARGET, label="episode layer (target preview off)")
        if reply != "OK":
            print(f"warning: giving up on target preview off-blink {cycle + 1} after retries ({reply!r})")
        time.sleep(_TARGET_PREVIEW_OFF_SECONDS)


def show_thinking(ser, env, action_probs):
    """Simulation-only visual aid: briefly lights the 4 candidate next cells
    (one per action) a shade of yellow scaled by that action's policy
    probability, right before the move actually happens -- so a viewer can
    see what the agent is "thinking" before it commits to a direction.

    Wired in as run_simulation's on_think callback for the perfect-world
    episode only (see main()) -- the real LDR-driven episode never calls
    this, matching how it already hides the target: both keep the board
    from giving away information the agent itself can't sense.

    `env` still reflects the pre-move state (on_think fires before
    perform_action), so agent_local + each action's delta gives the
    candidate cell it would move to. A candidate outside the board is
    simply not lit (ledBoardSetThinkingLayer filters to in-bounds points),
    which can happen for an illegal move at the subgrid edge.

    Clears itself automatically: the next set_dynamic_layer call (from
    LedGridDisplay16x16.update(), once this move actually completes) resets
    the board's retained thinking-layer state as a side effect, so nothing
    here needs to explicitly turn it back off."""
    points = []
    for action, (dx, dy) in ACTION_DELTAS.items():
        candidate_local = (env.agent_local[0] + dx, env.agent_local[1] + dy)
        cx, cy = env.local_to_global(*candidate_local)
        brightness = round(action_probs[action] * 255)
        points.append((cx, cy, brightness))

    reply = _retry_send(set_thinking_layer, ser, points, label="thinking layer")
    if reply != "OK":
        print(f"warning: giving up on thinking-layer preview after retries ({reply!r})")
    time.sleep(_THINKING_ON_SECONDS)


class LedGridDisplay16x16:
    """Mirrors env state onto the 16x16 WS2812B panel in global coordinates,
    and tracks a per-step LED count/current estimate for power_summary().
    See eval_demo_16-16.py's version of this class for the full rationale
    -- this copy always has `ser` connected (no dry_run).

    show_target=False sends an out-of-bounds sentinel instead of the real
    target coordinate -- ws2812b_matrix_driver.cpp's ledBoardSetEpisodeLayer
    sets hasTarget = inBounds(target), so anything outside [0, 16) leaves
    hasTarget false and the target LED simply never lights. Used for the
    real LDR-driven run, so the board doesn't visually give away what the
    agent is supposed to be sensing for itself."""

    def __init__(self, ser, trail_length, boundary_margin, step_delay_seconds=0.0, show_target=True):
        self.ser = ser
        self.trail = deque(maxlen=trail_length)
        self.boundary_margin = boundary_margin
        self.step_delay_seconds = step_delay_seconds
        self.show_target = show_target
        self._boundary_count = 0
        self.step_power_stats = []

    def update(self, env):
        # Cosmetic display state -- a persistent failure here means the
        # board shows a stale frame for a step, not a wrong RL observation
        # (unlike build_ldr_proximity_fn's own set_dynamic_layer call, which
        # feeds the policy and stays fatal-on-failure -- see its docstring).
        # Losing one frame's visual accuracy is a much better outcome than
        # losing an entire episode's progress to a transient hardware
        # glitch, so this warns and keeps going rather than raising.
        if env.steps == 0:
            boundary = compute_boundary_points(
                env.origin, env.subgrid_size, env.grid_size, self.boundary_margin
            )
            target_point = env.target_global if self.show_target else NO_TARGET
            reply = _retry_send(set_episode_layer, self.ser, boundary, target_point, label="episode layer")
            if reply != "OK":
                print(f"warning: giving up on episode layer after retries ({reply!r}) -- "
                      f"boundary/target may be stale this episode")
            self._boundary_count = len(boundary)
            self.trail.clear()
            self.step_power_stats = []

        agent_global = env.local_to_global(*env.agent_local)
        self.trail.append(agent_global)
        trail_points = (
            [] if env.terminated else [p for p in list(self.trail)[:-1] if p != agent_global]
        )

        reply = _retry_send(set_dynamic_layer, self.ser, agent_global, trail_points, label="dynamic layer")
        if reply != "OK":
            print(f"warning: giving up on dynamic layer after retries ({reply!r}) -- "
                  f"agent/trail display may be one frame stale")
        self.step_power_stats.append(
            frame_power_stats(self._boundary_count, len(trail_points),
                               has_target=self.show_target, has_agent=True)
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
    called from inside GridEnvironment.perform_action.

    Unlike LedGridDisplay16x16.update()'s cosmetic display calls, a
    persistent failure here is left fatal (raises rather than warns and
    continues): the LED not actually reaching `pos` means the LDR reading
    that follows would reflect the wrong position, feeding the policy a
    silently wrong observation -- worse than a loud crash."""
    def ldr_proximity(pos):
        global_pos = env.local_to_global(*pos)
        reply = _retry_send(set_dynamic_layer, ser, global_pos, [], label="dynamic layer (proximity measurement)")
        if reply != "OK":
            raise RuntimeError(f"LED board rejected dynamic layer after retries: {reply!r}")
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

    # --- 1. Perfect-world run first: same seed, noiseless proximity, shown
    # on the board with the target visibly lit (nothing to hide here). ---
    with temporary_seed(cfg.seed):
        dummy_env = GridEnvironment(**env_kwargs)

    dummy_display = LedGridDisplay16x16(
        ser, trail_length=dummy_env.history_length, boundary_margin=cfg.boundary_margin,
        step_delay_seconds=cfg.step_delay_seconds, show_target=True,
    )
    # on_think=show_thinking: simulation-only -- never passed to the real
    # LDR-driven run below, which must not reveal anything beyond its own
    # sensor reading.
    dummy_trajectory = run_simulation(
        env=dummy_env, engine="mlp_network", network=network,
        on_step=dummy_display.update, on_think=lambda env, probs: show_thinking(ser, env, probs),
    )

    # --- 2. Real run second: same seed (so the same subgrid/start/target),
    # proximity comes from the LDR, target deliberately left unlit. ---
    with temporary_seed(cfg.seed):
        env = GridEnvironment(**env_kwargs)

    reply = clear(ser)
    if reply != "OK":
        raise RuntimeError(f"LED board rejected clear: {reply!r}")

    # Draw the boundary with no target/agent yet, then take the single
    # ambient-baseline reading this whole game will be measured against.
    boundary = compute_boundary_points(env.origin, env.subgrid_size, env.grid_size, cfg.boundary_margin)
    reply = set_episode_layer(ser, boundary, NO_TARGET)
    if reply != "OK":
        raise RuntimeError(f"LED board rejected episode layer: {reply!r}")
    time.sleep(cfg.ldr_linger_seconds)
    baseline = read_ldr(ser)
    accelerator.print(f"LDR baseline for this game: {baseline}")

    # Briefly show a human where the target is before the run starts hiding
    # it -- purely cosmetic, doesn't touch the LDR or the policy.
    preview_target(ser, boundary, env.target_global)

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

    # ldr_proximity's own settle wait (cfg.ldr_linger_seconds) already
    # pauses every step, but it must stay fixed at whatever value
    # eval_ldr_sweep.py --calibrate used -- the calibration curve is only a
    # valid reference for readings that settle the same duration, so it
    # can't be slowed down just to make the demo read better. Instead, top
    # up the *visible* per-step pause with however much of
    # cfg.step_delay_seconds that settle wait doesn't already cover, so a
    # viewer sees the same overall pace here as in the perfect-world run
    # above -- only the proximity source differs, not the rhythm of play.
    real_step_delay_seconds = max(0.0, cfg.step_delay_seconds - cfg.ldr_linger_seconds)
    real_display = LedGridDisplay16x16(
        ser, trail_length=env.history_length, boundary_margin=cfg.boundary_margin,
        step_delay_seconds=real_step_delay_seconds, show_target=False,
    )
    real_trajectory = run_simulation(env=env, engine="mlp_network", network=network, on_step=real_display.update)

    ser.close()

    # --- Both episodes are done -- report both together. ---
    accelerator.print(f"Checkpoint: {checkpoint_dir}")
    dummy_steps = print_outcome(accelerator, "perfect-world run", dummy_trajectory, dummy_env.terminated)
    print_power_summary(accelerator, dummy_display)
    real_steps = print_outcome(accelerator, "LDR-feedback run", real_trajectory, env.terminated)
    print_power_summary(accelerator, real_display)
    accelerator.print(f"LDR feedback cost {real_steps - dummy_steps} extra step(s) vs. the noiseless optimum")


if __name__ == "__main__":
    main()
