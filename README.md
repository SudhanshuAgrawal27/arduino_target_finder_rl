# Grid Search RL

An agent learns to find an unknown target on a grid, guided only by a coarse proximity signal — no target coordinates are ever revealed to it.

## Task setup

- A fixed **16x16 global grid**, bottom-left indexed at `(0, 0)`. The **target** is fixed at the grid's center, `(8, 8)`.
- Each **problem instance** picks a random **8x8 subgrid** of the global grid, constrained so it always contains the global target (with correct edge handling — the subgrid's origin is sampled from the range that keeps both the subgrid in-bounds and the target inside it).
- Within that subgrid, a random **start** point is chosen for the agent (never exactly on the target).
- Both the target's position *within the subgrid* (its local coordinates) and the start point are only known internally to the environment — **the agent never observes the target's location**, in either global or local coordinates. The task is a search problem: find the target using only movement and the feedback described below.

## Environment

Implemented in [`simulator.py`](simulator.py) as `GridEnvironment`.

- `GridEnvironment(grid_size=16, subgrid_size=8, score_radius=2, max_steps=100, history_length=4, seed=None)` — constructs the environment and generates the first problem instance.
- `reset(seed=None)` — starts a new episode: fresh random subgrid placement and start point, drawn from the environment's own random stream (so repeated calls give different instances). Pass `seed=` to reproduce one specific instance.
- `perform_action(action)` — `action` is one of `"left" | "right" | "up" | "down"`. Moves the agent one cell if legal; moves that would leave the subgrid are silently no-ops. Returns the new observation (see below).
- `get_state()` — returns the current observation.
- `env.score` — the scalar reward for the most recent step (see [Reward](#reward)).
- `env.terminated` / `env.truncated` — episode-end flags (see [Episode termination](#episode-termination)).
- `render()` — prints the current 8x8 subgrid as text (`A` = agent, `T` = target, `X` = agent at target).

### Observation

The agent's observation is a sliding window of the **last `history_length` (default 4) `(x, y, score)` readings**, oldest first — local coordinates only, no target information. Before enough real steps have happened, the window is padded by repeating the initial reading.

The window exists because the agent has no other way to tell whether its last move helped: a single instantaneous `(x, y, score)` reading doesn't reveal a trend, but a short history lets the agent (or a recurrent policy layered on top of it) infer "am I getting warmer."

### Reward

Two distinct signals, kept separate on purpose — `score` (what the agent observes) and `reward` (what training optimizes):

**`score`** — a persistent, distance-graded "warmth" reading, based on Manhattan (L1) distance to the target (chosen over Chebyshev so every legal orthogonal move changes it by exactly 1, guaranteeing a strictly-warming move always exists): `1.0` at the target, decreasing linearly to `0` at the edge of `score_radius` (default 2: distance 1 → 0.667, distance 2 → 0.333), and `0` beyond it. It's a pure function of position — the same cell always reads the same value — so the observation history carries a followable gradient the agent can hill-climb.

**`reward`** — the PPO training signal for the most recent step:

| Condition | Reward |
|---|---|
| Reached the target | `success_bonus` (default 1.0) |
| Otherwise | `score(new) − score(old)` (potential-based shaping — positive for closing in, negative for backing off) |
| ...and the new cell is outside `score_radius` (`score == 0`) | additionally `− step_penalty` (default 0.01) |
| ...and the action was illegal (walked into the subgrid edge, a no-op) | additionally `− wall_penalty` (default 0.05) |

Because the shaping term is a difference of a position-only potential, any loop or lingering near the target telescopes to ~0 — the agent can't farm reward by circling in the radius instead of pushing on to the target. `step_penalty` nudges it out of empty space; `wall_penalty` discourages wasting a move bumping a wall (and stacks with `step_penalty` if that wasted move also leaves it outside the radius).

### Episode termination

Two distinct end conditions, both tracked separately because they matter differently for training:

- **`terminated`** — the agent actually reached the target. A true end of episode; there is no future to bootstrap.
- **`truncated`** — `max_steps` (default 100) was hit without reaching the target. An artificial cutoff; the episode didn't actually conclude, so a value estimate for the final state (rather than 0) should be used when bootstrapping returns.

### Manual play

[`play_manual.py`](play_manual.py) runs one episode with a human typing actions at the terminal, printing the board after every move:

```
python3 play_manual.py
```

### Tests

[`test_simulator.py`](test_simulator.py) covers: edge conditions of subgrid/target placement across many random seeds, movement boundary clamping in all four directions, the one-time radius bonus, the terminated/truncated distinction, `reset()` behavior (fresh instance per call, reproducibility with a seed), the observation history window (padding and sliding), and full `run_simulation` episodes (both reaching the target and hitting the step cap).

```
pytest test_simulator.py -v
```

## Network

[`network.py`](network.py) — `ActorCritic`, a shared-trunk MLP over a (possibly trimmed) observation window:

```
input: flattened (x, y, score) × window_length = 12 floats (window_length=4), x/y normalized to [0,1]
  → Linear(12→64) → ReLU → Linear(64→64) → ReLU → ... (num_layers hidden blocks)
    ├─ policy head: Linear(64→4)  (action logits)
    └─ value head:  Linear(64→1)  (scalar state value)
```

`window_length` (`network.window_length`, e.g. via `python3 train.py network.window_length=2`) is independent of the environment's `history_length`: the environment can produce a longer history than the network actually consumes, in which case only the most recent `window_length` readings are fed in (see `obs_to_tensor`). Must be `<= env.history_length`; `train.py` checks this at startup. `num_layers` (`network.num_layers`, default 2) controls how many hidden `Linear(hidden_dim→hidden_dim) → ReLU` blocks the trunk has (the first block is `Linear(window_length×3→hidden_dim)`), for trading network depth against width. All three of `hidden_dim`, `window_length`, and `num_layers` are plain Hydra config fields, overridable from the command line same as any other (e.g. `network.hidden_dim=32 network.num_layers=3`).

Weights use the standard PPO init: orthogonal, gain √2 on hidden layers, gain 1 on the value head, and a small gain (0.01) on the policy head so the initial policy starts close to uniform. `act()` samples an action (used during rollout collection); `act_deterministic()` picks the argmax action (used for eval); `get_value()` returns just the critic's estimate (used to bootstrap GAE at a truncated episode's end); `evaluate_actions()` recomputes log-probs/values/entropy under the current parameters with gradients (used during the PPO update).

## Training

`run_simulation(env, engine, network)` in `simulator.py` supports `engine="manual"` (interactive) and `engine="mlp_network"` (drives the env with `network.act(observation)` each step, storing `log_prob`/`value` per step for PPO). Training and eval are separate scripts, both driven by [Hydra](https://hydra.cc) configs in `conf/`:

- **[`train.py`](train.py)** — one epoch = collect `training.episodes_per_epoch` episodes serially (one `GridEnvironment` at a time, reused across episodes only in the sense that a fresh one is constructed per episode — see [`rollout.py`](rollout.py)), compute GAE per episode, flatten into one batch, run `ppo.update_epochs` passes of minibatch gradient descent ([`ppo.py`](ppo.py)), run a quick deterministic eval pass ([`eval_lib.py`](eval_lib.py)), then checkpoint. Uses `accelerate` for device placement, seeding (`accelerate.utils.set_seed`), and checkpointing (`accelerator.save_state`).
- **[`eval_random.py`](eval_random.py)** — standalone: loads a checkpoint's sibling `config.yaml` to reconstruct the exact network/env architecture, loads the weights via `accelerator.load_state`, and runs deterministic evaluation episodes independently of training, with its own seed.

Run with defaults (1 epoch × 1000 episodes):
```
python3 train.py
```
Override any field from the command line:
```
python3 train.py training.num_epochs=5 ppo.learning_rate=1e-4
```
Evaluate a saved checkpoint:
```
python3 eval_random.py checkpoint_dir=trained_models/2026-07-06_14-30-05/epoch_1 episodes=200 seed=7
```

### Fixed evaluation dataset

`eval_random.py` draws fresh random episodes each run (reproducible via `seed`, but the *set* of problem instances isn't fixed across different `episodes`/`seed` choices). For comparing checkpoints against each other on identical, difficulty-labeled problems, use the fixed dataset instead:

- [`build_eval_fixed_dataset.py`](build_eval_fixed_dataset.py) generates `eval_fixed_dataset.json`: 100 problem instances stratified across 6 categories crossing **distance tier** (short 1–4 / medium 5–8 / long 9–14 Manhattan steps from start to target) and **target locality** (central — target's `score_radius` warmth halo is fully unclipped by the subgrid edge — vs boundary — halo clipped on at least one side). Each entry is just a seed plus its recorded category/geometry; regenerate with `python3 build_eval_fixed_dataset.py` (deterministic given `GENERATION_SEED`, already committed so you normally don't need to).
- [`eval_fixed_dataset.py`](eval_fixed_dataset.py) loads a checkpoint and replays every instance in the dataset, reporting overall stats plus a per-category breakdown:
  ```
  python3 eval_fixed_dataset.py checkpoint_dir=trained_models/2026-07-06_14-30-05/epoch_1
  ```
  This is what answers "is this checkpoint worse on long-distance or boundary-target problems specifically," rather than just an aggregate success rate.

### LED matrix demo

[`eval_demo_8-8.py`](eval_demo_8-8.py) runs one episode from a checkpoint and mirrors the target/agent/trail live onto a physical 8x8 LED matrix, via `simulator.run_simulation`'s `on_step` callback:
```
python3 eval_demo_8-8.py checkpoint_dir=... seed=7
```

[`eval_demo_16-16.py`](eval_demo_16-16.py) is the same idea on the 16x16 WS2812B panel, but in **global** board coordinates (`env.local_to_global`) so the whole grid is visible, not just the current subgrid:
```
python3 eval_demo_16-16.py checkpoint_dir=... seed=7
```
Shows, simultaneously: the current subgrid's boundary (dim green, blinking ~33Hz), the target (dim blue, blinking ~33Hz), the agent's trail (dim red, also blinking, and cleared once the agent actually reaches the target), and the agent itself (full red, always lit). Reports steps/return/success plus an estimated LED count / power draw at the end of the run (see [`arduino/led_board_controller/power_model.py`](arduino/led_board_controller/power_model.py)); pass `dry_run=true` to get that same report -- including the real episode's boundary/trail geometry -- without any LED board attached at all.

See [`arduino/README.md`](arduino/README.md) for the LED hardware/firmware side (including the serial protocol both demos speak) and the WSL2/Docker USB setup this needs.

### Accelerate config

Running `python3 train.py` directly is fine — `Accelerator()` auto-detects the machine (single process, GPU if present). For an explicit, version-controlled setup, [`conf/accelerate.yaml`](conf/accelerate.yaml) pins a single-process, CPU-only run and is passed via `accelerate launch`:
```
accelerate launch --config_file conf/accelerate.yaml train.py training.num_epochs=5
```
It's CPU-pinned on purpose: the network is a tiny 12→64→64 MLP, so the GPU buys essentially nothing and the per-step single-observation host↔device transfers during rollout are pure overhead — the real bottleneck (serial episode simulation) lives on CPU regardless. Flip `use_cpu: false` if the network is ever scaled up enough to be GPU-bound. Note that metrics are reproducible *within* a device but not bit-identical across CPU vs GPU (float ops differ).

### Output layout

```
trained_models/<run_name>_seed<seed>/   one dir per (run, seed); name = [timestamp_]<output.run_name>_seed<seed>
├── config.yaml                      fully-resolved Hydra config for this run, plus seed_used
├── epoch_1/                         accelerate checkpoint (model.safetensors, optimizer.bin, random_states)
├── epoch_2/
└── ...
```

### Seeds and multi-seed runs

Training is seeded per-run from `cfg.seeds` (a list, default `[42, 43, 44]`): `train.py` trains one model per seed, each into its own `..._seed<seed>` directory, so a single invocation produces a whole multi-seed ablation point. Override to one seed for parallel launching (`seeds=[43]`). `output.timestamp` (default `true`) prepends a timestamp; set it `false` for a deterministic, restart-skippable directory name.

PPO is high-variance across seeds — different seeds land in different behavioral basins — so a *single* run's success rate can under- or over-state an architecture by several points. Reporting ≥3 seeds is what makes an architecture comparison trustworthy.

**Init-seed isolation.** Orthogonal init consumes a *width/depth-dependent* number of `torch` draws, which would otherwise leave the downstream exploration-sampling and minibatch-shuffle RNG at an architecture-dependent offset — so "the same seed" wouldn't actually be a controlled comparison across architectures. `train.py` re-seeds `torch` with `torch.manual_seed(seed)` immediately after building the model, making that downstream stream depend only on `seed`: two architectures at the same seed then see identical exploration/shuffle noise, while different seeds still vary the whole run (init included). See the tests in [`test_network.py`](test_network.py).

### Running the architecture ablation

[`run_ablation.py`](run_ablation.py) trains every (architecture, seed) combination — the nine `conf/config_train_h*_l*_hist*.yaml` configs × seeds `{42, 43, 44}` — a fixed number at a time, skipping any whose `epoch_150` checkpoint already exists (safe to restart):

```
python3 run_ablation.py --dry-run          # list the jobs
python3 run_ablation.py --parallel 4       # 4 concurrent, 2 threads each (CPU-pinned)
```

New fixed-code runs land in `<arch>_seed<seed>/`; the earlier pre-init-fix runs are kept under the `_s<seed>` suffix so the two never collide.

### Reproducibility

`train_one_seed` calls `accelerate.utils.set_seed(seed)` (seeding `torch`/`numpy`/`random`) plus `simulator.set_global_seed(seed)` for the env's `random` stream, then applies the init-seed isolation above. Episode geometry, rollout collection, and minibatch shuffling are all reproducible from `seed`. `eval_random.py` takes its own `seed` for the same guarantee on evaluation runs, independent of whatever seed trained the checkpoint.

### Core PPO formulas

**Advantage (GAE)**, from the TD errors `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)`:

```
A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

`V(s_final) = 0` when the step was `terminated`; when `truncated`, `V(s_final)` is estimated from the critic instead — this is exactly why the environment tracks the two separately.

**Clipped surrogate policy objective**:

```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
L_CLIP(θ) = E_t[ min( r_t(θ)·A_t, clip(r_t(θ), 1−ε, 1+ε)·A_t ) ]
```

`r_t(θ)·A_t` is the importance-sampling-corrected surrogate for the standard policy gradient (`∇θ log π_θ(a_t|s_t)·A_t`), which lets the same rollout batch be reused for several epochs of gradient descent. The clip removes the incentive to push the policy ratio outside `[1−ε, 1+ε]` in either direction, bounding how far a single batch of data can move the policy — a cheap stand-in for TRPO's explicit trust-region constraint.

**Total loss**:

```
L(θ) = −L_CLIP(θ) + c1·(V_φ(s_t) − R_t)² − c2·entropy(π_θ(·|s_t))
```

### Open design questions for the training loop (not yet decided)

- **Search mechanism**: whether the 4-step observation window alone is enough for the policy to search effectively, or whether a recurrent policy (LSTM/GRU carrying hidden state across the full episode) should sit on top of it. `history_length` is a tunable constructor parameter either way.
- Whether to add a small per-step penalty to encourage shorter paths once the agent is already near the target.
- Whether potential-based reward shaping (`γ·Φ(s') − Φ(s)` with `Φ = -distance_to_target`) is worth adding as a denser reward alongside the current milestone-based score. The environment already knows `target_local` internally (it's just never exposed to the agent as an observation), so it could compute this shaping term without breaking the "no target info in the observation" rule — not implemented yet, just noting it's compatible if the sparse 0/0.5/1 signal proves too hard to learn from.
