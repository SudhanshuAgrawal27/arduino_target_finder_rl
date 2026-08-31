# Blind Target Search

*Learning to find a hidden target with reinforcement learning, then evaluating the trained policy on an Arduino with a light sensor.*

## The problem

A target sits at the center of a fixed 16×16 grid. Each episode drops an agent somewhere inside a random 8×8 window of that grid — a window that always contains the target, but the agent is never told where. All it gets back each step is a "warmth" reading: strong near the target, fading to nothing a couple of cells out, and completely silent beyond that.

It's a blind search, like an ant following a faint scent trail to a sugar cube — no map, no coordinates, just a signal that gets stronger or weaker depending on which way it moves.

<p align="center"><img src="assets/grid-image.png" width="560" alt="Grid diagram: target with a two-ring proximity halo, a dashed start square, elsewhere blank"></p>

We train a PPO policy to solve this purely in simulation, then evaluate the trained policy on physical hardware: an LED marks the agent's position on a grid, and a photoresistor (LDR) pointed at the target's LED stands in for the proximity signal — the policy reacts to a live light reading instead of a computed distance.

<p align="center"><img src="assets/system-diagram.png" width="720" alt="System diagram: host machine running the policy and simulator, connected over USB serial to an Arduino driving an LED matrix and reading an LDR"></p>

The policy and the environment both live on a host machine in Python; the Arduino only drives the LED display and reports the LDR reading.

## Docker Setup

Training and evaluation run inside a CUDA-enabled container so the environment (PyTorch, Hydra, accelerate, the Arduino CLI for firmware verification) is reproducible across machines.

**Key files:** [`docker/Dockerfile`](docker/Dockerfile), [`docker/build_docker.sh`](docker/build_docker.sh), [`docker/run_docker.sh`](docker/run_docker.sh), [`docker/docker_setup.md`](docker/docker_setup.md)

```
./docker/build_docker.sh   # build the image once
./docker/run_docker.sh     # start the container, mounting the repo at /workspace
```

## Arduino Setup

One shared sketch drives whichever LED board is wired up; a compile-time flag picks the driver.

**Key files:** [`arduino/led_board_controller/firmware/led_serial_listener/`](arduino/led_board_controller/firmware/led_serial_listener/) (`led_serial_listener.ino`, `board_config.h`, `max7219_matrix_driver.cpp`, `ws2812b_matrix_driver.cpp`), [`arduino/README.md`](arduino/README.md)

Set the active driver in `board_config.h`:
```c
#define ACTIVE_BOARD BOARD_WS2812B_MATRIX   // or BOARD_MAX7219_MATRIX
```
Then, in the Arduino IDE:
1. Open `led_serial_listener.ino` (the IDE loads `board_config.h` and the driver `.cpp` files alongside it automatically).
2. Under **Tools → Board**, select your board (Arduino Uno).
3. Under **Tools → Port**, select the port the board is connected to.
4. Click **Upload**.
5. If you open the Serial Monitor to watch the board directly, set its baud rate to `115200` to match the sketch.

If you're passing an Arduino through from WSL2, [`docker/reconnect_usb.sh`](docker/reconnect_usb.sh) re-attaches the board's USB-serial device to a running container after a driver reinstall or unplug/replug, without recreating the container.

## Circuit Setup

*Coming soon.*

## RL Framework

**Environment** — [`simulator.py`](simulator.py)'s `GridEnvironment`: places the subgrid/target/start, steps the agent (`left`/`right`/`up`/`down`), and reports back a `(x, y, score)` reading. `score` is a pure function of position — 1.0 at the target, decaying linearly to 0 at `score_radius` (default 2) — so a short history of readings lets the agent tell whether its last move helped.

**Network** — [`network.py`](network.py)'s `ActorCritic`, a shared-trunk MLP:
```
(x, y, score) × window_length  →  Linear→ReLU × num_layers  ─┬─ policy head → 4 action logits
                                                               └─ value head  → 1 scalar
```

**Training signal** — [`ppo.py`](ppo.py) implements clipped-surrogate PPO with GAE advantages (computed per episode in [`rollout.py`](rollout.py)):

```
δ_t = r_t + γ·V(s_t+1) − V(s_t)                A_t = δ_t + (γλ)·δ_t+1 + (γλ)²·δ_t+2 + ...

r_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t)        L_CLIP(θ) = E_t[ min(r_t·A_t, clip(r_t, 1−ε, 1+ε)·A_t) ]

L(θ) = −L_CLIP(θ) + c1·(V(s_t) − R_t)² − c2·entropy(π_θ)
```

`V(s_final) = 0` when the episode actually `terminated` (target reached); it's bootstrapped from the critic when it was `truncated` by the step cap instead — this is why the environment tracks the two separately.

Try it by hand, or run the test suite:
```
python3 play_manual.py            # play one episode yourself from the terminal
pytest test_simulator.py test_network.py test_eval_lib.py -v
```

## Training

**Key files:** [`train.py`](train.py), [`conf/config.yaml`](conf/config.yaml) (fast smoke test), [`conf/config_train.yaml`](conf/config_train.yaml) (full 150-epoch run), [`run_ablation.py`](run_ablation.py)

One epoch = collect `episodes_per_epoch` episodes → GAE → a few epochs of minibatch PPO updates → a deterministic eval pass → checkpoint. Everything is a [Hydra](https://hydra.cc) config field, overridable on the command line.

Logging goes to [Weights & Biases](https://wandb.ai) — run `wandb login` once, then:
```
python3 train.py                                    # fast smoke test (1 epoch)
python3 train.py --config-name config_train         # full run, 3 seeds by default (cfg.seeds)
python3 train.py training.num_epochs=5 ppo.learning_rate=1e-4   # override any field
accelerate launch --config_file conf/accelerate.yaml train.py --config-name config_train
```
Set `conf/wandb_workspace_url.txt` to your own saved multi-run workspace view to have `train.py` print a link to it alongside each run's URL.

Checkpoints land in `trained_models/<run_name>_seed<seed>/epoch_N/`, one directory per seed, each with its own resolved `config.yaml` so an eval script can reconstruct the exact architecture later. [`run_ablation.py`](run_ablation.py) sweeps the `conf/config_train_h*_l*_hist*.yaml` architecture configs across `cfg.seeds`, skipping any checkpoint that already exists:
```
python3 run_ablation.py --parallel 4
```

## Evaluation

**Key files:** [`eval_random.py`](eval_random.py), [`build_eval_fixed_dataset.py`](build_eval_fixed_dataset.py), [`eval_fixed_dataset.py`](eval_fixed_dataset.py)

```
python3 eval_random.py checkpoint_dir=trained_models/<run>/epoch_150 episodes=200 seed=7
python3 eval_fixed_dataset.py checkpoint_dir=trained_models/<run>/epoch_150
```
`eval_random.py` draws fresh random episodes each run. `eval_fixed_dataset.py` instead replays the same 100 committed problem instances every time (`eval_fixed_dataset.json`, stratified by distance and by whether the target sits near the subgrid's edge), so two checkpoints can be compared on identical problems and broken down by category rather than just an aggregate success rate.

## Evaluation with Arduino

**Key files:** [`eval_ldr_sweep.py`](eval_ldr_sweep.py), [`eval_demo_16-16-ldr-feedback.py`](eval_demo_16-16-ldr-feedback.py), [`arduino/led_board_controller/led_board_client.py`](arduino/led_board_controller/led_board_client.py)

This is the closed-loop version of evaluation: the policy's proximity reading comes from a real LDR pointed at the target LED instead of the simulator's distance formula. Two steps:

**1. Calibrate once, offline.** A probe LED sweeps the 8×8 window around the target; the LDR's brightness delta from a single ambient baseline is recorded at every cell and pooled by Manhattan distance into the same four levels the trained policy expects.
```
python3 eval_ldr_sweep.py --calibrate
```
<p align="center"><img src="ldr_calibration_plot.png" width="420" alt="Calibration plot: LDR reading delta as a function of distance from the target"></p>

**2. Run the live episode.** Two passes with the same seed: a noiseless "perfect world" reference pass (target visibly lit), then the real LDR-driven pass (target deliberately unlit, so the board never gives away what the sensor has to find on its own).
```
python3 eval_demo_16-16-ldr-feedback.py checkpoint_dir=trained_models/<run>/epoch_150 seed=7
```
Reports steps/return/success for both passes plus the step-count gap between them.
