"""RL grid-world environment simulator.

A 16x16 global grid has a fixed target at (8, 8). Each problem instance
picks a random 8x8 subgrid (guaranteed to contain the target) and a random
start point inside it. The agent moves left/right/up/down within the
subgrid and is scored by how close it is to the target.
"""

import random
from collections import deque, namedtuple
from contextlib import contextmanager

GRID_SIZE = 16
SUBGRID_SIZE = 8
TARGET_GLOBAL = (GRID_SIZE // 2, GRID_SIZE // 2)  # (8, 8)
SCORE_RADIUS = 2
MAX_STEPS = 100
HISTORY_LENGTH = 4
STEP_PENALTY = 0.01   # reward cost for landing on an "empty" (proximity-0) cell
WALL_PENALTY = 0.05   # reward cost for an illegal move (walking into the subgrid edge)
SUCCESS_BONUS = 1.0   # terminal reward for reaching the target

ACTIONS = ("left", "right", "up", "down")
ACTION_DELTAS = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, 1),
    "down": (0, -1),
}

State = namedtuple("State", ["x", "y", "score"])


def proximity_score(distance, score_radius):
    """Persistent "warmth" for being `distance` cells (Manhattan) from the
    target: 1.0 at distance 0, decreasing linearly to 0 at the edge of
    score_radius, 0 beyond it. Pulled out as a module-level function (rather
    than staying private to GridEnvironment) so anything that needs to
    reproduce the trained policy's exact discrete score levels from a
    distance -- e.g. eval_ldr_sweep.py's LDR-to-proximity calibration --
    can import this instead of re-deriving/duplicating the formula."""
    if distance == 0:
        return 1.0
    if distance <= score_radius:
        return (score_radius + 1 - distance) / (score_radius + 1)
    return 0.0


def set_global_seed(seed):
    """Seed the shared random stream every GridEnvironment draws from."""
    random.seed(seed)


@contextmanager
def temporary_seed(seed):
    """Reseed the global random stream for the duration of the `with` block,
    then restore the prior state. Lets an eval pass reproduce a specific
    instance without disturbing an ongoing training rollout stream, and lets
    two different policies replay the same episode by wrapping each in
    `with temporary_seed(episode_seed):`.
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def derive_episode_seeds(seed, n_episodes):
    """n_episodes reproducible per-episode seeds, deterministic in `seed`."""
    rng = random.Random(seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n_episodes)]


class GridEnvironment:
    """The grid-world task. Call reset() to generate a new random problem
    instance for each episode. Randomness comes from the shared global
    stream -- see set_global_seed."""

    def __init__(self, grid_size=GRID_SIZE, subgrid_size=SUBGRID_SIZE,
                 score_radius=SCORE_RADIUS, max_steps=MAX_STEPS,
                 history_length=HISTORY_LENGTH, step_penalty=STEP_PENALTY,
                 wall_penalty=WALL_PENALTY, success_bonus=SUCCESS_BONUS):
        self.grid_size = grid_size
        self.subgrid_size = subgrid_size
        self.score_radius = score_radius
        self.max_steps = max_steps
        self.history_length = history_length
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.success_bonus = success_bonus
        self.target_global = (grid_size // 2, grid_size // 2)

        self.reset()

    def reset(self):
        """Start a new episode: fresh random subgrid placement and start
        point. Returns the initial observation."""
        # Subgrid origin (bottom-left corner, global coords), constrained to
        # stay in-bounds and to keep the target inside the subgrid.
        origin_min_x = max(0, self.target_global[0] - (self.subgrid_size - 1))
        origin_max_x = min(self.grid_size - self.subgrid_size, self.target_global[0])
        origin_min_y = max(0, self.target_global[1] - (self.subgrid_size - 1))
        origin_max_y = min(self.grid_size - self.subgrid_size, self.target_global[1])

        origin_x = random.randint(origin_min_x, origin_max_x)
        origin_y = random.randint(origin_min_y, origin_max_y)
        self.origin = (origin_x, origin_y)

        self.target_local = (
            self.target_global[0] - origin_x,
            self.target_global[1] - origin_y,
        )

        while True:
            start = (random.randint(0, self.subgrid_size - 1), random.randint(0, self.subgrid_size - 1))
            if start != self.target_local:
                break
        self.start_local = start
        self.agent_local = start

        self.steps = 0
        self.terminated = False
        self.truncated = False

        self._current_score = self._proximity(self.agent_local)
        self._current_reward = 0.0

        # Observation = sliding window of the last `history_length`
        # (x, y, score) readings, oldest first; padded with the initial
        # reading before enough real steps have happened.
        initial_reading = State(x=self.agent_local[0], y=self.agent_local[1],
                                 score=self._current_score)
        self._history = deque([initial_reading] * self.history_length,
                               maxlen=self.history_length)

        return self.get_state()

    def local_to_global(self, x, y):
        return (x + self.origin[0], y + self.origin[1])

    def _distance_to_target(self, pos):
        # Manhattan distance: every legal (orthogonal) move changes it by
        # exactly 1, so a strictly-warming move always exists.
        dx = abs(pos[0] - self.target_local[0])
        dy = abs(pos[1] - self.target_local[1])
        return dx + dy

    def _proximity(self, pos):
        """Persistent "warmth" for being at `pos` -- see proximity_score."""
        return proximity_score(self._distance_to_target(pos), self.score_radius)

    def get_state(self):
        """The policy's observation: last `history_length` (x, y, score)
        readings, oldest first. Contains no target information."""
        return tuple(self._history)

    @property
    def score(self):
        """Persistent proximity "warmth" at the current position (see
        _proximity). Not the training reward -- see `reward`."""
        return self._current_score

    @property
    def reward(self):
        """Training reward for the most recent step (0 before any action).
        See perform_action for how it's computed."""
        return self._current_reward

    def is_done(self):
        return self.agent_local == self.target_local

    def perform_action(self, action):
        if action not in ACTION_DELTAS:
            raise ValueError(f"Unknown action: {action!r}. Must be one of {ACTIONS}")

        prev_proximity = self._current_score

        dx, dy = ACTION_DELTAS[action]
        new_x = self.agent_local[0] + dx
        new_y = self.agent_local[1] + dy

        legal_move = 0 <= new_x < self.subgrid_size and 0 <= new_y < self.subgrid_size
        if legal_move:
            self.agent_local = (new_x, new_y)

        self.steps += 1
        self._current_score = self._proximity(self.agent_local)
        self._history.append(State(x=self.agent_local[0], y=self.agent_local[1],
                                    score=self._current_score))

        self.terminated = self.is_done()
        self.truncated = (not self.terminated) and (self.steps >= self.max_steps)

        # Reward: success_bonus on reaching the target; otherwise the CHANGE
        # in proximity (potential-based shaping, so circling near the target
        # nets ~0), minus step_penalty if the new cell is outside the radius
        # and minus wall_penalty if the move was illegal. terminated has no
        # future (GAE bootstraps V=0); truncated is an artificial cutoff
        # bootstrapped from the critic instead (see rollout.py).
        if self.terminated:
            self._current_reward = self.success_bonus
        else:
            self._current_reward = self._current_score - prev_proximity
            if self._current_score == 0.0:
                self._current_reward -= self.step_penalty
            if not legal_move:
                self._current_reward -= self.wall_penalty

        return self.get_state()

    def render(self):
        """Print the current 8x8 subgrid, bottom-left-origin (0,0) at the
        bottom-left corner."""
        lines = []
        for y in range(self.subgrid_size - 1, -1, -1):
            row = []
            for x in range(self.subgrid_size):
                pos = (x, y)
                if pos == self.agent_local and pos == self.target_local:
                    row.append("X")  # agent reached target
                elif pos == self.agent_local:
                    row.append("A")
                elif pos == self.target_local:
                    row.append("T")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        print("\n".join(lines))


def run_simulation(env, engine="manual", network=None, on_step=None, on_think=None):
    """Run one full episode and return the trajectory.

    `env` must already be constructed/configured by the caller. trajectory
    is a list of dicts: {"step", "action", "reward", "observation",
    "terminated", "truncated", "log_prob", "value"}. The first entry has
    action=None/reward=None and holds the initial observation.

    Each entry spans two timesteps: "action"/"log_prob"/"value" describe the
    state the agent acted FROM; "observation"/"reward" describe the result.
    So values[t] = entry[t+1]["value"] = V(s_t), while the observation for
    step t is entry[t]["observation"] = s_t (see rollout.py).

    "terminated" means the agent reached the target (V(final)=0 for GAE);
    "truncated" means max_steps was hit (V(final) must be bootstrapped from
    the critic instead -- run_simulation does not do this itself).

    "log_prob"/"value" are set only when engine="mlp_network" (needed for
    the PPO ratio and GAE advantages later).

    engine: "manual" prompts for input at each step; "mlp_network" drives
    the env via `network.act(observation) -> (action_index, log_prob, value,
    entropy)`; "random" picks uniformly via the shared global random stream.

    on_step: optional callback invoked with `env` once for the initial state
    and once after every action -- lets a caller mirror the episode
    somewhere else (e.g. eval_demo_8-8.py driving the physical LED matrix)
    without run_simulation itself knowing anything about that destination.

    on_think: optional callback, only ever invoked when engine="mlp_network",
    called as `on_think(env, action_probs)` right before each action is
    taken -- `env` still reflects the state the agent is acting FROM, and
    action_probs is a dict from simulator.ACTIONS to that action's policy
    probability for this observation (via `network.action_probs`). Lets a
    caller visualize the policy's confidence just before a move happens
    (e.g. eval_demo_16-16-ldr-feedback.py's simulation-only preview of the
    board's "thinking") without influencing which action actually gets
    taken -- that's always network.act()'s own choice, computed separately.
    """
    trajectory = [{
        "step": 0, "action": None, "reward": None, "observation": env.get_state(),
        "terminated": env.terminated, "truncated": env.truncated,
        "log_prob": None, "value": None,
    }]

    if on_step is not None:
        on_step(env)

    if engine == "manual":
        print("Initial board:")
        env.render()
        print(f"State: {trajectory[0]['observation'][-1]}\n")

    while not (env.terminated or env.truncated):
        log_prob = None
        value = None
        if engine == "manual":
            action = _prompt_for_action()
        elif engine == "mlp_network":
            if network is None:
                raise ValueError("engine='mlp_network' requires a `network` argument")
            observation = env.get_state()
            if on_think is not None:
                on_think(env, network.action_probs(observation))
            action_index, log_prob, value, _entropy = network.act(observation)
            action = ACTIONS[action_index]
        elif engine == "random":
            action = random.choice(ACTIONS)
        else:
            raise ValueError(f"Unknown engine: {engine!r}")

        observation = env.perform_action(action)
        trajectory.append({
            "step": env.steps, "action": action, "reward": env.reward, "observation": observation,
            "terminated": env.terminated, "truncated": env.truncated,
            "log_prob": log_prob, "value": value,
        })

        if on_step is not None:
            on_step(env)

        if engine == "manual":
            print(f"\nAfter action '{action}':")
            env.render()
            print(f"State: {observation[-1]}")
            if env.terminated:
                print("\nTarget reached!")
            elif env.truncated:
                print("\nStep limit reached.")

    return trajectory


def _prompt_for_action():
    while True:
        action = input(f"Enter action {ACTIONS}: ").strip().lower()
        if action in ACTION_DELTAS:
            return action
        print(f"Invalid action {action!r}. Must be one of {ACTIONS}.")
