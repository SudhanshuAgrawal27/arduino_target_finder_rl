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
STEP_PENALTY = 0.01   # per-step reward cost for landing on an "empty" (proximity-0) cell
WALL_PENALTY = 0.05   # reward cost for an illegal move (walking into the subgrid edge, a no-op)
SUCCESS_BONUS = 1.0   # terminal reward for reaching the target

ACTIONS = ("left", "right", "up", "down")
ACTION_DELTAS = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, 1),
    "down": (0, -1),
}

State = namedtuple("State", ["x", "y", "score"])


def set_global_seed(seed):
    """Seed the shared random stream used by every GridEnvironment.

    Call this once at the start of a training or evaluation script. Every
    environment constructed or reset() afterward draws from this single
    global stream, so the exact sequence of problem instances produced
    depends only on call order, not on any seed threaded through individual
    constructor/reset calls. Re-call with the same seed (e.g. at the start
    of an eval script) to reproduce a run exactly.
    """
    random.seed(seed)


@contextmanager
def temporary_seed(seed):
    """Reseed the shared global random stream to `seed` for the duration of
    the `with` block, then restore it exactly to whatever it was before --
    so a deterministic aside (e.g. a fixed-seed eval pass) doesn't perturb
    an ongoing training rollout stream's continuity.

    Also what makes two different policies (e.g. a trained model vs. a
    random baseline) play the *same* set of problem instances: wrap each
    episode in `with temporary_seed(episode_seed):` for both, and since
    each block starts from the same known seed, GridEnvironment() draws the
    same origin/target/start regardless of what either policy's actions
    consumed on a previous episode.
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def derive_episode_seeds(seed, n_episodes):
    """n_episodes reproducible per-episode seeds, deterministic in `seed`.

    Shared by run_eval and collect_rollouts' validation-batch path so that
    two different callers given the same (seed, n_episodes) -- e.g. an eval
    pass and a validation-loss pass -- wrap the same episode i in
    `temporary_seed(result[i])` and therefore play the exact same set of
    problem instances.
    """
    rng = random.Random(seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n_episodes)]


class GridEnvironment:
    """The grid-world task. Call reset() to generate a new random problem
    instance (fresh subgrid placement and start point) for each episode.
    Randomness comes from the shared global stream -- see set_global_seed."""

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
        """Start a new episode: a fresh random subgrid placement and start
        point, drawn from the shared global random stream. Returns the
        initial observation."""
        # Choose the subgrid's origin (its bottom-left corner in global
        # coordinates) such that the subgrid stays within the global grid
        # AND contains the target point.
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

        # Random start point, excluding the target itself.
        while True:
            start = (random.randint(0, self.subgrid_size - 1), random.randint(0, self.subgrid_size - 1))
            if start != self.target_local:
                break
        self.start_local = start
        self.agent_local = start

        self.steps = 0
        self.terminated = False
        self.truncated = False

        # `score` is a persistent, distance-graded proximity reading (see
        # _proximity): higher the closer the agent is to the target, the
        # same every time a given cell is occupied. This is what the agent
        # observes -- a "warmth" signal it can hill-climb to land exactly on
        # the target. `reward` (the training signal, see perform_action) is
        # kept separate from it so that a persistent observable warmth can
        # coexist with a farm-proof reward. Before any action the reward is 0.
        self._current_score = self._proximity(self.agent_local)
        self._current_reward = 0.0

        # Observation = a sliding window of the last `history_length`
        # (x, y, score) readings, oldest first. The agent has no target
        # info, so this recent trend is the only way it can tell whether its
        # last move helped or hurt. Padded with the initial reading before
        # enough real steps have happened.
        initial_reading = State(x=self.agent_local[0], y=self.agent_local[1],
                                 score=self._current_score)
        self._history = deque([initial_reading] * self.history_length,
                               maxlen=self.history_length)

        return self.get_state()

    def local_to_global(self, x, y):
        return (x + self.origin[0], y + self.origin[1])

    def _distance_to_target(self, pos):
        # Manhattan (L1) distance. Chosen over Chebyshev so that every legal
        # (orthogonal) move changes the distance by exactly 1: there are no
        # equidistant "plateaus" where moving toward the target leaves
        # proximity unchanged. That guarantees a strictly-warming move always
        # exists, so the greedy policy can follow the proximity gradient all
        # the way onto the exact target cell (Chebyshev left the 4
        # diagonally-adjacent cells with a zero-gradient landing step).
        dx = abs(pos[0] - self.target_local[0])
        dy = abs(pos[1] - self.target_local[1])
        return dx + dy

    def _proximity(self, pos):
        """Persistent, distance-graded "warmth" for being at `pos`: 1.0 on
        the target, linearly decreasing to 0 at the edge of score_radius,
        and 0 beyond it. For score_radius=2: dist 1 -> 0.667, dist 2 ->
        0.333. Pure function of position -- no side effects, and the same
        cell always reads the same value (unlike the old one-time bonus), so
        the agent's observation history carries a followable gradient toward
        the exact target cell."""
        d = self._distance_to_target(pos)
        if d == 0:
            return 1.0
        if d <= self.score_radius:
            return (self.score_radius + 1 - d) / (self.score_radius + 1)
        return 0.0

    def get_state(self):
        """The policy's observation: the last `history_length` (x, y, score)
        readings, oldest first. Contains no target information."""
        return tuple(self._history)

    @property
    def score(self):
        """The persistent proximity "warmth" at the current position -- what
        the agent observes (see _proximity). This is NOT the training reward
        (see `reward`)."""
        return self._current_score

    @property
    def reward(self):
        """The training reward for the most recent step (0 before any
        action). See perform_action for how it's computed."""
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

        # Illegal moves (off the subgrid) are simply not performed.
        legal_move = 0 <= new_x < self.subgrid_size and 0 <= new_y < self.subgrid_size
        if legal_move:
            self.agent_local = (new_x, new_y)

        self.steps += 1
        self._current_score = self._proximity(self.agent_local)
        self._history.append(State(x=self.agent_local[0], y=self.agent_local[1],
                                    score=self._current_score))

        self.terminated = self.is_done()
        self.truncated = (not self.terminated) and (self.steps >= self.max_steps)

        # Reward (kept separate from the observed proximity `score`):
        #   - reaching the target: a flat success_bonus.
        #   - otherwise: the CHANGE in proximity (potential-based shaping) --
        #     positive for getting closer, negative for backing off. Because
        #     it's a difference of a position-only potential, any loop or
        #     lingering in the radius telescopes to ~0, so the agent cannot
        #     farm reward by circling near the target (an effectively
        #     infinite discount on revisits).
        #   - plus a small step_penalty whenever the new cell is "empty"
        #     (proximity 0, i.e. outside the radius), nudging the agent to
        #     head toward the target region instead of wandering empty space.
        #   - plus a wall_penalty whenever the action was illegal (walked
        #     into the subgrid edge, a no-op) -- distinct from step_penalty
        #     since bumping a wall wastes a move regardless of whether the
        #     agent's (unchanged) position happens to be within the radius,
        #     and it's what directly discourages the wall-hugging/oscillating
        #     loops a purely position-based reward doesn't punish.
        # terminated has no future (GAE bootstraps V=0); truncated is an
        # artificial cutoff, so its final state is bootstrapped from the
        # critic instead (see rollout.py).
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
        """Print the current 8x8 subgrid. Row y=7 is printed first so that
        the grid reads bottom-left-origin (0,0) at the bottom-left corner."""
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


def run_simulation(env, engine="manual", network=None):
    """Run one full episode and return the trajectory.

    `env` must be constructed and configured by the caller (grid size,
    subgrid size, score radius, max steps, etc.) -- run_simulation never
    constructs a default environment itself. Call set_global_seed() before
    constructing it for a reproducible run.

    trajectory is a list of dicts: {"step", "action", "reward",
    "observation", "terminated", "truncated", "log_prob", "value"}. The
    first entry has action=None and reward=None and holds the initial
    observation (before any action is taken).

    "observation" is the last history_length (x, y, score) readings (what
    the policy sees, where score is the persistent proximity warmth);
    "reward" is env.reward for that step -- the shaped training signal,
    distinct from the observed score (see GridEnvironment.perform_action).
    "terminated" means the agent actually reached the
    target; "truncated" means the episode was cut off by env.max_steps
    without reaching it -- keeping these separate matters for GAE
    bootstrapping (a terminated episode has V(final state) = 0, a truncated
    one does not; note run_simulation never queries the network on the
    final state, so a caller doing PPO training must separately call
    network.get_value(env.get_state()) after this returns if env.truncated).

    "log_prob" and "value" are the acting policy's log-probability of the
    chosen action and its value estimate at that step -- both None except
    when engine="mlp_network", where they're needed to compute the PPO
    ratio and GAE advantages later.

    `network` is only used when engine="mlp_network": any object exposing
    `.act(observation) -> (action_index, log_prob, value, entropy)`, where
    action_index indexes into ACTIONS (see network.ActorCritic.act). It is
    ignored for other engines.

    engine="random" picks a uniformly random action each step via the
    shared global random stream (see set_global_seed / temporary_seed) --
    a baseline policy that needs no network.
    """
    trajectory = [{
        "step": 0, "action": None, "reward": None, "observation": env.get_state(),
        "terminated": env.terminated, "truncated": env.truncated,
        "log_prob": None, "value": None,
    }]

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
            action_index, log_prob, value, _entropy = network.act(env.get_state())
            action = ACTIONS[action_index]
        elif engine == "random":
            action = random.choice(ACTIONS)
        else:
            raise ValueError(f"Unknown engine: {engine!r}")

        observation = env.perform_action(action)
        # NOTE: fields in this entry span two timesteps. "action", "log_prob"
        # and "value" describe the state the agent acted FROM (s_t: log_prob
        # and value were computed above from env.get_state() *before*
        # perform_action). "observation" and "reward" describe the RESULT of
        # that action (s_{t+1}). GAE/PPO consumers (see rollout.py) rely on
        # this: values[t] = entry[t+1]["value"] = V(s_t), while the batch's
        # observation for step t comes from entry[t]["observation"] = s_t.
        trajectory.append({
            "step": env.steps, "action": action, "reward": env.reward, "observation": observation,
            "terminated": env.terminated, "truncated": env.truncated,
            "log_prob": log_prob, "value": value,
        })

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
