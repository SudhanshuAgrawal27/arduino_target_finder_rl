"""Tests for the grid-world environment simulator.

Run with:
    pytest test_simulator.py -v
"""

import builtins

import pytest

from simulator import ACTIONS, State, GridEnvironment, run_simulation, set_global_seed

N_RANDOM_SEEDS = 500


@pytest.fixture(autouse=True)
def _deterministic_global_seed():
    """Every test starts from the same known point in the global random
    stream, so tests are deterministic and isolated from each other and
    from run order -- without any test having to pass a seed into
    GridEnvironment itself. Tests that care about a *specific* seed value
    call set_global_seed() again explicitly within the test body."""
    set_global_seed(0)


# ---------------------------------------------------------------------------
# Problem-instance validity / edge conditions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(N_RANDOM_SEEDS))
def test_subgrid_stays_within_global_grid(seed):
    set_global_seed(seed)
    env = GridEnvironment()
    ox, oy = env.origin
    assert 0 <= ox <= env.grid_size - env.subgrid_size
    assert 0 <= oy <= env.grid_size - env.subgrid_size


@pytest.mark.parametrize("seed", range(N_RANDOM_SEEDS))
def test_subgrid_always_contains_global_target(seed):
    set_global_seed(seed)
    env = GridEnvironment()
    global_pos = env.local_to_global(*env.target_local)
    assert global_pos == env.target_global
    assert 0 <= env.target_local[0] < env.subgrid_size
    assert 0 <= env.target_local[1] < env.subgrid_size


@pytest.mark.parametrize("seed", range(N_RANDOM_SEEDS))
def test_start_point_is_in_bounds_and_not_the_target(seed):
    set_global_seed(seed)
    env = GridEnvironment()
    x, y = env.start_local
    assert 0 <= x < env.subgrid_size
    assert 0 <= y < env.subgrid_size
    assert env.start_local != env.target_local


def test_target_global_is_grid_center():
    env = GridEnvironment()
    assert env.target_global == (8, 8)


# ---------------------------------------------------------------------------
# Movement / boundary conditions
# ---------------------------------------------------------------------------

def test_illegal_move_off_left_edge_is_a_no_op():
    env = GridEnvironment()
    env.agent_local = (0, 4)
    env.perform_action("left")
    assert env.agent_local == (0, 4)


def test_illegal_move_off_right_edge_is_a_no_op():
    env = GridEnvironment()
    env.agent_local = (env.subgrid_size - 1, 4)
    env.perform_action("right")
    assert env.agent_local == (env.subgrid_size - 1, 4)


def test_illegal_move_off_bottom_edge_is_a_no_op():
    env = GridEnvironment()
    env.agent_local = (4, 0)
    env.perform_action("down")
    assert env.agent_local == (4, 0)


def test_illegal_move_off_top_edge_is_a_no_op():
    env = GridEnvironment()
    env.agent_local = (4, env.subgrid_size - 1)
    env.perform_action("up")
    assert env.agent_local == (4, env.subgrid_size - 1)


@pytest.mark.parametrize(
    "action,delta", [("left", (-1, 0)), ("right", (1, 0)), ("up", (0, 1)), ("down", (0, -1))]
)
def test_legal_move_updates_position_by_one_cell(action, delta):
    env = GridEnvironment()
    env.agent_local = (4, 4)
    env.perform_action(action)
    assert env.agent_local == (4 + delta[0], 4 + delta[1])


def test_unknown_action_raises():
    env = GridEnvironment()
    with pytest.raises(ValueError):
        env.perform_action("diagonal")


def test_step_counter_increments_only_on_perform_action():
    env = GridEnvironment()
    assert env.steps == 0
    env.perform_action("up")
    assert env.steps == 1
    env.perform_action("left")
    assert env.steps == 2


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_proximity_is_one_at_target():
    env = GridEnvironment()
    assert env._proximity(env.target_local) == 1.0


def test_proximity_is_graded_by_distance():
    # Persistent warmth decreasing with distance: dist 1 -> 2/3, dist 2 ->
    # 1/3 (for score_radius=2), 0 beyond the radius.
    env = GridEnvironment(score_radius=2)
    tx, ty = env.target_local
    for d, expected in [(1, 2 / 3), (2, 1 / 3), (3, 0.0)]:
        pos = (tx - d, ty) if tx - d >= 0 else (tx + d, ty)
        if not (0 <= pos[0] < env.subgrid_size):
            pytest.skip("degenerate placement for this seed")
        assert env._proximity(pos) == pytest.approx(expected)


def test_proximity_is_persistent_across_revisits():
    # Unlike the old one-time radius bonus, proximity is a pure function of
    # position: the same cell always reads the same warmth, so a revisited
    # cell in the observation history is not aliased to "far away".
    env = GridEnvironment(score_radius=2)
    tx, ty = env.target_local
    near = (tx - 1, ty)
    far = (tx - 2, ty)
    if far[0] < 0:
        pytest.skip("degenerate placement for this seed")
    first = env._proximity(near)
    assert env._proximity(near) == first          # same value on revisit
    assert first > env._proximity(far) > 0.0       # closer reads strictly warmer


def test_potential_shaping_makes_a_round_trip_net_zero():
    # The non-terminal reward is the CHANGE in proximity, so moving away and
    # back nets ~0 -- the agent cannot farm reward by circling near the
    # target. step_penalty=0 isolates the shaping term.
    env = GridEnvironment(score_radius=2, max_steps=100, step_penalty=0.0)
    tx, ty = env.target_local
    if tx - 2 < 0 or tx >= env.subgrid_size:
        pytest.skip("degenerate placement for this seed")
    env.agent_local = (tx - 2, ty)                 # distance 2
    env._current_score = env._proximity(env.agent_local)

    env.perform_action("right")                    # -> distance 1 (warmer)
    r_in = env.reward
    env.perform_action("left")                     # -> distance 2 (colder)
    r_out = env.reward
    assert r_in > 0 and r_out < 0
    assert r_in + r_out == pytest.approx(0.0)


def test_step_penalty_charged_on_empty_cells():
    env = GridEnvironment(score_radius=2, max_steps=100, step_penalty=0.1)
    env.target_local = (env.subgrid_size - 1, env.subgrid_size - 1)  # far corner
    env.agent_local = (0, 0)
    env._current_score = env._proximity(env.agent_local)
    assert env._current_score == 0.0               # start is empty (outside radius)

    env.perform_action("up")                       # empty -> empty, proximity delta 0
    assert env.reward == pytest.approx(-0.1)


def test_wall_penalty_charged_on_illegal_move():
    env = GridEnvironment(max_steps=100, step_penalty=0.0, wall_penalty=0.1)
    env.agent_local = (0, 4)
    env._current_score = env._proximity(env.agent_local)
    env.perform_action("left")  # illegal: off the left edge, a no-op
    assert env.agent_local == (0, 4)
    assert env.reward == pytest.approx(-0.1)


def test_wall_penalty_not_charged_on_legal_move():
    env = GridEnvironment(max_steps=100, step_penalty=0.0, wall_penalty=0.1)
    env.agent_local = (4, 4)
    env._current_score = env._proximity(env.agent_local)
    prev_proximity = env._current_score
    env.perform_action("up")  # legal
    assert env.reward == pytest.approx(env._current_score - prev_proximity)


def test_wall_penalty_stacks_with_step_penalty_outside_radius():
    env = GridEnvironment(score_radius=2, max_steps=100, step_penalty=0.01, wall_penalty=0.05)
    env.target_local = (env.subgrid_size - 1, env.subgrid_size - 1)  # far corner
    env.agent_local = (0, 0)
    env._current_score = env._proximity(env.agent_local)
    assert env._current_score == 0.0                # start is empty (outside radius)

    env.perform_action("left")                      # illegal AND stays outside radius
    assert env.agent_local == (0, 0)
    assert env.reward == pytest.approx(-0.06)


def test_reaching_target_pays_success_bonus():
    env = GridEnvironment(max_steps=100, success_bonus=1.0)
    tx, ty = env.target_local
    sx, sy = env.start_local
    while (sx, sy) != (tx, ty):
        if sx != tx:
            action = "right" if tx > sx else "left"
            sx += 1 if tx > sx else -1
        else:
            action = "up" if ty > sy else "down"
            sy += 1 if ty > sy else -1
        env.perform_action(action)
    assert env.terminated is True
    assert env.reward == 1.0


# ---------------------------------------------------------------------------
# Observation (history window)
# ---------------------------------------------------------------------------

def test_observation_length_matches_history_length():
    env = GridEnvironment(history_length=4)
    assert len(env.get_state()) == 4
    env.perform_action("up")
    assert len(env.get_state()) == 4


def test_observation_is_padded_with_initial_reading_at_episode_start():
    env = GridEnvironment(history_length=4)
    obs = env.get_state()
    assert all(entry == obs[0] for entry in obs)
    assert obs[0].x == env.agent_local[0]
    assert obs[0].y == env.agent_local[1]


def test_observation_slides_as_steps_accumulate():
    env = GridEnvironment(history_length=3)
    env.agent_local = (3, 3)
    env._current_score = env._proximity(env.agent_local)
    env._history = env._history.__class__([State(3, 3, env.score)] * 3, maxlen=3)

    env.perform_action("up")
    env.perform_action("up")
    obs = env.get_state()
    assert obs[-1].y == 5
    assert obs[-2].y == 4
    assert obs[-3].y == 3


def test_observation_never_contains_target_information():
    env = GridEnvironment()
    for reading in env.get_state():
        assert set(reading._fields) == {"x", "y", "score"}


# ---------------------------------------------------------------------------
# terminated vs truncated
# ---------------------------------------------------------------------------

def test_reaching_target_sets_terminated_not_truncated():
    env = GridEnvironment(max_steps=100)
    tx, ty = env.target_local
    sx, sy = env.start_local
    while (sx, sy) != (tx, ty):
        if sx != tx:
            action = "right" if tx > sx else "left"
            sx += 1 if tx > sx else -1
        else:
            action = "up" if ty > sy else "down"
            sy += 1 if ty > sy else -1
        env.perform_action(action)
    assert env.terminated is True
    assert env.truncated is False
    assert env.score == 1.0


def test_hitting_step_cap_sets_truncated_not_terminated():
    env = GridEnvironment(max_steps=5)
    env.target_local = (env.subgrid_size - 1, env.subgrid_size - 1)
    env.agent_local = (0, 0)
    for _ in range(5):
        env.perform_action("left")  # illegal every time; agent never reaches target
    assert env.terminated is False
    assert env.truncated is True


def test_episode_does_not_terminate_or_truncate_before_its_time():
    env = GridEnvironment(max_steps=10)
    env.target_local = (env.subgrid_size - 1, env.subgrid_size - 1)
    env.agent_local = (0, 0)
    for _ in range(9):
        env.perform_action("left")
        assert env.terminated is False
        assert env.truncated is False


# ---------------------------------------------------------------------------
# seeding / reset()
# ---------------------------------------------------------------------------

def test_reset_without_reseeding_produces_new_instances():
    set_global_seed(1)
    env = GridEnvironment()
    instances = set()
    for _ in range(20):
        instances.add((env.origin, env.target_local, env.start_local))
        env.reset()
    assert len(instances) > 1


def test_set_global_seed_makes_a_run_reproducible():
    set_global_seed(99)
    env = GridEnvironment()
    a = (env.origin, env.target_local, env.start_local)

    set_global_seed(99)
    env = GridEnvironment()
    b = (env.origin, env.target_local, env.start_local)

    assert a == b


def test_set_global_seed_makes_reset_reproducible():
    env = GridEnvironment()

    set_global_seed(99)
    env.reset()
    a = (env.origin, env.target_local, env.start_local)

    set_global_seed(99)
    env.reset()
    b = (env.origin, env.target_local, env.start_local)

    assert a == b


def test_reset_clears_episode_bookkeeping():
    env = GridEnvironment()
    env.perform_action("up")
    env.perform_action("up")
    env.reset()
    assert env.steps == 0
    assert env.terminated is False
    assert env.truncated is False
    assert env.reward == 0.0


def test_reset_returns_initial_observation():
    env = GridEnvironment()
    obs = env.reset()
    assert obs == env.get_state()


# ---------------------------------------------------------------------------
# render() smoke test (just verify it doesn't crash and marks agent/target)
# ---------------------------------------------------------------------------

def test_render_does_not_crash(capsys):
    env = GridEnvironment()
    env.render()
    out = capsys.readouterr().out
    assert out.count("A") == 1
    assert out.count("T") == 1


def test_render_marks_agent_at_target_with_x(capsys):
    env = GridEnvironment()
    env.agent_local = env.target_local
    env.render()
    out = capsys.readouterr().out
    assert "X" in out


# ---------------------------------------------------------------------------
# run_simulation
# ---------------------------------------------------------------------------

def _scripted_input(monkeypatch, actions):
    it = iter(actions)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(it))


def test_run_simulation_manual_reaches_target(monkeypatch, capsys):
    env = GridEnvironment(max_steps=100)
    tx, ty = env.target_local
    sx, sy = env.start_local
    actions = []
    while (sx, sy) != (tx, ty):
        if sx != tx:
            actions.append("right" if tx > sx else "left")
            sx += 1 if tx > sx else -1
        else:
            actions.append("up" if ty > sy else "down")
            sy += 1 if ty > sy else -1
    _scripted_input(monkeypatch, actions)

    trajectory = run_simulation(engine="manual", env=env)

    assert trajectory[0]["action"] is None
    assert trajectory[0]["reward"] is None
    assert len(trajectory[0]["observation"]) == env.history_length
    assert trajectory[-1]["terminated"] is True
    assert trajectory[-1]["truncated"] is False
    assert trajectory[-1]["reward"] == 1.0
    assert len(trajectory) == len(actions) + 1


def test_run_simulation_manual_truncates_at_max_steps(monkeypatch):
    env = GridEnvironment(max_steps=5)
    env.target_local = (env.subgrid_size - 1, env.subgrid_size - 1)
    env.agent_local = (0, 0)
    env._history = env._history.__class__(
        [State(0, 0, env.score)] * env.history_length, maxlen=env.history_length
    )
    _scripted_input(monkeypatch, ["left"] * 5)

    trajectory = run_simulation(engine="manual", env=env)

    assert trajectory[-1]["truncated"] is True
    assert trajectory[-1]["terminated"] is False
    assert len(trajectory) == 6  # initial + 5 steps


class _StubNetwork:
    """Duck-typed stand-in for network.ActorCritic: exposes the same
    `.act(observation) -> (action_index, log_prob, value, entropy)` shape
    run_simulation expects, without depending on torch in this test file."""

    def __init__(self, action_indices):
        self._it = iter(action_indices)

    def act(self, observation):
        return next(self._it), None, None, None


def test_mlp_network_engine_requires_a_network():
    env = GridEnvironment()
    with pytest.raises(ValueError):
        run_simulation(engine="mlp_network", env=env, network=None)


def test_mlp_network_engine_uses_networks_actions():
    env = GridEnvironment(max_steps=5)
    env.target_local = (env.subgrid_size - 1, env.subgrid_size - 1)
    env.agent_local = (0, 0)
    env._history = env._history.__class__(
        [State(0, 0, env.score)] * env.history_length, maxlen=env.history_length
    )
    right_index = ACTIONS.index("right")
    network = _StubNetwork([right_index] * 5)

    trajectory = run_simulation(engine="mlp_network", env=env, network=network)

    assert [t["action"] for t in trajectory[1:]] == ["right"] * 5
    assert trajectory[-1]["observation"][-1].x == 5


def test_mlp_network_engine_can_reach_target():
    env = GridEnvironment(max_steps=100)
    tx, ty = env.target_local
    sx, sy = env.start_local
    action_indices = []
    while (sx, sy) != (tx, ty):
        if sx != tx:
            action = "right" if tx > sx else "left"
            sx += 1 if tx > sx else -1
        else:
            action = "up" if ty > sy else "down"
            sy += 1 if ty > sy else -1
        action_indices.append(ACTIONS.index(action))
    network = _StubNetwork(action_indices)

    trajectory = run_simulation(engine="mlp_network", env=env, network=network)

    assert trajectory[-1]["terminated"] is True
    assert trajectory[-1]["reward"] == 1.0


def test_run_simulation_rejects_unknown_engine():
    env = GridEnvironment()
    with pytest.raises(ValueError):
        run_simulation(engine="random_walk", env=env)


def test_manual_engine_rejects_invalid_typed_action(monkeypatch, capsys):
    env = GridEnvironment(max_steps=1)
    responses = iter(["banana", ACTIONS[0]])
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(responses))
    run_simulation(engine="manual", env=env)
    out = capsys.readouterr().out
    assert "Invalid action" in out