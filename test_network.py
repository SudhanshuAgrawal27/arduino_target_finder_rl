"""Tests for network.py's observation windowing.

Run with:
    pytest test_network.py -v
"""

import torch

from network import ActorCritic, obs_to_tensor
from simulator import State


def _readings(n):
    """n distinct State readings, oldest first, so a windowed slice is
    trivially identifiable by its x value."""
    return tuple(State(x=i, y=0, score=0.0) for i in range(n))


def test_obs_to_tensor_uses_full_observation_by_default():
    obs = _readings(4)
    tensor = obs_to_tensor(obs, subgrid_size=8)
    assert tensor.shape == (4 * 3,)


def test_obs_to_tensor_window_length_keeps_only_trailing_readings():
    obs = _readings(6)  # x = 0..5
    windowed = obs_to_tensor(obs, subgrid_size=8, window_length=2)
    full = obs_to_tensor(obs[-2:], subgrid_size=8)
    assert torch.equal(windowed, full)
    assert windowed.shape == (2 * 3,)


def test_obs_to_tensor_window_length_equal_to_length_is_a_no_op():
    obs = _readings(4)
    assert torch.equal(
        obs_to_tensor(obs, subgrid_size=8, window_length=4),
        obs_to_tensor(obs, subgrid_size=8),
    )


def test_actor_critic_input_dim_matches_window_length():
    model = ActorCritic(window_length=2, hidden_dim=16, subgrid_size=8)
    assert model.trunk[0].in_features == 2 * 3


def test_actor_critic_accepts_observation_longer_than_window_length():
    # The environment may produce more history than the network consumes;
    # act_deterministic must still work off the trailing window_length.
    model = ActorCritic(window_length=2, hidden_dim=16, subgrid_size=8)
    obs = _readings(6)
    action = model.act_deterministic(obs)
    assert action in range(4)


def _linear_layers(model):
    return [layer for layer in model.trunk if isinstance(layer, torch.nn.Linear)]


def test_actor_critic_defaults_to_two_trunk_layers():
    model = ActorCritic(window_length=4, hidden_dim=16, subgrid_size=8)
    assert len(_linear_layers(model)) == 2


def test_actor_critic_num_layers_controls_trunk_depth():
    model = ActorCritic(window_length=4, hidden_dim=16, subgrid_size=8, num_layers=3)
    linears = _linear_layers(model)
    assert len(linears) == 3
    assert linears[0].in_features == 4 * 3 and linears[0].out_features == 16
    assert all(layer.in_features == 16 and layer.out_features == 16 for layer in linears[1:])


def test_actor_critic_forward_works_with_custom_num_layers():
    model = ActorCritic(window_length=4, hidden_dim=8, subgrid_size=8, num_layers=4)
    obs = _readings(4)
    action = model.act_deterministic(obs)
    assert action in range(4)


def test_actor_critic_param_count_matches_hand_derived_formula():
    # Cross-checks the manual "params(hidden_dim, window_length, layers)" formula
    # used to plan the hidden_dim/num_layers sweep against the real module.
    window_length, hidden_dim, num_layers = 4, 32, 3
    model = ActorCritic(window_length=window_length, hidden_dim=hidden_dim, subgrid_size=8, num_layers=num_layers)
    actual = sum(p.numel() for p in model.parameters())

    input_dim = window_length * 3
    expected = input_dim * hidden_dim + hidden_dim
    expected += (num_layers - 1) * (hidden_dim * hidden_dim + hidden_dim)
    expected += hidden_dim * 4 + 4   # policy head
    expected += hidden_dim * 1 + 1   # value head
    assert actual == expected == 2693
