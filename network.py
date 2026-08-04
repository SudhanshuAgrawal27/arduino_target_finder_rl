"""Actor-critic network for PPO: a simple fixed-window MLP.

Consumes the environment's stacked (x, y, score) observation window
(GridEnvironment.get_state()) and outputs action logits and a state value
estimate. No target information reaches this network.
"""

import torch
import torch.nn as nn

from simulator import ACTIONS, HISTORY_LENGTH, SUBGRID_SIZE

N_ACTIONS = len(ACTIONS)
FEATURES_PER_STEP = 3  # x, y, score


def obs_to_tensor(observation, subgrid_size=SUBGRID_SIZE, window_length=None):
    """Flatten a tuple of State(x, y, score) readings into one input vector,
    oldest first. `window_length` keeps only the most recent readings
    (default: all of them) -- lets the network see fewer steps than the
    environment's history_length actually produces. x/y normalized to
    [0, 1]; score is already in [0, 1]."""
    if window_length is not None:
        observation = observation[-window_length:]
    scale = max(subgrid_size - 1, 1)
    features = []
    for reading in observation:
        features.extend([reading.x / scale, reading.y / scale, reading.score])
    return torch.tensor(features, dtype=torch.float32)


def _orthogonal_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class ActorCritic(nn.Module):
    """Shared-trunk MLP over the fixed observation window, with a policy
    head (action logits) and a value head (scalar state value)."""

    def __init__(self, window_length=HISTORY_LENGTH, hidden_dim=64, n_actions=N_ACTIONS,
                 subgrid_size=SUBGRID_SIZE, num_layers=2):
        super().__init__()
        self.subgrid_size = subgrid_size
        self.window_length = window_length
        self.num_layers = num_layers
        input_dim = window_length * FEATURES_PER_STEP

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Standard PPO init: gain sqrt(2) on hidden layers, gain 1 on the
        # value head, small gain (0.01) on the policy head so the initial
        # policy starts close to uniform.
        for linear in self.trunk[0::2]:
            _orthogonal_init(linear, gain=2 ** 0.5)
        _orthogonal_init(self.value_head, gain=1.0)
        _orthogonal_init(self.policy_head, gain=0.01)

    def forward(self, obs):
        """obs: tensor of shape (window_length * 3,) or (batch, window_length * 3).
        Returns (logits, value)."""
        features = self.trunk(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def _forward_single(self, observation):
        """Run the network on one raw observation (a tuple of State readings,
        possibly longer than window_length -- only the trailing window_length
        of them are used) under no_grad, returning (logits, value) with the
        batch dim removed."""
        with torch.no_grad():
            obs_tensor = obs_to_tensor(
                observation, subgrid_size=self.subgrid_size, window_length=self.window_length
            ).to(self._device())
            logits, value = self.forward(obs_tensor.unsqueeze(0))
            return logits.squeeze(0), value.squeeze(0)

    def _device(self):
        return next(self.parameters()).device

    def act(self, observation):
        """Sample an action for a single raw observation (used during
        rollout collection). Returns (action_index, log_prob, value,
        entropy) as plain Python numbers; action_index indexes into
        simulator.ACTIONS."""
        logits, value = self._forward_single(observation)
        dist = torch.distributions.Categorical(logits=logits)
        action_index = dist.sample()
        return (
            action_index.item(),
            dist.log_prob(action_index).item(),
            value.item(),
            dist.entropy().item(),
        )

    def act_deterministic(self, observation):
        """Greedy (argmax) action for a single raw observation -- used for
        evaluation."""
        logits, _value = self._forward_single(observation)
        return logits.argmax().item()

    def action_probs(self, observation):
        """Softmax probability over all four actions for a single raw
        observation. Unlike act_deterministic (which only returns the
        greedy argmax), this exposes the full distribution -- used to
        visualize the policy's per-action confidence rather than to act,
        e.g. eval_demo_16-16-ldr-feedback.py's simulation-only "thinking"
        preview. Returns a dict keyed by simulator.ACTIONS."""
        logits, _value = self._forward_single(observation)
        probs = torch.softmax(logits, dim=-1)
        return {action: probs[i].item() for i, action in enumerate(ACTIONS)}

    def get_value(self, observation):
        """Value estimate for a single raw observation. Used to bootstrap
        GAE at a truncated episode's final state (a terminated episode's
        final state has no future, so its bootstrap value is 0 instead)."""
        _logits, value = self._forward_single(observation)
        return value.item()

    def evaluate_actions(self, obs_batch, action_batch):
        """Recompute log-probs, values, and entropy for a batch of
        (observation, action) pairs under the *current* policy parameters,
        with gradients -- used each PPO minibatch step. action_batch holds
        the actions taken during rollout (under the old policy), so the
        returned log_probs form the ratio r_t(theta) = exp(new_log_prob -
        old_log_prob).

        obs_batch: tensor (N, window_length * 3). action_batch: tensor (N,)
        of integer action indices. Returns (log_probs, values, entropy),
        each shape (N,).
        """
        logits, values = self.forward(obs_batch)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(action_batch)
        return log_probs, values, dist.entropy()
