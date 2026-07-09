"""Actor-critic network for PPO: a simple fixed-window MLP.

Consumes the environment's stacked (x, y, score) observation window
(GridEnvironment.get_state()) and outputs action logits and a state value
estimate. No target information ever reaches this network -- only what the
environment's observation already contains.
"""

import torch
import torch.nn as nn

from simulator import ACTIONS, HISTORY_LENGTH, SUBGRID_SIZE

N_ACTIONS = len(ACTIONS)
FEATURES_PER_STEP = 3  # x, y, score


def obs_to_tensor(observation, subgrid_size=SUBGRID_SIZE):
    """Flatten a history_length-long tuple of State(x, y, score) readings
    into a single input vector, oldest reading first (matching the
    environment's ordering). x and y are normalized to [0, 1] by the
    subgrid size; score is already in [0, 1]."""
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

    def __init__(self, history_length=HISTORY_LENGTH, hidden_dim=64, n_actions=N_ACTIONS,
                 subgrid_size=SUBGRID_SIZE):
        super().__init__()
        self.subgrid_size = subgrid_size
        input_dim = history_length * FEATURES_PER_STEP
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Standard PPO init: orthogonal weights, gain sqrt(2) on hidden
        # layers, gain 1 on the value head, and a small gain (0.01) on the
        # policy head so the initial policy starts close to uniform instead
        # of confidently wrong -- this matters for early exploration.
        _orthogonal_init(self.trunk[0], gain=2 ** 0.5)
        _orthogonal_init(self.trunk[2], gain=2 ** 0.5)
        _orthogonal_init(self.value_head, gain=1.0)
        _orthogonal_init(self.policy_head, gain=0.01)

    def forward(self, obs):
        """obs: tensor of shape (history_length * 3,) or (batch, history_length * 3).
        Returns (logits, value), each gaining/dropping the batch dim to match obs."""
        features = self.trunk(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def _forward_single(self, observation):
        """Run the network on one raw observation (a history_length-long
        tuple of State readings from GridEnvironment.get_state()) under
        no_grad, returning (logits, value) with the batch dim removed. The
        three inference helpers below all go through this; the PPO update
        uses evaluate_actions() instead, which keeps gradients and batches."""
        with torch.no_grad():
            obs_tensor = obs_to_tensor(observation, subgrid_size=self.subgrid_size).to(self._device())
            logits, value = self.forward(obs_tensor.unsqueeze(0))
            return logits.squeeze(0), value.squeeze(0)

    def _device(self):
        return next(self.parameters()).device

    def act(self, observation):
        """Sample an action for a single raw observation. This is what
        run_simulation(engine="mlp_network", ...) calls each step during
        rollout collection; it returns plain Python numbers since no graph
        is needed here (the PPO update recomputes everything with gradients
        via evaluate_actions()).

        Returns (action_index, log_prob, value, entropy). `action_index`
        indexes into `simulator.ACTIONS` for the actual action string.
        """
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
        evaluation, where we want the policy's best guess rather than an
        exploratory sample."""
        logits, _value = self._forward_single(observation)
        return logits.argmax().item()

    def get_value(self, observation):
        """Value estimate for a single raw observation, with no action
        sampling. Used to bootstrap GAE at a truncated episode's final state
        (a terminated episode's final state has no future, so its
        bootstrap value is 0 instead of this)."""
        _logits, value = self._forward_single(observation)
        return value.item()

    def evaluate_actions(self, obs_batch, action_batch):
        """Recompute log-probs, values, and entropy for a batch of
        (observation, action) pairs already converted to tensors, under the
        *current* policy parameters -- with gradients. This is what the PPO
        update calls each minibatch step; `action_batch` holds the actions
        actually taken during rollout collection (under the old policy), so
        the returned log_probs let the training loop form the ratio
        r_t(theta) = exp(new_log_prob - old_log_prob).

        obs_batch: tensor (N, history_length * 3). action_batch: tensor (N,)
        of integer action indices. Returns (log_probs, values, entropy),
        each a tensor of shape (N,).
        """
        logits, values = self.forward(obs_batch)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(action_batch)
        return log_probs, values, dist.entropy()
