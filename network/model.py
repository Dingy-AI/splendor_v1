import torch
import torch.nn as nn

from splendor_v1.network.network_constants import NUM_HIDDEN_LAYERS, HIDDEN_SIZE, VALUE_OUTPUT_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE


class SplendorNetwork(nn.Module):

    def __init__(
        self,
        observation_size: int,
        action_space_size: int = ACTION_SPACE_SIZE,
        hidden_size: int = HIDDEN_SIZE,
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(
            hidden_size,
            action_space_size,
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, VALUE_OUTPUT_SIZE),
            nn.Tanh(),
        )

    def forward(self, x):

        features = self.trunk(x)

        policy_logits = self.policy_head(features)

        value = self.value_head(features)

        return policy_logits, value