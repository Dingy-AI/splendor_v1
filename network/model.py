import torch
import torch.nn as nn

from splendor_v1.network.network_constants import NUM_HIDDEN_LAYERS, HIDDEN_SIZE, VALUE_OUTPUT_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.env.core.constants import OBSERVATION_SIZE
import torch.nn.functional as F

NUM_RESIDUAL_BLOCKS = 3
class ResidualBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
    ):
        super().__init__()

        self.linear = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.layer_norm = nn.LayerNorm(
            hidden_size
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = F.relu(
            self.layer_norm(
                self.linear(x)
            )
        )

        return x + residual



class SplendorNetwork(nn.Module):

    def __init__(
        self,
        observation_size: int = OBSERVATION_SIZE,
        action_space_size: int = ACTION_SPACE_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_residual_blocks: int = NUM_RESIDUAL_BLOCKS,        
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(
                observation_size,
                hidden_size,
            ),
            nn.ReLU(),
        )

        # -------------------------
        # Residual trunk
        # -------------------------
        # 512 -> 512 residual transformations
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_size)
                for _ in range(
                    num_residual_blocks
                )
            ]
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

        features = self.stem(x)

        features = self.residual_blocks(
            features
        )

        policy_logits = self.policy_head(
            features
        )

        value = self.value_head(
            features
        )

        return policy_logits, value