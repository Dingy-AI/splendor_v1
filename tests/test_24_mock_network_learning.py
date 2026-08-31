# tests/training/test_training.py

import numpy as np
import torch

from splendor_v1.network.model import SplendorNetwork
from splendor_v1.network.losses import policy_value_loss
from splendor_v1.training.replay_buffer import ReplayBuffer
from splendor_v1.env.core.constants import OBSERVATION_SIZE

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE

def test_network_can_learn_from_replay_buffer():

    torch.manual_seed(0)
    np.random.seed(0)

    # -------------------------
    # Create model
    # -------------------------
    model = SplendorNetwork()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # -------------------------
    # Create easy fake example
    # -------------------------
    observation = np.random.randn(
        OBSERVATION_SIZE
    ).astype(np.float32)

    target_policy = np.zeros(
        ACTION_SPACE_SIZE,
        dtype=np.float32,
    )

    # Teach the network that action 42
    # should have probability 1.
    target_policy[42] = 1.0

    # Teach the network that this is
    # a winning position.
    target_value = 1.0

    # -------------------------
    # Fill replay buffer
    # -------------------------
    buffer = ReplayBuffer()

    for _ in range(32):
        buffer.add(
            observation,
            target_policy,
            target_value,
        )

    # -------------------------
    # Sample batch
    # -------------------------
    batch = buffer.sample(
        batch_size=16
    )

    observations, target_policies, target_values = zip(
        *batch
    )

    observations = torch.as_tensor(
        np.stack(observations),
        dtype=torch.float32,
    )

    target_policies = torch.as_tensor(
        np.stack(target_policies),
        dtype=torch.float32,
    )

    target_values = torch.as_tensor(
        target_values,
        dtype=torch.float32,
    )

    # -------------------------
    # Initial loss
    # -------------------------
    policy_logits, predicted_values = model(
        observations
    )

    initial_loss, _, _ = policy_value_loss(
        policy_logits,
        predicted_values,
        target_policies,
        target_values,
    )

    initial_loss_value = initial_loss.item()

    # -------------------------
    # Train
    # -------------------------
    for _ in range(20):

        policy_logits, predicted_values = model(
            observations
        )

        loss, _, _ = policy_value_loss(
            policy_logits,
            predicted_values,
            target_policies,
            target_values,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    # -------------------------
    # Final loss
    # -------------------------
    policy_logits, predicted_values = model(
        observations
    )

    final_loss, _, _ = policy_value_loss(
        policy_logits,
        predicted_values,
        target_policies,
        target_values,
    )

    final_loss_value = final_loss.item()

    # -------------------------
    # Verify learning
    # -------------------------
    assert final_loss_value < initial_loss_value