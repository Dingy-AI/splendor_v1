import os

import pytest
import torch

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


@pytest.fixture
def model():
    return SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )


def test_save_checkpoint_creates_file(
    tmp_path,
    model,
    optimizer,
):
    checkpoint_path = (
        tmp_path / "model_100_games.pt"
    )

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        games_played=100,
        history=[],
    )

    assert checkpoint_path.exists()


def test_load_checkpoint_restores_model_parameters(
    tmp_path,
    model,
    optimizer,
):
    checkpoint_path = (
        tmp_path / "model_100_games.pt"
    )

    original_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        games_played=100,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    load_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
    )

    for original, restored in zip(
        original_parameters,
        model.parameters(),
    ):
        assert torch.equal(
            original,
            restored,
        )


def test_load_checkpoint_restores_metadata(
    tmp_path,
    model,
    optimizer,
):
    checkpoint_path = (
        tmp_path / "model_250_games.pt"
    )

    history = [
        {
            "loss": 4.5,
            "policy_loss": 3.8,
            "value_loss": 0.7,
        }
    ]

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        games_played=250,
        history=history,
    )

    checkpoint_info = load_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
    )

    assert (
        checkpoint_info["games_played"]
        == 250
    )

    assert (
        checkpoint_info["history"]
        == history
    )


def test_load_checkpoint_restores_optimizer_state(
    tmp_path,
    model,
    optimizer,
):
    checkpoint_path = (
        tmp_path / "model_100_games.pt"
    )

    # Create optimizer state by doing one update.
    observation = torch.zeros(
        1,
        OBSERVATION_SIZE,
    )

    policy_logits, value = model(
        observation
    )

    loss = (
        policy_logits.sum()
        + value.sum()
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        games_played=100,
    )

    new_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    assert len(
        new_optimizer.state
    ) == 0

    load_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=new_optimizer,
    )

    assert len(
        new_optimizer.state
    ) > 0