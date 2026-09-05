import os

import torch

from splendor_v1.training.replay_buffer import ReplayBuffer


def save_checkpoint(
    path,
    model,
    optimizer,
    games_played,
    history=None,
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    checkpoint = {
        "games_played": games_played,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return {
        "games_played": checkpoint["games_played"],
        "history": checkpoint.get("history"),
    }

def save_model_if_needed(
    model,
    optimizer,
    games_played,
    history,
    checkpoint_every_games,
    next_checkpoint,
    checkpoint_dir="checkpoints",
    replay_buffer:ReplayBuffer = None,
):
    if games_played < next_checkpoint:
        return next_checkpoint

    save_checkpoint(
        path=(
            f"{checkpoint_dir}/"
            f"model_{games_played}_games.pt"
        ),
        model=model,
        optimizer=optimizer,
        games_played=games_played,
        history=history,
    )

    print(
        f"Saved checkpoint after "
        f"{games_played} games."
    )

    replay_buffer.save(
        path=(
                f"{checkpoint_dir}/"
                f"replay_{games_played}_games.pkl"
            ),
    )


    while next_checkpoint <= games_played:
        next_checkpoint += checkpoint_every_games

    return next_checkpoint