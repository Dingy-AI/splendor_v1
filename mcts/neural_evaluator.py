import torch

from splendor_v1.network.policy import get_policy_probs
from splendor_v1.env.core.constants import OBSERVATION_SIZE
def neural_evaluate(
    env,
    model,
    state,
    legal_actions=None
):
    obs = env.observation_encoder.encoder(
        state
    )
    # if len(obs) != OBSERVATION_SIZE:

    #     print("Observation size:", len(obs))
    #     print("Node type:", state.node_type)
    #     print("Current player:", state.current_player)

    #     for i, player in enumerate(state.players):
    #         print(
    #             f"Player {i} reserved:",
    #             len(player.reserved_cards)
    #         )

    #     print("Nobles:", len(state.nobles))


    # assert len(obs) == OBSERVATION_SIZE, (
    #     f"Expected observation size "
    #     f"{OBSERVATION_SIZE}, got {len(obs)}. "
    #     f"Node type: {state.node_type}, "
    #     f"current player: {state.current_player}"
    # )


    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.inference_mode():
        policy_logits, value = model(
            obs_tensor
        )

    action_mask = env.action_mask(
        state,
        legal_actions
    )

    policy_probs = get_policy_probs(
        policy_logits,
        action_mask,
    )

    return (
        policy_probs.squeeze(0),
        value.item(),
    )