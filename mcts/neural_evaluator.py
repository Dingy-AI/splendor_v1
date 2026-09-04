import torch

from splendor_v1.network.policy import get_policy_probs
from splendor_v1.env.core.constants import OBSERVATION_SIZE
def slow_neural_evaluate(
    env,
    model,
    state,
    legal_actions=None
):
    obs = env.observation_encoder.encoder(
        state
    )

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

def neural_evaluate(
    env,
    model,
    state,
    legal_actions=None,
):
    obs = env.observation_encoder.encoder(
        state
    )

    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.inference_mode():
        policy_logits, value = model(
            obs_tensor
        )

    if legal_actions is None:
        legal_actions = env._legal_actions(
            state
        )

    if not legal_actions:
        raise ValueError(
            "neural_evaluate received no legal actions."
        )

    legal_action_ids = [
        env.action_to_id(action)
        for action in legal_actions
    ]

    legal_ids_tensor = torch.as_tensor(
        legal_action_ids,
        dtype=torch.long,
        device=policy_logits.device,
    )

    legal_logits = policy_logits[
        0,
        legal_ids_tensor,
    ]

    legal_probs = torch.softmax(
        legal_logits,
        dim=0,
    )

    return (
        legal_probs,
        value.item(),
    )
