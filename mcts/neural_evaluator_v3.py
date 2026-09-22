import torch

from splendor_v1.network.policy import get_policy_probs


# ============================================================
# WDL
# ============================================================

WDL_LOSS = 0
WDL_DRAW = 1
WDL_WIN = 2


def wdl_logits_to_value(
    wdl_logits,
):
    """
    Convert raw WDL logits into the scalar value expected by MCTS.

    WDL order:

        0 = LOSS
        1 = DRAW
        2 = WIN

    Scalar value:

        P(WIN) - P(LOSS)

    Range:
        [-1, +1]
    """

    wdl_probs = torch.softmax(
        wdl_logits,
        dim=-1,
    )

    value = (
        wdl_probs[..., WDL_WIN]
        - wdl_probs[..., WDL_LOSS]
    )

    return value


# ============================================================
# SLOW / FULL POLICY EVALUATOR
# ============================================================

def slow_neural_evaluate(
    env,
    model,
    state,
    legal_actions=None,
):
    observation = (
        env.observation_encoder.encoder(
            state
        )
    )

    weight = model.policy_head.weight

    obs_tensor = torch.as_tensor(
        observation,
        dtype=weight.dtype,
        device=weight.device,
    ).unsqueeze(0)

    with torch.inference_mode():

        (
            policy_logits,
            wdl_logits,
        ) = model(
            obs_tensor
        )

        value = wdl_logits_to_value(
            wdl_logits
        )

    action_mask = env.action_mask(
        state,
        legal_actions,
    )

    policy_probs = get_policy_probs(
        policy_logits,
        action_mask,
    )

    return (
        policy_probs.squeeze(0),
        value.item(),
    )


# ============================================================
# FAST LEGAL-ACTION EVALUATOR
# ============================================================

def neural_evaluate(
    env,
    model,
    state,
    legal_actions=None,
):
    if legal_actions is None:

        legal_actions = (
            env._legal_actions(
                state
            )
        )

    if not legal_actions:

        raise ValueError(
            "neural_evaluate received "
            "no legal actions."
        )

    observation = (
        env.observation_encoder.encoder(
            state
        )
    )

    weight = (
        model.policy_head.weight
    )

    obs_tensor = torch.as_tensor(
        observation,
        dtype=weight.dtype,
        device=weight.device,
    ).unsqueeze(0)

    legal_action_ids = [
        env.action_to_id(
            action
        )
        for action in legal_actions
    ]

    legal_ids_tensor = (
        torch.as_tensor(
            legal_action_ids,
            dtype=torch.long,
            device=weight.device,
        )
    )

    with torch.inference_mode():

        (
            legal_logits,
            wdl_logits,
        ) = model.forward_legal(
            obs_tensor,
            legal_ids_tensor,
        )

        legal_probs = torch.softmax(
            legal_logits[0],
            dim=0,
        )

        value = wdl_logits_to_value(
            wdl_logits
        )

        value_number = (
            value.item()
        )

    return (
        legal_probs,
        value_number,
    )