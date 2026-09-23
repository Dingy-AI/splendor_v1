import torch

from splendor_v1.env.core.action_constants import (
    ACTION_SPACE_SIZE,
)
from splendor_v1.network.policy import (
    get_policy_probs,
)


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
# MODEL DEVICE / DTYPE
# ============================================================

def _model_device_and_dtype(
    model,
):
    """
    Model 4 has no flat policy_head.

    Use the action embedding as the canonical source of model
    device and floating-point dtype.
    """

    weight = (
        model.action_embedding.weight
    )

    return (
        weight.device,
        weight.dtype,
    )


# ============================================================
# SLOW / FULL ACTION-SPACE EVALUATOR
# ============================================================

def slow_neural_evaluate(
    env,
    model,
    state,
    legal_actions=None,
):
    """
    Diagnostic / compatibility evaluator.

    Model 4 does not have a flat 1,139-logit policy head, so to
    reproduce the old "full policy" behavior we explicitly score
    every canonical action ID:

        [0, 1, ..., ACTION_SPACE_SIZE - 1]

    The normal environment action mask then removes illegal actions.

    This is intentionally slower than neural_evaluate() and should
    not be used in normal MCTS.
    """

    if legal_actions is None:

        legal_actions = (
            env._legal_actions(
                state
            )
        )

    if not legal_actions:

        raise ValueError(
            "slow_neural_evaluate received "
            "no legal actions."
        )

    observation = (
        env.observation_encoder.encoder(
            state
        )
    )

    (
        device,
        dtype,
    ) = _model_device_and_dtype(
        model
    )

    obs_tensor = torch.as_tensor(
        observation,
        dtype=dtype,
        device=device,
    ).unsqueeze(0)

    all_action_ids = torch.arange(
        ACTION_SPACE_SIZE,
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():

        (
            policy_logits,
            wdl_logits,
        ) = model.forward_legal(
            obs_tensor,
            all_action_ids,
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
    """
    Fast Model 4 evaluator used by MCTS.

    Only the legal candidate actions are scored.

    Returns:

        legal_probs:
            Tensor with shape [num_legal_actions].

            Ordering exactly matches legal_actions.

        value_number:
            Scalar MCTS value in [-1, +1], computed as

                P(WIN) - P(LOSS)
    """

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

    (
        device,
        dtype,
    ) = _model_device_and_dtype(
        model
    )

    obs_tensor = torch.as_tensor(
        observation,
        dtype=dtype,
        device=device,
    ).unsqueeze(0)

    legal_action_ids = [
        env.action_to_id(
            action
        )
        for action
        in legal_actions
    ]

    legal_ids_tensor = torch.as_tensor(
        legal_action_ids,
        dtype=torch.long,
        device=device,
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
