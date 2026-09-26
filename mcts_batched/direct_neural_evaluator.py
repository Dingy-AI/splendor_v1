import torch


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
# DIRECT MODEL 4 EVALUATOR
# ============================================================


class DirectNeuralEvaluator:
    """
    Synchronous batch-size-1 neural evaluator for Model 4.

    This class is intentionally equivalent to the existing V4/V5
    ``neural_evaluate(...)`` function. Its purpose is to establish
    an evaluator interface that MCTS can depend on without directly
    owning the neural-network inference implementation.

    Later, ``BatchedNeuralEvaluator`` can implement the same public
    ``evaluate(...)`` method while sending requests to a shared GPU
    batching service.

    Public contract
    ---------------

    evaluate(env, state, legal_actions=None)
        -> (legal_probs, value_number)

    legal_probs:
        Tensor with shape [num_legal_actions].
        Ordering exactly matches ``legal_actions``.

    value_number:
        Python float in [-1, +1], computed as

            P(WIN) - P(LOSS)

    Notes
    -----
    - This evaluator performs inference immediately.
    - It does not queue requests.
    - It does not batch across games.
    - It does not alter policy/value semantics.
    - The model is expected to already be on the desired device.
    """

    def __init__(
        self,
        model,
    ):
        self.model = model

    # --------------------------------------------------------
    # Model metadata
    # --------------------------------------------------------

    @property
    def device(self):
        """
        Device used by Model 4.

        Model 4 has no flat policy head, so the action embedding is
        used as the canonical source of device information.
        """

        return self.model.action_embedding.weight.device

    @property
    def dtype(self):
        """
        Floating-point dtype used by Model 4.
        """

        return self.model.action_embedding.weight.dtype

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def evaluate(
        self,
        env,
        state,
        legal_actions=None,
    ):
        """
        Evaluate one Splendor state immediately with Model 4.

        Only legal candidate actions are scored.

        Parameters
        ----------
        env:
            Splendor environment instance used for observation
            encoding and canonical action-ID conversion.

        state:
            Splendor state to evaluate.

        legal_actions:
            Optional precomputed legal actions. If omitted, they are
            generated from ``env._legal_actions(state)``.

        Returns
        -------
        (legal_probs, value_number)

        legal_probs:
            Tensor with shape [num_legal_actions]. Ordering exactly
            matches ``legal_actions``.

        value_number:
            Python float in [-1, +1].
        """

        if legal_actions is None:
            legal_actions = env._legal_actions(
                state
            )

        if not legal_actions:
            raise ValueError(
                "DirectNeuralEvaluator.evaluate received "
                "no legal actions."
            )

        observation = (
            env.observation_encoder.encoder(
                state
            )
        )

        obs_tensor = torch.as_tensor(
            observation,
            dtype=self.dtype,
            device=self.device,
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
            device=self.device,
        )

        with torch.inference_mode():
            (
                legal_logits,
                wdl_logits,
            ) = self.model.forward_legal(
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

            value_number = value.item()

        return (
            legal_probs,
            value_number,
        )

    def __call__(
        self,
        env,
        state,
        legal_actions=None,
    ):
        """
        Convenience alias for ``evaluate(...)``.
        """

        return self.evaluate(
            env=env,
            state=state,
            legal_actions=legal_actions,
        )
