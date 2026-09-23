import torch
import torch.nn.functional as F


WDL_LOSS = 0
WDL_DRAW = 1
WDL_WIN = 2


def policy_wdl_loss(
    policy_logits,
    wdl_logits,
    target_policy,
    target_wdl,
    legal_action_mask=None,
):
    """
    Model 4 policy + WDL loss.

    Shapes:
        policy_logits:
            [batch, num_candidates]

        wdl_logits:
            [batch, 3]

        target_policy:
            [batch, num_candidates]

        target_wdl:
            [batch]

        legal_action_mask:
            [batch, num_candidates] bool
            True  = real candidate action
            False = padding

    The same function works for:

        Rich replay:
            dynamic padded legal-action batches.

        Legacy H12 replay:
            all 1,139 actions are candidates and the mask is all True.
    """

    # ============================================================
    # BASIC SHAPE VALIDATION
    # ============================================================

    if policy_logits.ndim != 2:
        raise ValueError(
            "policy_logits must have shape "
            "[batch, num_candidates]."
        )

    if target_policy.shape != policy_logits.shape:
        raise ValueError(
            "target_policy must have the same shape as "
            "policy_logits. "
            f"Got {tuple(target_policy.shape)} vs "
            f"{tuple(policy_logits.shape)}."
        )

    if wdl_logits.ndim != 2:
        raise ValueError(
            "wdl_logits must have shape [batch, 3]."
        )

    if wdl_logits.shape[0] != policy_logits.shape[0]:
        raise ValueError(
            "policy_logits and wdl_logits must have "
            "the same batch size."
        )

    if wdl_logits.shape[1] != 3:
        raise ValueError(
            "wdl_logits final dimension must be 3 "
            "(LOSS, DRAW, WIN)."
        )

    if target_wdl.ndim != 1:
        raise ValueError(
            "target_wdl must have shape [batch]."
        )

    if target_wdl.shape[0] != policy_logits.shape[0]:
        raise ValueError(
            "target_wdl batch size does not match "
            "policy_logits."
        )

    # ============================================================
    # LEGAL / PADDING MASK
    # ============================================================

    if legal_action_mask is None:

        legal_action_mask = torch.ones_like(
            policy_logits,
            dtype=torch.bool,
        )

    else:

        if legal_action_mask.shape != policy_logits.shape:
            raise ValueError(
                "legal_action_mask must have the same shape "
                "as policy_logits."
            )

        legal_action_mask = legal_action_mask.to(
            device=policy_logits.device,
            dtype=torch.bool,
        )

    if not torch.all(
        legal_action_mask.any(
            dim=-1
        )
    ):
        raise ValueError(
            "Every position must contain at least one "
            "real candidate action."
        )

    # ============================================================
    # TARGET VALIDATION
    # ============================================================

    if not torch.isfinite(
        target_policy
    ).all():
        raise ValueError(
            "target_policy contains NaN or infinity."
        )

    if torch.any(
        target_policy < 0
    ):
        raise ValueError(
            "target_policy contains negative probability."
        )

    # Padded actions must never receive target probability.
    padded_target_mass = target_policy.masked_select(
        ~legal_action_mask
    )

    if (
        padded_target_mass.numel() > 0
        and torch.any(
            padded_target_mass != 0
        )
    ):
        raise ValueError(
            "target_policy assigns probability to a padded "
            "or masked action."
        )

    target_row_sums = target_policy.sum(
        dim=-1
    )

    if not torch.allclose(
        target_row_sums,
        torch.ones_like(
            target_row_sums
        ),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "Each target_policy row must sum to 1."
        )

    if torch.any(
        target_wdl < WDL_LOSS
    ) or torch.any(
        target_wdl > WDL_WIN
    ):
        raise ValueError(
            "target_wdl values must be 0=LOSS, "
            "1=DRAW, or 2=WIN."
        )

    # ============================================================
    # POLICY LOSS
    # ============================================================
    #
    # Model 4 may pad a batch to the largest candidate count
    # present in that batch.
    #
    # Mask BEFORE log-softmax so padding receives effectively zero
    # probability and cannot steal probability mass from real actions.
    #
    # Use the smallest finite value rather than -inf. This avoids
    # possible 0 * -inf -> NaN behavior in probability-target losses.
    # ============================================================

    masked_value = torch.finfo(
        policy_logits.dtype
    ).min

    masked_policy_logits = policy_logits.masked_fill(
        ~legal_action_mask,
        masked_value,
    )

    log_probs = F.log_softmax(
        masked_policy_logits,
        dim=-1,
    )

    policy_loss = -(
        target_policy
        * log_probs
    ).sum(
        dim=-1
    ).mean()

    # ============================================================
    # WDL VALUE LOSS
    # ============================================================
    #
    # target_wdl:
    #
    #     0 = LOSS
    #     1 = DRAW
    #     2 = WIN
    #
    # wdl_logits are RAW logits.
    # Cross entropy performs log-softmax internally.
    # ============================================================

    value_loss = F.cross_entropy(
        wdl_logits,
        target_wdl.long(),
    )

    # ============================================================
    # TOTAL LOSS
    # ============================================================

    total_loss = (
        policy_loss
        + value_loss
    )

    # ============================================================
    # DIAGNOSTICS ONLY
    # ============================================================
    #
    # KL(target || model)
    #
    # target_policy is exactly zero on padding, so padded slots make
    # zero contribution here.
    # ============================================================

    with torch.no_grad():

        target_log_probs = torch.log(
            target_policy.clamp_min(
                1e-8
            )
        )

        policy_kl = (
            target_policy
            * (
                target_log_probs
                - log_probs
            )
        ).sum(
            dim=-1
        ).mean()

    return (
        total_loss,
        policy_loss,
        value_loss,
        policy_kl,
    )
