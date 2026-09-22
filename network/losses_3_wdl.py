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
):
    # ============================================================
    # POLICY LOSS
    # ============================================================

    log_probs = F.log_softmax(
        policy_logits,
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