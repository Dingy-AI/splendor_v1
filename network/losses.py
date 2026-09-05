import torch
import torch.nn.functional as F


def policy_value_loss(
    policy_logits,
    predicted_value,
    target_policy,
    target_value,
):
    log_probs = F.log_softmax(
        policy_logits,
        dim=-1,
    )

    # -------------------------
    # Actual training losses
    # -------------------------

    policy_loss = -(
        target_policy * log_probs
    ).sum(dim=-1).mean()

    value_loss = F.mse_loss(
        predicted_value.squeeze(-1),
        target_value,
    )

    total_loss = policy_loss + value_loss

    # -------------------------
    # Diagnostics only
    # -------------------------

    with torch.no_grad():

        target_log_probs = torch.log(
            target_policy.clamp_min(1e-8)
        )

        policy_kl = (
            target_policy *
            (target_log_probs - log_probs)
        ).sum(dim=-1).mean()

    return (
        total_loss,
        policy_loss,
        value_loss,
        policy_kl,
    )