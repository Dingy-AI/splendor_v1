import torch
import torch.nn.functional as F


def policy_value_loss(
    policy_logits,
    predicted_value,
    target_policy,
    target_value,
):

    policy_loss = -(
        target_policy
        * F.log_softmax(policy_logits, dim=-1)
    ).sum(dim=-1).mean()

    value_loss = F.mse_loss(
        predicted_value.squeeze(-1),
        target_value,
    )

    total_loss = policy_loss + value_loss

    return total_loss, policy_loss, value_loss