import numpy as np
import torch

#TODO DELETE later this whole section is useless now :'()

def apply_action_mask(
    policy_logits: torch.Tensor,
    action_mask: np.ndarray,
) -> torch.Tensor:

    mask_tensor = torch.as_tensor(
        action_mask,
        dtype=torch.bool,
        device=policy_logits.device,
    )

    # If policy_logits has a batch dimension,
    # make the mask match it.
    if policy_logits.dim() == 2 and mask_tensor.dim() == 1:
        mask_tensor = mask_tensor.unsqueeze(0)

    if not torch.all(mask_tensor.any(dim=-1)):
        raise ValueError(
            "Action mask contains no legal actions."
        )

    return policy_logits.masked_fill(
        ~mask_tensor,
        float("-inf"),
    )


def get_policy_probs(
    policy_logits: torch.Tensor,
    action_mask: np.ndarray,
    ) -> torch.Tensor:

    masked_logits = apply_action_mask(
        policy_logits,
        action_mask
    )

    return torch.softmax(
        masked_logits,
        dim=-1
    )