import pytest
import torch
import torch.nn as nn

from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.mcts.neural_evaluator import neural_evaluate

class FakeModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.policy_head = torch.nn.Linear(
            1,
            ACTION_SPACE_SIZE,
        )

        with torch.no_grad():
            self.policy_head.weight.zero_()

            self.policy_head.bias.copy_(
                torch.arange(
                    ACTION_SPACE_SIZE,
                    dtype=torch.float32,
                )
            )

    def forward(self, x):
        # Zero weights make each output equal its bias:
        # action 0 -> 0, action 1 -> 1, etc.
        logits = self.policy_head(x[:, :1])

        value = x.new_full(
            (x.shape[0], 1),
            0.5,
        )

        return logits, value

    def forward_legal(self, x, legal_action_ids):
        logits, value = self.forward(x)

        legal_logits = logits.index_select(
            1,
            legal_action_ids,
        )

        return legal_logits, value

def test_neural_evaluate_uses_correct_legal_action_logits():

    env = SplendorEnv()
    env.reset()

    state = env.state

    legal_actions = env._legal_actions(state)

    legal_action_ids = [
        env.action_to_id(action)
        for action in legal_actions
    ]

    model = FakeModel()

    legal_probs, value = neural_evaluate(
        env,
        model,
        state,
        legal_actions=legal_actions,
    )

    expected_logits = torch.tensor(
        legal_action_ids,
        dtype=torch.float32,
    )

    expected_probs = torch.softmax(
        expected_logits,
        dim=0,
    )

    print("\nNEURAL EVALUATE TEST")
    print("legal action ids:")
    print(legal_action_ids)

    print("\nexpected logits:")
    print(expected_logits.tolist())

    print("\nexpected probs:")
    print(expected_probs.tolist())

    print("\nactual legal_probs:")
    print(legal_probs.cpu().tolist())

    print("\nvalue:")
    print(value)

    print(
        "\nmax absolute probability difference:",
        torch.max(
            torch.abs(
                legal_probs.cpu()
                - expected_probs
            )
        ).item(),
    )

    assert len(legal_probs) == len(
        legal_actions
    )

    assert torch.allclose(
        legal_probs.cpu(),
        expected_probs,
        atol=1e-6,
    )

    assert value == pytest.approx(
        0.5
    )

def test_neural_evaluate_preserves_legal_action_order():

    env = SplendorEnv()
    env.reset()

    state = env.state

    legal_actions = env._legal_actions(state)

    legal_action_ids = [
        env.action_to_id(action)
        for action in legal_actions
    ]

    model = FakeModel()

    legal_probs, _ = neural_evaluate(
        env,
        model,
        state,
        legal_actions=legal_actions,
    )

    expected_best_index = max(
        range(len(legal_action_ids)),
        key=lambda i: legal_action_ids[i],
    )

    actual_best_index = (
        legal_probs.argmax().item()
    )

    print("\nORDER TEST")

    print(
        "expected best index:",
        expected_best_index,
    )

    print(
        "expected best action id:",
        legal_action_ids[
            expected_best_index
        ],
    )

    print(
        "actual best index:",
        actual_best_index,
    )

    print(
        "actual best action id:",
        legal_action_ids[
            actual_best_index
        ],
    )

    print(
        "actual best probability:",
        legal_probs[
            actual_best_index
        ].item(),
    )

    print("\nAll action IDs and probabilities:")

    for action_id, prob in zip(
        legal_action_ids,
        legal_probs,
    ):
        print(
            f"id={action_id:4d} "
            f"prob={prob.item():.8f}"
        )

    assert actual_best_index == expected_best_index