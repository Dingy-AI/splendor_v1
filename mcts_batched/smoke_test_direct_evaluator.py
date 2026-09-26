"""
Smoke test for the MCTS evaluator abstraction.

Confirms that:

1. The legacy V4/V5 ``neural_evaluate(...)`` function and
   ``DirectNeuralEvaluator.evaluate(...)`` return equivalent
   policy probabilities and scalar WDL values.

2. Original MCTS V5 pruning and the evaluator-injected fork
   produce equivalent search results from the same Splendor states.

The test intentionally disables root Dirichlet noise by default so
the search comparison is deterministic. Use --test-root-noise to
also test the noisy-root path with matched RNG seeds.
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

from splendor_v1.env.env import SplendorEnv
from splendor_v1.network.model_4_legal_scorer import (
    SplendorNetwork,
)

from splendor_v1.mcts.neural_evaluator_v4 import (
    neural_evaluate,
)

from splendor_v1.mcts.mcts_v5_pruning import (
    MCTS as OriginalMCTS,
)

from splendor_v1.mcts_batched.direct_neural_evaluator import (
    DirectNeuralEvaluator,
)

from splendor_v1.mcts_batched.mcts_v5_direct import (
    MCTS as DirectMCTS,
)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_CHECKPOINT = (
    "splendor_v1/training_v5/data/"
    "model_1900_games.pt"
)


# ============================================================
# MODEL LOADING
# ============================================================

def extract_model_state_dict(
    checkpoint,
):
    """
    Accept either:

      - a raw PyTorch state_dict
      - a training checkpoint containing model_state_dict
    """

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint[
            "model_state_dict"
        ]

    if (
        isinstance(checkpoint, dict)
        and checkpoint
        and all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        )
    ):
        return checkpoint

    raise RuntimeError(
        "Checkpoint must either be a raw model state_dict "
        "or contain 'model_state_dict'."
    )


def load_model(
    checkpoint_path,
    device,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = SplendorNetwork()

    model.load_state_dict(
        extract_model_state_dict(
            checkpoint
        ),
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# STATE SAMPLING
# ============================================================

def sample_reachable_states(
    num_states,
    seed,
    stride,
):
    """
    Produce real reachable Splendor states by playing random legal
    moves from seeded environments.

    This is only for smoke-test coverage. Search/evaluator equality
    is what is being tested, not the quality of these random games.
    """

    states = []

    chooser = random.Random(
        seed
    )

    game_index = 0

    while len(states) < num_states:

        env = SplendorEnv()

        game_seed = (
            int(seed)
            + game_index
        )

        env.reset(
            seed=game_seed
        )

        step_index = 0

        while (
            len(states) < num_states
            and step_index < 300
        ):

            state = env.state

            if env._check_terminated(
                state
            ):
                break

            legal_actions = (
                env._legal_actions(
                    state
                )
            )

            if not legal_actions:
                break

            if (
                step_index == 0
                or step_index % stride == 0
            ):
                states.append(
                    state.clone()
                )

                if len(states) >= num_states:
                    break

            action = legal_actions[
                chooser.randrange(
                    len(legal_actions)
                )
            ]

            env.step(
                action
            )

            step_index += 1

        game_index += 1

        if game_index > 100:
            raise RuntimeError(
                "Could not collect enough reachable "
                "states for the smoke test."
            )

    return states


# ============================================================
# COMPARISON HELPERS
# ============================================================

def action_key(
    env,
    action,
):
    try:
        return int(
            env.action_to_id(
                action
            )
        )
    except Exception:
        return repr(action)


def assert_close(
    label,
    a,
    b,
    atol=1e-7,
    rtol=1e-6,
):
    if not math.isclose(
        float(a),
        float(b),
        abs_tol=atol,
        rel_tol=rtol,
    ):
        raise AssertionError(
            f"{label} mismatch: "
            f"{a!r} != {b!r}"
        )


def compare_metadata(
    original,
    direct,
    prefix="metadata",
):
    original_keys = set(
        original.keys()
    )

    direct_keys = set(
        direct.keys()
    )

    if original_keys != direct_keys:
        missing = sorted(
            original_keys - direct_keys
        )

        extra = sorted(
            direct_keys - original_keys
        )

        raise AssertionError(
            f"{prefix} key mismatch. "
            f"Missing={missing}, extra={extra}"
        )

    for key in sorted(
        original_keys
    ):
        a = original[key]
        b = direct[key]

        label = (
            f"{prefix}.{key}"
        )

        if isinstance(
            a,
            (float, np.floating),
        ):
            assert_close(
                label,
                a,
                b,
                atol=1e-7,
                rtol=1e-6,
            )

        else:
            if a != b:
                raise AssertionError(
                    f"{label} mismatch: "
                    f"{a!r} != {b!r}"
                )


def root_child_stats(
    env,
    root,
):
    result = {}

    for child in root.children:

        key = action_key(
            env,
            child.action,
        )

        if key in result:
            raise AssertionError(
                "Duplicate root action key: "
                f"{key!r}"
            )

        visits = int(
            child.visits
        )

        value = float(
            child.value
        )

        result[key] = {
            "visits":
                visits,

            "value":
                value,

            "q":
                (
                    value / visits
                    if visits > 0
                    else 0.0
                ),

            "prior":
                float(
                    getattr(
                        child,
                        "prior",
                        0.0,
                    )
                ),

            "network_prior":
                float(
                    getattr(
                        child,
                        "network_prior",
                        getattr(
                            child,
                            "prior",
                            0.0,
                        ),
                    )
                ),
        }

    return result


def compare_roots(
    env_original,
    root_original,
    env_direct,
    root_direct,
):
    if int(root_original.visits) != int(
        root_direct.visits
    ):
        raise AssertionError(
            "Root visit mismatch: "
            f"{root_original.visits} != "
            f"{root_direct.visits}"
        )

    stats_original = root_child_stats(
        env_original,
        root_original,
    )

    stats_direct = root_child_stats(
        env_direct,
        root_direct,
    )

    keys_original = set(
        stats_original
    )

    keys_direct = set(
        stats_direct
    )

    if keys_original != keys_direct:
        raise AssertionError(
            "Root child action-set mismatch. "
            f"Original-only="
            f"{sorted(keys_original - keys_direct)}, "
            f"Direct-only="
            f"{sorted(keys_direct - keys_original)}"
        )

    for key in sorted(
        keys_original,
        key=str,
    ):
        a = stats_original[key]
        b = stats_direct[key]

        if a["visits"] != b["visits"]:
            raise AssertionError(
                f"Action {key} visit mismatch: "
                f"{a['visits']} != "
                f"{b['visits']}"
            )

        for field in (
            "value",
            "q",
            "prior",
            "network_prior",
        ):
            assert_close(
                f"action {key} {field}",
                a[field],
                b[field],
                atol=1e-7,
                rtol=1e-6,
            )


# ============================================================
# TEST 1: EVALUATOR EQUIVALENCE
# ============================================================

def test_evaluator_equivalence(
    model,
    states,
    atol,
    rtol,
):
    print()
    print("=" * 72)
    print("TEST 1: DIRECT EVALUATOR EQUIVALENCE")
    print("=" * 72)

    evaluator = (
        DirectNeuralEvaluator(
            model=model,
        )
    )

    max_policy_diff = 0.0
    max_value_diff = 0.0

    for index, state in enumerate(
        states,
        start=1,
    ):
        env = SplendorEnv()

        legal_actions = (
            env._legal_actions(
                state
            )
        )

        if not legal_actions:
            raise AssertionError(
                f"State {index} unexpectedly "
                "has no legal actions."
            )

        with torch.inference_mode():

            old_probs, old_value = (
                neural_evaluate(
                    env,
                    model,
                    state,
                    legal_actions=(
                        legal_actions
                    ),
                )
            )

            new_probs, new_value = (
                evaluator.evaluate(
                    env=env,
                    state=state,
                    legal_actions=(
                        legal_actions
                    ),
                )
            )

        old_probs_cpu = (
            old_probs.detach()
            .float()
            .cpu()
        )

        new_probs_cpu = (
            new_probs.detach()
            .float()
            .cpu()
        )

        if old_probs_cpu.shape != (
            new_probs_cpu.shape
        ):
            raise AssertionError(
                f"State {index} policy shape "
                f"mismatch: "
                f"{tuple(old_probs_cpu.shape)} != "
                f"{tuple(new_probs_cpu.shape)}"
            )

        diff = torch.max(
            torch.abs(
                old_probs_cpu
                - new_probs_cpu
            )
        ).item()

        value_diff = abs(
            float(old_value)
            - float(new_value)
        )

        max_policy_diff = max(
            max_policy_diff,
            diff,
        )

        max_value_diff = max(
            max_value_diff,
            value_diff,
        )

        if not torch.allclose(
            old_probs_cpu,
            new_probs_cpu,
            atol=atol,
            rtol=rtol,
        ):
            worst_index = int(
                torch.argmax(
                    torch.abs(
                        old_probs_cpu
                        - new_probs_cpu
                    )
                ).item()
            )

            raise AssertionError(
                f"State {index} policy mismatch. "
                f"Max diff={diff:.12g} at "
                f"legal-action index "
                f"{worst_index}. "
                f"Old="
                f"{old_probs_cpu[worst_index].item():.12g}, "
                f"new="
                f"{new_probs_cpu[worst_index].item():.12g}"
            )

        if not math.isclose(
            float(old_value),
            float(new_value),
            abs_tol=atol,
            rel_tol=rtol,
        ):
            raise AssertionError(
                f"State {index} value mismatch: "
                f"{old_value!r} != "
                f"{new_value!r}"
            )

        print(
            f"State {index:>2}: PASS | "
            f"legal={len(legal_actions):>2} | "
            f"policy max diff={diff:.3e} | "
            f"value diff={value_diff:.3e}"
        )

    print()
    print(
        "Evaluator equivalence: PASS"
    )

    print(
        "Maximum policy difference:",
        f"{max_policy_diff:.3e}",
    )

    print(
        "Maximum value difference:",
        f"{max_value_diff:.3e}",
    )


# ============================================================
# TEST 2: MCTS SEARCH EQUIVALENCE
# ============================================================

def make_original_mcts(
    model,
    simulations,
):
    return OriginalMCTS(
        simulations=simulations,
        rollout_type="neural",
        selection_type="puct",
        model=model,
        c_puct=3.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        adaptive_simulations=True,
        min_simulations=min(
            80,
            simulations,
        ),
        check_interval=20,
        target_visits_per_action=20.0,
        single_action_simulations=4,
        stability_checks=3,
    )


def make_direct_mcts(
    model,
    simulations,
):
    evaluator = (
        DirectNeuralEvaluator(
            model=model,
        )
    )

    return DirectMCTS(
        simulations=simulations,
        rollout_type="neural",
        selection_type="puct",
        evaluator=evaluator,
        c_puct=3.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        adaptive_simulations=True,
        min_simulations=min(
            80,
            simulations,
        ),
        check_interval=20,
        target_visits_per_action=20.0,
        single_action_simulations=4,
        stability_checks=3,
    )


def test_mcts_equivalence(
    model,
    states,
    simulations,
    test_root_noise,
    rng_seed,
):
    print()
    print("=" * 72)
    print("TEST 2: MCTS V5 SEARCH EQUIVALENCE")
    print("=" * 72)

    for index, state in enumerate(
        states,
        start=1,
    ):
        env_original = (
            SplendorEnv()
        )

        env_direct = (
            SplendorEnv()
        )

        original_mcts = (
            make_original_mcts(
                model=model,
                simulations=simulations,
            )
        )

        direct_mcts = (
            make_direct_mcts(
                model=model,
                simulations=simulations,
            )
        )

        # Only relevant if root noise is enabled.
        # Match RNG streams exactly.
        original_mcts.rng = (
            np.random.default_rng(
                rng_seed + index
            )
        )

        direct_mcts.rng = (
            np.random.default_rng(
                rng_seed + index
            )
        )

        action_original, root_original = (
            original_mcts.search(
                env=env_original,
                state=state.clone(),
                root=None,
                return_root=True,
                debug=False,
                add_root_noise=(
                    test_root_noise
                ),
                teacher_mode=False,
            )
        )

        action_direct, root_direct = (
            direct_mcts.search(
                env=env_direct,
                state=state.clone(),
                root=None,
                return_root=True,
                debug=False,
                add_root_noise=(
                    test_root_noise
                ),
                teacher_mode=False,
            )
        )

        key_original = action_key(
            env_original,
            action_original,
        )

        key_direct = action_key(
            env_direct,
            action_direct,
        )

        if key_original != key_direct:
            raise AssertionError(
                f"State {index} chosen-action "
                f"mismatch: "
                f"{key_original!r} != "
                f"{key_direct!r}"
            )

        compare_roots(
            env_original,
            root_original,
            env_direct,
            root_direct,
        )

        compare_metadata(
            original_mcts.last_search_metadata,
            direct_mcts.last_search_metadata,
            prefix=(
                f"state[{index}]"
                ".search_metadata"
            ),
        )

        metadata = (
            original_mcts
            .last_search_metadata
        )

        print(
            f"State {index:>2}: PASS | "
            f"action={key_original} | "
            f"legal="
            f"{metadata['num_legal_actions']} | "
            f"sims="
            f"{metadata['actual_simulations']} | "
            f"root_visits="
            f"{metadata['final_root_visits']} | "
            f"stop="
            f"{metadata['stop_reason']}"
        )

    print()
    print(
        "MCTS search equivalence: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Confirm that DirectNeuralEvaluator and the "
            "evaluator-injected MCTS V5 fork preserve "
            "the original V5 behavior."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--num-evaluator-states",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-mcts-states",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--state-stride",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help=(
            "MCTS hard limit for the smoke test. "
            "Use 400 for a full-budget confirmation."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=31000,
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-7,
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--test-root-noise",
        action="store_true",
        help=(
            "Also enable root Dirichlet noise. The two "
            "MCTS instances receive identical RNG seeds."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_evaluator_states < 1:
        raise ValueError(
            "--num-evaluator-states must be >= 1"
        )

    if args.num_mcts_states < 1:
        raise ValueError(
            "--num-mcts-states must be >= 1"
        )

    if args.simulations < 1:
        raise ValueError(
            "--simulations must be >= 1"
        )

    num_states = max(
        args.num_evaluator_states,
        args.num_mcts_states,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print(
        "DIRECT NEURAL EVALUATOR "
        "SMOKE TEST"
    )
    print("=" * 72)

    print(
        "Device:",
        device,
    )

    print(
        "Checkpoint:",
        args.checkpoint,
    )

    print(
        "Evaluator states:",
        args.num_evaluator_states,
    )

    print(
        "MCTS states:",
        args.num_mcts_states,
    )

    print(
        "MCTS simulations:",
        args.simulations,
    )

    print(
        "Root noise:",
        args.test_root_noise,
    )

    model = load_model(
        checkpoint_path=(
            args.checkpoint
        ),
        device=device,
    )

    states = sample_reachable_states(
        num_states=num_states,
        seed=args.seed,
        stride=max(
            1,
            args.state_stride,
        ),
    )

    test_evaluator_equivalence(
        model=model,
        states=states[
            :args.num_evaluator_states
        ],
        atol=args.atol,
        rtol=args.rtol,
    )

    test_mcts_equivalence(
        model=model,
        states=states[
            :args.num_mcts_states
        ],
        simulations=(
            args.simulations
        ),
        test_root_noise=(
            args.test_root_noise
        ),
        rng_seed=args.seed + 100000,
    )

    print()
    print("=" * 72)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 72)

    print(
        "The DirectNeuralEvaluator abstraction "
        "preserved the tested V5 neural and "
        "MCTS behavior."
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print()
        print("=" * 72)
        print("SMOKE TEST FAILED")
        print("=" * 72)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise
