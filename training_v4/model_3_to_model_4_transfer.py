import argparse
import os
from pathlib import Path

import torch

from splendor_v1.network.model_3_wdl_output import (
    SplendorNetwork as Model3Network,
)
from splendor_v1.network.model_4_legal_scorer import (
    SplendorNetwork as Model4Network,
)


MODEL4_FRESH_PREFIXES = (
    "action_embedding.",
    "legal_action_scorer.",
)

MODEL3_DISCARDED_PREFIXES = (
    "policy_head.",
)


def _extract_model_state_dict(
    checkpoint,
):
    """
    Accept either:
        - a raw PyTorch state_dict
        - a checkpoint dictionary containing one of the common
          state-dict keys used in this project
    """

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Expected checkpoint to be a dictionary."
        )

    if (
        checkpoint
        and all(
            torch.is_tensor(
                value
            )
            for value
            in checkpoint.values()
        )
    ):
        state_dict = checkpoint

    else:

        state_dict = None

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):

            candidate = checkpoint.get(
                key
            )

            if isinstance(
                candidate,
                dict,
            ):

                state_dict = candidate
                break

        if state_dict is None:

            raise KeyError(
                "Could not find a model state_dict in "
                "the checkpoint. Expected a raw state_dict "
                "or one of: 'model_state_dict', "
                "'state_dict', 'model'."
            )

    # Optional DataParallel compatibility.
    keys = list(
        state_dict.keys()
    )

    if (
        keys
        and all(
            key.startswith(
                "module."
            )
            for key in keys
        )
    ):

        state_dict = {
            key[
                len("module.") :
            ]:
                value
            for (
                key,
                value,
            )
            in state_dict.items()
        }

    return state_dict


def _tensor_element_count(
    state_dict,
    keys,
):
    return sum(
        int(
            state_dict[
                key
            ].numel()
        )
        for key
        in keys
    )


def load_model3_checkpoint(
    checkpoint_path,
    map_location="cpu",
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    state_dict = (
        _extract_model_state_dict(
            checkpoint
        )
    )

    model3 = Model3Network()

    model3.load_state_dict(
        state_dict,
        strict=True,
    )

    model3.eval()

    return (
        model3,
        checkpoint,
        state_dict,
    )


def transfer_model3_to_model4(
    model3_state,
    model4,
    verbose=True,
):
    """
    Transfer all name+shape compatible Model 3 parameters into
    Model 4.

    Expected behavior:

    COPIED:
        - all observation/entity encoders
        - all learned context embeddings
        - state token / hidden reserved embedding
        - all attention blocks
        - final normalization
        - trained WDL head

    DISCARDED FROM MODEL 3:
        - policy_head.*

    FRESH IN MODEL 4:
        - action_embedding.*
        - legal_action_scorer.*

    Any other missing or shape-mismatched parameter is treated as
    an error rather than silently producing a partially initialized
    model.
    """

    model4_state = (
        model4.state_dict()
    )

    copied = []

    discarded_model3_policy = []

    unexpected_source_only = []

    shape_mismatches = []

    transfer_state = {}

    # ========================================================
    # COPY NAME + SHAPE COMPATIBLE WEIGHTS
    # ========================================================

    for (
        key,
        source_tensor,
    ) in model3_state.items():

        if key.startswith(
            MODEL3_DISCARDED_PREFIXES
        ):

            discarded_model3_policy.append(
                key
            )

            continue

        if key not in model4_state:

            unexpected_source_only.append(
                key
            )

            continue

        target_tensor = (
            model4_state[
                key
            ]
        )

        if (
            source_tensor.shape
            != target_tensor.shape
        ):

            shape_mismatches.append(
                {
                    "key":
                        key,

                    "model3_shape":
                        tuple(
                            source_tensor.shape
                        ),

                    "model4_shape":
                        tuple(
                            target_tensor.shape
                        ),
                }
            )

            continue

        transfer_state[
            key
        ] = source_tensor

        copied.append(
            key
        )

    # ========================================================
    # FAIL ON UNEXPECTED DIFFERENCES
    # ========================================================

    if unexpected_source_only:

        raise RuntimeError(
            "Model 3 contains parameters that do not "
            "exist in Model 4 and are not the expected "
            "old policy_head: "
            f"{unexpected_source_only}"
        )

    if shape_mismatches:

        raise RuntimeError(
            "Shared Model 3 / Model 4 parameters have "
            "shape mismatches. This means the shared "
            "architecture changed unexpectedly: "
            f"{shape_mismatches}"
        )

    # ========================================================
    # LOAD TRANSFER
    # ========================================================

    load_result = (
        model4.load_state_dict(
            transfer_state,
            strict=False,
        )
    )

    missing_after_load = list(
        load_result.missing_keys
    )

    unexpected_after_load = list(
        load_result.unexpected_keys
    )

    if unexpected_after_load:

        raise RuntimeError(
            "Unexpected keys appeared while loading "
            "Model 4: "
            f"{unexpected_after_load}"
        )

    # ========================================================
    # VERIFY ONLY NEW MODEL 4 POLICY COMPONENTS ARE FRESH
    # ========================================================

    fresh_model4_keys = [
        key
        for key
        in model4_state
        if key.startswith(
            MODEL4_FRESH_PREFIXES
        )
    ]

    unexpected_missing = [
        key
        for key
        in missing_after_load
        if not key.startswith(
            MODEL4_FRESH_PREFIXES
        )
    ]

    if unexpected_missing:

        raise RuntimeError(
            "Model 4 has parameters that were not "
            "transferred and are not part of the new "
            "legal-action policy components: "
            f"{unexpected_missing}"
        )

    if (
        set(
            missing_after_load
        )
        != set(
            fresh_model4_keys
        )
    ):

        raise RuntimeError(
            "Fresh Model 4 key set does not exactly match "
            "the expected legal-action scorer components.\n"
            f"Expected fresh: {fresh_model4_keys}\n"
            f"Actually missing: {missing_after_load}"
        )

    # ========================================================
    # VERIFY OLD POLICY HEAD WAS REALLY DISCARDED
    # ========================================================

    expected_old_policy_keys = [
        key
        for key
        in model3_state
        if key.startswith(
            MODEL3_DISCARDED_PREFIXES
        )
    ]

    if (
        set(
            discarded_model3_policy
        )
        != set(
            expected_old_policy_keys
        )
    ):

        raise RuntimeError(
            "Model 3 policy-head discard verification "
            "failed."
        )

    if not expected_old_policy_keys:

        raise RuntimeError(
            "No Model 3 policy_head parameters were "
            "found. Are you loading the expected "
            "Model 3 architecture?"
        )

    # ========================================================
    # VERIFY WDL HEAD TRANSFERRED COMPLETELY
    # ========================================================

    model3_wdl_keys = [
        key
        for key
        in model3_state
        if key.startswith(
            "wdl_head."
        )
    ]

    copied_wdl_keys = [
        key
        for key
        in copied
        if key.startswith(
            "wdl_head."
        )
    ]

    if (
        set(
            copied_wdl_keys
        )
        != set(
            model3_wdl_keys
        )
    ):

        raise RuntimeError(
            "Model 3 WDL head was not completely "
            "transferred to Model 4.\n"
            f"Expected: {model3_wdl_keys}\n"
            f"Copied: {copied_wdl_keys}"
        )

    # ========================================================
    # PARAMETER / TENSOR STATISTICS
    # ========================================================

    copied_elements = (
        _tensor_element_count(
            model4_state,
            copied,
        )
    )

    fresh_elements = (
        _tensor_element_count(
            model4_state,
            fresh_model4_keys,
        )
    )

    model4_total_elements = (
        sum(
            int(
                tensor.numel()
            )
            for tensor
            in model4_state.values()
        )
    )

    discarded_policy_elements = (
        _tensor_element_count(
            model3_state,
            discarded_model3_policy,
        )
    )

    transfer_fraction = (
        copied_elements
        / model4_total_elements
        if model4_total_elements > 0
        else 0.0
    )

    report = {
        "copied_key_count":
            len(
                copied
            ),

        "copied_keys":
            copied,

        "copied_elements":
            copied_elements,

        "fresh_model4_keys":
            fresh_model4_keys,

        "fresh_model4_elements":
            fresh_elements,

        "model4_total_elements":
            model4_total_elements,

        "transfer_fraction_by_elements":
            transfer_fraction,

        "discarded_model3_policy_keys":
            discarded_model3_policy,

        "discarded_model3_policy_elements":
            discarded_policy_elements,

        "copied_wdl_keys":
            copied_wdl_keys,

        "shape_mismatches":
            shape_mismatches,

        "missing_after_load":
            missing_after_load,

        "unexpected_after_load":
            unexpected_after_load,
    }

    if verbose:

        print()
        print(
            "=" * 72
        )

        print(
            "MODEL 3 -> MODEL 4 WEIGHT TRANSFER"
        )

        print(
            "=" * 72
        )

        print(
            "Copied state-dict tensors:",
            len(
                copied
            ),
        )

        print(
            "Copied tensor elements:",
            f"{copied_elements:,}",
        )

        print(
            "Fresh Model 4 tensors:",
            len(
                fresh_model4_keys
            ),
        )

        print(
            "Fresh Model 4 elements:",
            f"{fresh_elements:,}",
        )

        print(
            "Model 4 total elements:",
            f"{model4_total_elements:,}",
        )

        print(
            "Model 4 initialized from Model 3:",
            f"{transfer_fraction:.2%}",
        )

        print()

        print(
            "Transferred WDL head:",
            copied_wdl_keys,
        )

        print()

        print(
            "Discarded Model 3 flat policy head:",
            discarded_model3_policy,
        )

        print(
            "Discarded policy elements:",
            f"{discarded_policy_elements:,}",
        )

        print()

        print(
            "Fresh Model 4 policy components:"
        )

        for key in fresh_model4_keys:

            print(
                "  ",
                key,
                tuple(
                    model4_state[
                        key
                    ].shape
                ),
            )

        print()
        print(
            "Structural transfer verification: PASS"
        )

        print(
            "=" * 72
        )
        print()

    return report


def verify_shared_outputs(
    model3,
    model4,
    device="cpu",
):
    """
    Strong functional verification.

    Because Model 4 keeps the Model 3 state trunk and WDL head,
    the same observation must produce identical:

        - contextualized state features
        - WDL logits

    immediately after transfer.

    The policy output is intentionally NOT comparable because
    Model 4 has a completely new policy mechanism.
    """

    model3 = model3.to(
        device
    )

    model4 = model4.to(
        device
    )

    model3.eval()
    model4.eval()

    # Use simple deterministic probe observations.
    #
    # The node-type slice gets a valid one-hot value so the
    # STATE token receives a normal node encoding.
    probe = torch.zeros(
        4,
        258,
        dtype=torch.float32,
        device=device,
    )

    probe[
        :,
        255
    ] = 1.0

    with torch.inference_mode():

        model3_features = (
            model3._forward_features(
                probe
            )
        )

        model4_features = (
            model4._forward_features(
                probe
            )
        )

        model3_policy, model3_wdl = (
            model3(
                probe
            )
        )

        model4_wdl = (
            model4.forward_wdl(
                probe
            )
        )

    if not torch.allclose(
        model3_features,
        model4_features,
        rtol=1e-6,
        atol=1e-7,
    ):

        max_difference = (
            model3_features
            - model4_features
        ).abs().max().item()

        raise RuntimeError(
            "Transferred Model 4 state features do "
            "not exactly match Model 3. "
            f"Max difference: {max_difference}"
        )

    if not torch.allclose(
        model3_wdl,
        model4_wdl,
        rtol=1e-6,
        atol=1e-7,
    ):

        max_difference = (
            model3_wdl
            - model4_wdl
        ).abs().max().item()

        raise RuntimeError(
            "Transferred Model 4 WDL logits do not "
            "exactly match Model 3. "
            f"Max difference: {max_difference}"
        )

    if (
        model3_policy.shape[-1]
        != model3.action_space_size
    ):

        raise RuntimeError(
            "Unexpected Model 3 policy output shape."
        )

    print(
        "Shared feature verification: PASS"
    )

    print(
        "Shared WDL verification:     PASS"
    )


def create_model4_from_model3_checkpoint(
    checkpoint_path,
    device="cpu",
    **model4_kwargs,
):
    """
    Construct Model 4, transfer the compatible Model 3 weights,
    verify the shared representation/WDL behavior, and return the
    initialized Model 4.
    """

    (
        model3,
        checkpoint,
        model3_state,
    ) = load_model3_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        map_location="cpu",
    )

    model4 = Model4Network(
        **model4_kwargs
    )

    report = (
        transfer_model3_to_model4(
            model3_state=(
                model3_state
            ),
            model4=model4,
            verbose=True,
        )
    )

    verify_shared_outputs(
        model3=model3,
        model4=model4,
        device=device,
    )

    model4 = model4.to(
        device
    )

    model4.eval()

    return (
        model4,
        report,
        checkpoint,
    )


def save_model4_checkpoint(
    model4,
    output_path,
    source_checkpoint,
    report,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model_state_dict":
            model4.state_dict(),

        "model_generation":
            4,

        "architecture":
            "attention_wdl_legal_action_scorer",

        "source_generation":
            3,

        "source_checkpoint":
            str(
                source_checkpoint
            ),

        "transfer_type":
            "model3_to_model4_legal_action_scorer",

        "transfer_report": {
            "copied_key_count":
                report[
                    "copied_key_count"
                ],

            "copied_elements":
                report[
                    "copied_elements"
                ],

            "fresh_model4_keys":
                report[
                    "fresh_model4_keys"
                ],

            "fresh_model4_elements":
                report[
                    "fresh_model4_elements"
                ],

            "model4_total_elements":
                report[
                    "model4_total_elements"
                ],

            "transfer_fraction_by_elements":
                report[
                    "transfer_fraction_by_elements"
                ],

            "discarded_model3_policy_keys":
                report[
                    "discarded_model3_policy_keys"
                ],

            "discarded_model3_policy_elements":
                report[
                    "discarded_model3_policy_elements"
                ],

            "copied_wdl_keys":
                report[
                    "copied_wdl_keys"
                ],
        },
    }

    temp_path = Path(
        str(
            output_path
        )
        + ".tmp"
    )

    torch.save(
        payload,
        temp_path,
    )

    os.replace(
        temp_path,
        output_path,
    )

    print()

    print(
        "Model 4 checkpoint saved:"
    )

    print(
        " ",
        output_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transfer Model 3's attention trunk and "
            "WDL head into Model 4 while replacing "
            "the flat policy head with a fresh "
            "legal-action scorer."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/gen_3/"
            "gen_3_wdl_head_warmup_best.pt"
        ),
        help=(
            "Source Model 3 checkpoint."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "checkpoints/gen_4/"
            "gen_4_legal_scorer_transfer_g3.pt"
        ),
        help=(
            "Output Model 4 checkpoint."
        ),
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Source Model 3 checkpoint "
            f"not found: {checkpoint_path}"
        )

    print()
    print(
        "=" * 72
    )

    print(
        "CREATE MODEL 4 FROM MODEL 3"
    )

    print(
        "=" * 72
    )

    print(
        "Source:",
        checkpoint_path,
    )

    print(
        "Output:",
        args.output,
    )

    print(
        "Device:",
        args.device,
    )

    (
        model4,
        report,
        _,
    ) = create_model4_from_model3_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        device=args.device,
    )

    save_model4_checkpoint(
        model4=model4,
        output_path=args.output,
        source_checkpoint=(
            checkpoint_path
        ),
        report=report,
    )

    print()
    print(
        "Transfer complete."
    )

    print(
        "Model 4 device:",
        next(
            model4.parameters()
        ).device,
    )

    print(
        "Transferred fraction:",
        (
            f"{report['transfer_fraction_by_elements']:.2%}"
        ),
    )


if __name__ == "__main__":
    main()
