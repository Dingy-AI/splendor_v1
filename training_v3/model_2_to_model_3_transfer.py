import torch

from splendor_v1.network.model_3_wdl_output import SplendorNetwork


def _extract_model_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected checkpoint to be a dictionary."
        )

    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return checkpoint

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, dict):
            return state_dict

    raise KeyError(
        "Could not find a model state_dict in the checkpoint. "
        "Expected a raw state_dict or one of: "
        "'model_state_dict', 'state_dict', 'model'."
    )


def load_model2_weights_into_model3(
    model3,
    checkpoint_path,
    map_location="cpu",
    require_policy_head=True,
    verbose=True,
):
    """
    Load all name+shape compatible Model 2 weights into Model 3.

    Explicitly skips Model 2's scalar value_head and leaves
    Model 3's new wdl_head at its fresh random initialization.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    model2_state = _extract_model_state_dict(
        checkpoint
    )

    model3_state = model3.state_dict()

    copied = []
    skipped_old_value_head = []
    skipped_missing_in_model3 = []
    skipped_shape_mismatch = []
    transfer_state = {}

    for key, source_tensor in model2_state.items():

        if key.startswith("value_head."):
            skipped_old_value_head.append(key)
            continue

        if key not in model3_state:
            skipped_missing_in_model3.append(key)
            continue

        target_tensor = model3_state[key]

        if source_tensor.shape != target_tensor.shape:
            skipped_shape_mismatch.append(
                {
                    "key": key,
                    "source_shape": tuple(source_tensor.shape),
                    "target_shape": tuple(target_tensor.shape),
                }
            )
            continue

        transfer_state[key] = source_tensor
        copied.append(key)

    load_result = model3.load_state_dict(
        transfer_state,
        strict=False,
    )

    missing_after_load = list(
        load_result.missing_keys
    )

    unexpected_after_load = list(
        load_result.unexpected_keys
    )

    wdl_keys = [
        key
        for key in model3_state
        if key.startswith("wdl_head.")
    ]

    wdl_loaded = [
        key
        for key in copied
        if key.startswith("wdl_head.")
    ]

    if wdl_loaded:
        raise RuntimeError(
            "WDL head was unexpectedly loaded from Model 2: "
            f"{wdl_loaded}"
        )

    policy_keys = [
        key
        for key in model3_state
        if key.startswith("policy_head.")
    ]

    copied_policy_keys = [
        key
        for key in copied
        if key.startswith("policy_head.")
    ]

    if (
        require_policy_head
        and set(copied_policy_keys) != set(policy_keys)
    ):
        raise RuntimeError(
            "Policy head was not completely transferred. "
            f"Expected {policy_keys}, copied {copied_policy_keys}."
        )

    unexpected_missing = [
        key
        for key in missing_after_load
        if not key.startswith("wdl_head.")
    ]

    if unexpected_missing:
        raise RuntimeError(
            "Some Model 3 parameters were not initialized from "
            "Model 2 and are not part of the WDL head: "
            f"{unexpected_missing}"
        )

    if unexpected_after_load:
        raise RuntimeError(
            "Unexpected keys were produced while loading Model 3: "
            f"{unexpected_after_load}"
        )

    report = {
        "checkpoint_path": str(checkpoint_path),
        "copied_count": len(copied),
        "copied_keys": copied,
        "copied_policy_keys": copied_policy_keys,
        "skipped_old_value_head": skipped_old_value_head,
        "skipped_missing_in_model3": skipped_missing_in_model3,
        "skipped_shape_mismatch": skipped_shape_mismatch,
        "fresh_wdl_keys": wdl_keys,
        "missing_after_load": missing_after_load,
        "unexpected_after_load": unexpected_after_load,
    }

    if verbose:
        print()
        print("=" * 70)
        print("MODEL 2 -> MODEL 3 WEIGHT TRANSFER")
        print("=" * 70)
        print("Checkpoint:", checkpoint_path)
        print("Copied parameters:", len(copied))
        print("Policy head copied:", copied_policy_keys)
        print("Skipped Model 2 value head:", skipped_old_value_head)
        print("Fresh Model 3 WDL head:", wdl_keys)

        if skipped_missing_in_model3:
            print(
                "Other checkpoint keys absent from Model 3:",
                skipped_missing_in_model3,
            )

        if skipped_shape_mismatch:
            print(
                "Shape mismatches:",
                skipped_shape_mismatch,
            )

        print()
        print("Transfer verification: PASS")
        print("=" * 70)
        print()

    return report


def create_model3_from_model2_checkpoint(
    checkpoint_path,
    device="cpu",
    **model_kwargs,
):
    """
    Construct Model 3, load the compatible Model 2 weights,
    leave wdl_head fresh, then move the model to `device`.
    """

    model = SplendorNetwork(
        **model_kwargs
    )

    report = load_model2_weights_into_model3(
        model3=model,
        checkpoint_path=checkpoint_path,
        map_location="cpu",
        require_policy_head=True,
        verbose=True,
    )

    model = model.to(device)

    return model, report


if __name__ == "__main__":

    CHECKPOINT_PATH = (
        "checkpoints/gen_2/gen_2_h12.pt"
    )

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, report = (
        create_model3_from_model2_checkpoint(
            checkpoint_path=CHECKPOINT_PATH,
            device=DEVICE,
        )
    )

    print(
        "Model device:",
        next(model.parameters()).device,
    )

    print(
        "Transferred parameters:",
        report["copied_count"],
    )


    import os

    OUTPUT_PATH = (
        "checkpoints/gen_3/"
        "gen_3_wdl_transfer_g2h12.pt"
    )

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "model_generation":
                3,

            "source_generation":
                2,

            "source_checkpoint":
                CHECKPOINT_PATH,

            "transfer_type":
                "model2_to_model3_wdl",

            "transferred_parameters":
                report["copied_count"],
        },
        OUTPUT_PATH,
    )

    print(
        "Model 3 saved:",
        OUTPUT_PATH,
    )