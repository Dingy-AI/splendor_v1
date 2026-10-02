"""Checkpoint-compatible math, CPU validation, packing, buckets, and precision."""
from pathlib import Path
import hashlib
import numpy as np
import pytest
import torch

from splendor_v1.network.model_4_legal_scorer import SplendorNetwork
from splendor_v1.rust_engine.mcts import Model4Evaluator
from splendor_v1.rust_engine_v2.inference import (
    InferenceNetwork, InferenceOptions, PackedModel4Evaluator, load_model,
)
from splendor_v1.rust_engine_v2.profile_inference import positions, pack


@pytest.fixture(scope="module")
def models():
    torch.set_num_threads(1)
    checkpoint = Path(__file__).parents[1] / "training_v6/data/model4_v6_inference_latest.pt"
    native = load_model(checkpoint, "cpu")
    reference = SplendorNetwork().eval()
    reference.load_state_dict(native.state_dict())
    assert reference.state_dict().keys() == native.state_dict().keys()
    return reference, native


@pytest.mark.parametrize("rows", [1, 3, 8])
def test_fp32_matches_existing_model_and_evaluator(models, rows):
    reference, native = models
    requests = positions(rows, 10000)
    expected = Model4Evaluator(reference).evaluate_batch(requests)
    actual = PackedModel4Evaluator(native).evaluate_packed(*pack(requests))
    for row, ((policy, value), (_, ids)) in enumerate(zip(expected, requests)):
        np.testing.assert_allclose(actual[row, :len(ids)], policy, rtol=0, atol=1e-7)
        assert float(actual[row, -1]) == pytest.approx(value, rel=0, abs=1e-7)
    assert actual.dtype == np.float32 and np.isfinite(actual).all()


def test_buckets_and_reused_buffers_preserve_predictions(models):
    _, model = models
    arrays = pack(positions(3, 10001))
    expected = PackedModel4Evaluator(model).evaluate_packed(*arrays)
    evaluator = PackedModel4Evaluator(model, InferenceOptions(bucket_shapes=True))
    first = evaluator.evaluate_packed(*arrays)
    held = first.copy()
    evaluator.evaluate_packed(*pack(positions(3, 10004)))
    np.testing.assert_array_equal(first, held)  # Caller owns the returned predictions.
    np.testing.assert_allclose(first, expected, rtol=0, atol=1e-6)
    assert evaluator.profile()["padded_positions"] == 2


@pytest.mark.parametrize("bad", ["negative_id", "large_id", "empty_mask", "nan", "shape", "mask_dtype"])
def test_checks_inputs_before_model_execution(models, bad, monkeypatch):
    _, model = models
    obs, ids, mask = [a.copy() for a in pack(positions(2, 10000))]
    if bad == "negative_id": ids[0, 0] = -1
    if bad == "large_id": ids[0, 0] = 1139
    if bad == "empty_mask": mask[0] = False
    if bad == "nan": obs[0, 0] = np.nan
    if bad == "shape": obs = obs[:, :-1]
    if bad == "mask_dtype": mask = mask.astype(np.int32)
    def fail(*args, **kwargs): raise AssertionError("Invalid inputs reached the network")
    monkeypatch.setattr(model, "forward_legal", fail)
    with pytest.raises(ValueError): PackedModel4Evaluator(model).evaluate_packed(obs, ids, mask)


def test_cpu_validation_avoids_gpu_predicate_method(models, monkeypatch):
    _, model = models
    def fail(*args, **kwargs): raise AssertionError("General-purpose GPU validator was called")
    monkeypatch.setattr(SplendorNetwork, "_prepare_legal_action_inputs", fail)
    evaluator = PackedModel4Evaluator(model)
    result = evaluator.evaluate_packed(*pack(positions(2, 10000)))
    assert np.isfinite(result).all()


def test_cpu_bf16_is_finite_and_close_to_fp32(models):
    _, model = models
    arrays = pack(positions(8, 10000))
    baseline = PackedModel4Evaluator(model).evaluate_packed(*arrays)
    actual = PackedModel4Evaluator(model, InferenceOptions(precision="bf16")).evaluate_packed(*arrays)
    np.testing.assert_allclose(actual, baseline, rtol=0, atol=0.015)
    np.testing.assert_allclose(actual[:, :-1].sum(axis=1), 1, rtol=0, atol=1e-6)


def test_options_and_cpu_fp16_rejected(models):
    _, model = models
    for options in (InferenceOptions(precision="fp8"), InferenceOptions(compile_mode="invalid"),
                    InferenceOptions(precision="fp16")):
        with pytest.raises(ValueError): PackedModel4Evaluator(model, options)


def test_profile_counts_match_workload(models):
    _, model = models
    evaluator = PackedModel4Evaluator(model, InferenceOptions(profile=True))
    for rows in [2, 3]: evaluator.evaluate_packed(*pack(positions(rows, 10000)))
    profile = evaluator.profile()
    assert profile["total_batches"] == 2 and profile["total_inference_positions"] == 5
    assert profile["average_batch_size"] == 2.5 and profile["max_observed_batch_size"] == 3
    phases = sum(profile[k] for k in ["host_prepare_seconds", "host_transfer_enqueue_seconds",
                                    "host_forward_dispatch_seconds", "host_readback_seconds"])
    assert phases == pytest.approx(profile["total_inference_seconds"], rel=0, abs=1e-9)
    assert profile["cuda_event_profiling"] is False
    evaluator.reset_profile()
    assert evaluator.profile()["total_batches"] == 0


def test_loaded_checkpoint_identity_is_retained(tmp_path):
    original = Path(__file__).parents[1] / "training_v6/data/model4_v6_inference_latest.pt"
    path = tmp_path / "snapshot.pt"
    payload = original.read_bytes()
    path.write_bytes(payload)
    model = load_model(path, "cpu")
    path.write_bytes(b"a later replacement")
    assert model.checkpoint_sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_cuda_precision_and_events(models, precision):
    _, original = models
    model = InferenceNetwork().cuda().eval()
    model.load_state_dict(original.state_dict())
    if precision == "bf16" and not torch.cuda.is_bf16_supported(): pytest.skip("BF16 unsupported")
    arrays = pack(positions(8, 10000))
    expected = PackedModel4Evaluator(model).evaluate_packed(*arrays)
    evaluator = PackedModel4Evaluator(model, InferenceOptions(precision=precision, profile=True))
    actual = evaluator.evaluate_packed(*arrays)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.015 if precision != "fp32" else 1e-7)
    assert evaluator.profile()["gpu_forward_seconds"] > 0


def test_inference_profiler_cli_cpu(tmp_path):
    from splendor_v1.rust_engine_v2.profile_inference import main
    path = tmp_path / "profile.json"
    checkpoint = Path(__file__).parents[1] / "training_v6/data/model4_v6_inference_latest.pt"
    report = main(["--checkpoint", str(checkpoint), "--device", "cpu", "--batch-sizes", "2", "3", "--precisions", "fp32", "bf16",
                   "--warmup", "1", "--repeats", "2", "--report", str(path)])
    assert path.exists() and len(report["results"]) == 4
    assert all(row["status"] == "ok" for row in report["results"])
