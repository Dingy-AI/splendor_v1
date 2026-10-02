"""Checkpoint-compatible Model 4 inference with one packed result readback.

Native search inputs are checked on the CPU, avoiding scalar CUDA reads in
Model 4's general-purpose legal-input validator. Training still uses Model 4.
"""
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import time

import numpy as np
import torch

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model_4_legal_scorer import SplendorNetwork


@dataclass(frozen=True)
class InferenceOptions:
    precision: str = "fp32"
    compile_mode: str = "off"
    profile: bool = False
    bucket_shapes: bool = False

    def validate(self):
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.compile_mode not in ("off", "default", "reduce-overhead"):
            raise ValueError("compile_mode must be off, default, or reduce-overhead")


class InferenceNetwork(SplendorNetwork):
    """Identical parameters/math; the evaluator validates IDs before CUDA copy."""

    def _prepare_legal_action_inputs(self, legal_action_ids, legal_action_mask,
                                    batch_size, device):
        # Private inference boundary: callers must use PackedModel4Evaluator.
        # Skipping the four GPU -> Python predicates keeps this graph capturable.
        return legal_action_ids, legal_action_mask

    def forward_legal(self, x, legal_action_ids, legal_action_mask=None):
        features = self._forward_features(x)
        # Policy logits can be close together at a large common offset. Keep
        # both output heads in FP32 so BF16 rounding cannot flatten that policy.
        with torch.autocast(device_type=x.device.type, enabled=False):
            features = features.float()
            logits = self._score_legal_actions(features, legal_action_ids, legal_action_mask)
            wdl_logits = self.wdl_head(features)
        return logits, wdl_logits


def load_model(checkpoint_path, device):
    payload = Path(checkpoint_path).read_bytes()
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    weights = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model = InferenceNetwork()
    model.load_state_dict(weights)
    model.checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
    return model.to(device).eval()


class PackedModel4Evaluator:
    """Evaluate already packed [B,258], [B,A] IDs/masks; return [B,A+1] float32.

    The last column is player-to-move P(WIN)-P(LOSS). CPU buffers are reused
    (pinned on CUDA); all policies and values return in one blocking readback.
    Default fp32/eager preserves weights and normal Model 4 arithmetic.
    """

    def __init__(self, model, options=None):
        self.options = options or InferenceOptions()
        self.options.validate()
        if model.training:
            raise ValueError("Inference model must be in eval mode")
        self.model = model
        self.device = model.action_embedding.weight.device
        self.dtype = model.action_embedding.weight.dtype
        if self.dtype != torch.float32:
            raise ValueError("Keep checkpoint weights in float32; select precision via autocast")
        if self.options.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 inference requires CUDA; use fp32 or bf16 for CPU checks")
        if (self.options.precision == "bf16" and self.device.type == "cuda"
                and not torch.cuda.is_bf16_supported()):
            raise ValueError("This CUDA device does not support bf16")
        self._forward = model.forward_legal
        if self.options.compile_mode != "off":
            self._forward = torch.compile(self._forward, mode=self.options.compile_mode)
        self._buffers = OrderedDict()
        self.reset_profile()

    def reset_profile(self):
        self.totals = dict(total_batches=0, total_inference_positions=0,
            max_observed_batch_size=0, padded_positions=0, padded_action_cells=0,
            total_action_cells=0, total_inference_seconds=0.0,
            host_prepare_seconds=0.0, host_transfer_enqueue_seconds=0.0,
            host_forward_dispatch_seconds=0.0, host_readback_seconds=0.0,
            gpu_input_transfer_seconds=0.0, gpu_forward_seconds=0.0,
            gpu_postprocess_seconds=0.0, gpu_readback_seconds=0.0)

    def profile(self):
        result = dict(self.totals)
        rows, batches = result["total_inference_positions"], result["total_batches"]
        result.update(backend="rust_arena_v2", precision=self.options.precision,
            output_heads_precision="fp32" if isinstance(self.model, InferenceNetwork) else "model_default",
            compile_mode=self.options.compile_mode, bucket_shapes=self.options.bucket_shapes,
            cuda_event_profiling=self.options.profile and self.device.type == "cuda",
            average_batch_size=rows / batches if batches else 0.0,
            inference_positions_per_second=rows / result["total_inference_seconds"]
                if result["total_inference_seconds"] else 0.0)
        if result["cuda_event_profiling"] and result["gpu_forward_seconds"]:
            result["gpu_forward_positions_per_second"] = (
                (rows + result["padded_positions"]) / result["gpu_forward_seconds"])
        return result

    def _buffers_for(self, rows, width):
        key = rows, width
        if key not in self._buffers:
            pinned = self.device.type == "cuda"
            cpu = (
                torch.empty((rows, 258), dtype=torch.float32, pin_memory=pinned),
                torch.empty((rows, width), dtype=torch.long, pin_memory=pinned),
                torch.empty((rows, width), dtype=torch.bool, pin_memory=pinned),
                torch.empty((rows, width + 1), dtype=torch.float32, pin_memory=pinned),
            )
            gpu = tuple(torch.empty(t.shape, dtype=t.dtype, device=self.device) for t in cpu[:3]) if pinned else cpu[:3]
            self._buffers[key] = (*cpu, *gpu)
            if len(self._buffers) > 8:
                self._buffers.popitem(last=False)
        self._buffers.move_to_end(key)
        return self._buffers[key]

    @staticmethod
    def _validate(observations, ids, mask):
        if observations.ndim != 2 or observations.shape[1] != 258 or not len(observations):
            raise ValueError("Expected nonempty [batch,258] observations")
        if (ids.ndim != 2 or ids.shape[0] != len(observations) or ids.shape[1] == 0
                or ids.shape != mask.shape or not np.issubdtype(ids.dtype, np.integer)
                or mask.dtype != np.bool_):
            raise ValueError("Expected matching integer IDs and boolean mask [batch,candidates]")
        if not np.isfinite(observations).all():
            raise ValueError("Observations must be finite")
        if (ids < 0).any() or (ids >= ACTION_SPACE_SIZE).any() or not mask.any(axis=1).all():
            raise ValueError("Invalid canonical legal IDs or empty legal mask")

    def evaluate_packed(self, observations, ids, mask):
        if self.model.training:
            raise ValueError("Inference model must remain in eval mode")
        started = time.perf_counter()
        self._validate(observations, ids, mask)
        rows, width = ids.shape
        padded_rows, padded_width = rows, width
        if self.options.bucket_shapes:
            padded_rows = 1 << (rows - 1).bit_length()
            padded_width = ((width + 15) // 16) * 16
        cpu_obs, cpu_ids, cpu_mask, cpu_output, x, actions, legal_mask = self._buffers_for(padded_rows, padded_width)
        obs_view, id_view, mask_view = cpu_obs.numpy(), cpu_ids.numpy(), cpu_mask.numpy()
        if (padded_rows, padded_width) != (rows, width):
            obs_view.fill(0); id_view.fill(0); mask_view.fill(False)
            mask_view[rows:, 0] = True  # Valid dummy rows prevent empty-mask NaNs.
        np.copyto(obs_view[:rows], observations)
        np.copyto(id_view[:rows, :width], ids)
        np.copyto(mask_view[:rows, :width], mask)
        prepared = time.perf_counter()
        events = None
        stream = None
        if self.options.profile and self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            events[0].record(stream)
        if self.device.type == "cuda":
            x.copy_(cpu_obs, non_blocking=True)
            actions.copy_(cpu_ids, non_blocking=True)
            legal_mask.copy_(cpu_mask, non_blocking=True)
        if events: events[1].record(stream)
        transferred = time.perf_counter()
        autocast = nullcontext() if self.options.precision == "fp32" else torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.options.precision == "fp16" else torch.bfloat16)
        try:
            with torch.inference_mode(), autocast:
                logits, wdl_logits = self._forward(x, actions, legal_mask)
        except Exception as exc:
            if self.options.compile_mode != "off":
                raise RuntimeError("Compiled inference failed on this setup; retry with --compile off. "
                                   f"Original error: {exc}") from exc
            raise
        if events: events[2].record(stream)
        forwarded = time.perf_counter()
        with torch.inference_mode():
            policy = torch.softmax(logits.float(), dim=-1)
            wdl = torch.softmax(wdl_logits.float(), dim=-1)
            packed = torch.cat((policy, (wdl[:, 2] - wdl[:, 0]).unsqueeze(1)), dim=1)
            if events: events[3].record(stream)
            cpu_output.copy_(packed, non_blocking=False)
            if events:
                events[4].record(stream)
                events[4].synchronize()
        # Drop dummy columns; return a contiguous result owned by this call.
        output = np.empty((rows, width + 1), dtype=np.float32)
        np.copyto(output[:, :width], cpu_output.numpy()[:rows, :width])
        np.copyto(output[:, width], cpu_output.numpy()[:rows, padded_width])
        if not np.isfinite(output).all():
            raise ValueError("Inference produced nonfinite policy/value results")
        ended = time.perf_counter()
        stats = self.totals
        stats["total_batches"] += 1
        stats["total_inference_positions"] += rows
        stats["max_observed_batch_size"] = max(stats["max_observed_batch_size"], rows)
        stats["padded_positions"] += padded_rows - rows
        stats["padded_action_cells"] += padded_rows * padded_width - int(mask.sum())
        stats["total_action_cells"] += padded_rows * padded_width
        stats["total_inference_seconds"] += ended - started
        stats["host_prepare_seconds"] += prepared - started
        stats["host_transfer_enqueue_seconds"] += transferred - prepared
        stats["host_forward_dispatch_seconds"] += forwarded - transferred
        stats["host_readback_seconds"] += ended - forwarded
        if events:
            for index, key in enumerate(("gpu_input_transfer_seconds", "gpu_forward_seconds",
                                         "gpu_postprocess_seconds", "gpu_readback_seconds")):
                stats[key] += events[index].elapsed_time(events[index + 1]) / 1000.0
        return output

    def evaluate_batch(self, requests):
        """Compatibility adapter for tests/manual searches; production uses packed batches."""
        if not requests: return []
        width = max(len(ids) for _, ids in requests)
        actions = np.zeros((len(requests), width), dtype=np.int64)
        mask = np.zeros_like(actions, dtype=np.bool_)
        for row, (_, ids) in enumerate(requests):
            actions[row, :len(ids)] = ids
            mask[row, :len(ids)] = True
        output = self.evaluate_packed(np.stack([obs for obs, _ in requests]), actions, mask)
        return [(output[row, :len(ids)], float(output[row, -1]))
                for row, (_, ids) in enumerate(requests)]

    def evaluate(self, observation, legal_ids):
        return self.evaluate_batch([(observation, legal_ids)])[0]
