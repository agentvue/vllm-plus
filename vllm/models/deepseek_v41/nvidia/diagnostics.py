# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, bounded eager diagnostics; never enabled for production graphs."""

import hashlib
import json
import os

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)
CHUNK = 65536


def diagnostic_enabled():
    return os.getenv("VLLM_DEEPSEEK_V41_DIAGNOSTICS", "0") == "1"


def tensor_digest(tensor):
    """Hash logical bytes in bounded CPU staging buffers, including float8."""
    digest = hashlib.sha256()
    # Slice the leading dimension before making a possibly strided tensor dense.
    tensor = tensor.detach()
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    row_bytes = tensor[0].numel() * tensor.element_size() if len(tensor) else 1
    rows = max(1, CHUNK // max(1, row_bytes))
    for offset in range(0, len(tensor), rows):
        data = tensor[offset : offset + rows].contiguous().view(torch.uint8)
        digest.update(data.cpu().numpy().tobytes())
    return digest.hexdigest()


def tensor_statistics(tensor):
    """Check every element without creating a full-size FP32 temporary."""
    tensor = tensor.detach()
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    row_size = tensor[0].numel() if len(tensor) else 1
    rows = max(1, CHUNK // max(1, row_size))
    count, nonfinite, total, squares = 0, 0, 0.0, 0.0
    low, high = float("inf"), float("-inf")
    for offset in range(0, len(tensor), rows):
        values = tensor[offset : offset + rows].float()
        finite = torch.isfinite(values)
        nonfinite += int((~finite).sum().item())
        values = values[finite]
        if values.numel():
            count += values.numel()
            low = min(low, values.min().item())
            high = max(high, values.max().item())
            total += values.sum(dtype=torch.float64).item()
            squares += values.double().square().sum().item()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "nonfinite": nonfinite,
        "min": low if count else None,
        "max": high if count else None,
        "mean": total / count if count else None,
        "rms": (squares / count) ** 0.5 if count else None,
    }


def check_received_rows(expected, actual, slots, positions, ratio):
    """Compare payloads only where the destination is supposed to insert rows."""
    count = min(slots.numel(), expected.shape[0])
    valid = (
        (slots[:count] >= 0)
        & (positions[:count] >= 0)
        & ((positions[:count] + 1) % ratio == 0)
        & (expected[:count, -1] != 0)
    )
    if not torch.equal(expected[:count][valid], actual[:count][valid]):
        raise RuntimeError("DeepSeek PP inserted cache bytes or scales differ")
    return int(valid.sum().item())


def _tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensors(item, f"{prefix}.{index}")


class DiagnosticProbe:
    def __init__(self, stage, rank, max_steps=2):
        self.stage = stage
        self.rank = rank
        self.max_steps = max_steps
        self.step = 0
        self.active = False
        self.references = {}
        self.reference_bytes = 0

    def emit(self, event, **values):
        logger.info(
            "DSV41_DIAG %s",
            json.dumps(
                dict(event=event, stage=self.stage, tp=self.rank, step=self.step)
                | values,
                allow_nan=False,
            ),
        )

    def begin(self, model, args, kwargs):
        self.active = False
        if self.step >= self.max_steps or not is_forward_context_available():
            return
        context = get_forward_context()
        if not isinstance(context.attn_metadata, dict) or not context.attn_metadata:
            return
        padding = context.is_padding
        if padding is not None and bool(padding.all().item()):
            return
        self.step += 1
        self.active = True
        self.record(
            "stage.input_ids", kwargs.get("input_ids", args[0] if args else None)
        )
        intermediate = kwargs.get(
            "intermediate_tensors", args[2] if len(args) > 2 else None
        )
        if intermediate is not None:
            self.packet("receive", self.stage, intermediate.tensors)
            for key in ("hidden_states", "pre_mix"):
                self.record(f"stage.received.{key}", intermediate[key])

    def record(self, name, value):
        if not self.active:
            return
        for suffix, tensor in _tensors(value):
            stats = tensor_statistics(tensor)
            if self.rank == 0 or stats["nonfinite"]:
                self.emit("activation", name=name + suffix, **stats)
            if stats["nonfinite"]:
                raise RuntimeError(
                    f"DeepSeek first non-finite activation: {name}{suffix}"
                )

    def packet(self, direction, boundary, tensors):
        if self.active:
            self.emit(
                "packet",
                direction=direction,
                boundary=boundary,
                tensors={
                    name: {"shape": list(value.shape), "sha256": tensor_digest(value)}
                    for name, value in tensors.items()
                    if name in ("hidden_states", "pre_mix") or name.startswith("pp_")
                },
            )

    def capture_weights(self, model, loaded):
        parameters = dict(model.named_parameters())
        missing = sorted(
            name
            for name in parameters
            if name not in loaded and not name.endswith("hc_attn_fn_broadcast")
        )
        self.emit(
            "loading",
            missing=missing[:128],
            missing_count=len(missing),
            tracked=len(loaded),
        )
        # Save a few native rows before post-load kernel packing changes their layout.
        for name, module in model.named_modules():
            if not name.endswith(("fused_wqa_wkv", "wq_b", "wo_b", "weights_proj")):
                continue
            weight = getattr(module, "weight", None)
            scale = getattr(module, "weight_scale", None)
            if weight is None or weight.ndim != 2:
                continue
            rows = min(8, weight.shape[0])
            size = rows * weight.shape[1] * weight.element_size()
            if weight.dtype == torch.float8_e4m3fn:
                size += rows * (weight.shape[1] // 32)
            if self.reference_bytes + size > 8 * 1024**2:
                continue
            if weight.dtype == torch.float8_e4m3fn:
                if scale is None or scale.shape != (
                    weight.shape[0],
                    weight.shape[1] // 32,
                ):
                    continue
                scales = scale[:rows].detach().clone().view(torch.uint8)
            elif weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
                scales = None
            else:
                continue
            self.references[id(module)] = (weight[:rows].detach().clone(), scales)
            self.reference_bytes += size
        self.emit("reference_storage", bytes=self.reference_bytes)

    def compare_linear(self, module, args, output, name):
        if not self.active or id(module) not in self.references or not args:
            return
        x = args[0]
        y = output[0] if isinstance(output, tuple) else output
        weight, scales = self.references[id(module)]
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            return
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != weight.shape[1]:
            return
        reference_weight = weight.float()
        if scales is not None:
            reference_weight *= torch.exp2(scales.float() - 127).repeat_interleave(
                32, 1
            )
        reference = x[:2].float() @ reference_weight.T
        if getattr(module, "reduce_results", False):
            from vllm.distributed import tensor_model_parallel_all_reduce

            reference = tensor_model_parallel_all_reduce(reference)
        actual = y[:2, : weight.shape[0]].float()
        if self.rank == 0:
            if not bool(torch.isfinite(reference).all().item()):
                self.emit("linear_reference_invalid", name=name)
                return
            error = actual.double() - reference.double()
            self.emit(
                "linear_reference",
                name=name,
                max_abs_error=error.abs().max().item(),
                relative_l2=(
                    error.norm() / reference.double().norm().clamp_min(1e-12)
                ).item(),
            )

    def install(self, model):
        model.register_forward_pre_hook(self.begin, with_kwargs=True)
        for name, module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            # Compressor buffers deliberately leave non-boundary rows unwritten.
            observed = leaf in (
                "attn",
                "ffn",
                "engram",
                "attn_norm",
                "ffn_norm",
                "q_norm",
                "kv_norm",
                "fused_wqa_wkv",
                "wq_b",
                "wo_b",
                "weights_proj",
            ) or (name.startswith("layers.") and name.count(".") == 1)
            if name != "norm" and not (name.startswith("layers.") and observed):
                continue

            def before(module, args, name=name):
                self.record(name + ".input", args)

            def observe(module, args, output, name=name):
                self.record(name, output)
                self.compare_linear(module, args, output, name)

            module.register_forward_pre_hook(before)
            module.register_forward_hook(observe)
