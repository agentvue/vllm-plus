# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 stage dependencies and mHC/DSpark outputs across shared-cache splits."""

from bisect import bisect_right
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v41.common.pp import shared_cache_dependency
from vllm.models.deepseek_v41.nvidia import model as dsv41
from vllm.models.deepseek_v41.nvidia import vl_model


def config():
    return SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
    )


@pytest.mark.parametrize(
    "boundary,source,index,ratio,candidates",
    [
        (10, 8, 8, 2, False),
        (21, 20, 20, 1, True),
        (30, 20, 28, 1, True),
        (36, 20, 36, 1, True),
        (39, 20, 36, 1, True),
    ],
)
def test_boundary_carries_the_original_cache_and_latest_selection(
    boundary, source, index, ratio, candidates
):
    dependency = shared_cache_dependency(config(), boundary)
    assert dependency is not None
    assert (
        dependency.source,
        dependency.index_source,
        dependency.compress_ratio,
        dependency.candidates,
    ) == (source, index, ratio, candidates)


@pytest.mark.parametrize("boundary", [0, 1, 2, 8, 14, 20, 40])
def test_new_source_does_not_transfer_obsolete_cache(boundary):
    assert shared_cache_dependency(config(), boundary) is None


class _Layer:
    def __init__(self, index):
        self.index = index

    def __call__(
        self,
        hidden,
        positions,
        ids,
        pre,
        post,
        mix,
        residual,
        hashes,
        mask,
        capture_previous_aux,
        **kwargs,
    ):
        previous = None
        if residual is not None:
            hidden = residual + hidden[:, None, :]
            if capture_previous_aux:
                previous = hidden.mean(1)
        elif hidden.ndim == 2:
            hidden = hidden[:, None, :].expand(-1, 2, -1).contiguous()
        x = torch.full_like(hidden[:, 0], self.index + 1)
        return x, hidden, torch.ones(1), torch.ones(1), torch.ones(1), previous


def _model(start, end):
    model = object.__new__(dsv41.DeepseekV4Model)
    nn.Module.__init__(model)
    model.start_layer, model.end_layer = start, end
    model.config = SimpleNamespace(num_hidden_layers=4)
    model.aux_hidden_state_layers = (1, 2, 3)
    model._aux_slot_base_cached = bisect_right((1, 2, 3), start)
    model._aux_upstream_total_cached = model._aux_slot_base_cached if end == 4 else 0
    model.use_sequence_parallel = False
    model.use_mega_moe = False
    model.fuse_mhc_all_reduce = False
    model.engram_hash = None
    model.engram_dp_shared_memory = False
    model.topk_indices_buffer = torch.empty((3, 4), dtype=torch.int32)
    model.candidate_block_buffer = None
    model._mtp_hidden_buffer = None
    model.layers = [_Layer(i) for i in range(4)]
    model.norm = lambda x: x * 2
    model.pp_shared_state = SimpleNamespace(receive=lambda *a: None, send=lambda *a: {})
    return model


@pytest.mark.parametrize("boundaries", [(0, 1, 2, 3, 4), (0, 2, 4), (0, 3, 4)])
def test_split_pipeline_preserves_logits_inputs_and_dspark_attention_inputs(
    monkeypatch, boundaries
):
    group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(dsv41, "get_pp_group", lambda: group)
    monkeypatch.setattr(
        dsv41, "mhc_post_tilelang", lambda x, residual, *a: residual + x[:, None, :]
    )
    monkeypatch.setattr(dsv41, "hc_collapse_triton", lambda x, *a: x.mean(1))
    inputs = torch.arange(12, dtype=torch.float32).view(3, 4)
    positions = torch.arange(3)
    expected, expected_aux = _model(0, 4).forward(None, positions, None, inputs)
    intermediate = None
    for start, end in zip(boundaries, boundaries[1:]):
        group.is_first_rank, group.is_last_rank = start == 0, end == 4
        output = _model(start, end).forward(
            None, positions, intermediate, inputs if start == 0 else None
        )
        if end < 4:
            # The real PPHandler relays upstream auxiliary tensors after forward.
            if intermediate is not None:
                output.tensors.update(
                    {
                        key: value
                        for key, value in intermediate.tensors.items()
                        if key.startswith("aux_hidden_states_")
                    }
                )
            intermediate = output
        else:
            actual, actual_aux = output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(actual_aux) == len(expected_aux) == 3
    for actual, expected in zip(actual_aux, expected_aux):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_multimodal_loading_streams_interleaved_weights_and_finalizes_once(monkeypatch):
    import weakref

    model = object.__new__(vl_model.DeepseekV41ForCausalLM)
    nn.Module.__init__(model)
    model.hf_to_vllm_mapper = SimpleNamespace(apply=lambda weights: weights)
    live, peak, finalized = 0, 0, 0
    seen = set()

    def released():
        nonlocal live
        live -= 1

    def weights():
        nonlocal live, peak
        for i in range(64):
            tensor = torch.ones(1)
            live += 1
            peak = max(peak, live)
            weakref.finalize(tensor, released)
            prefix = "vision" if i % 3 == 0 else "language_model"
            yield f"{prefix}.weight_{i}", tensor

    class Loader:
        def __init__(self, *args):
            pass

        def load_weights(self, weights):
            return {name for name, _ in weights}

    def load_decoder(weights):
        nonlocal finalized
        for name, _ in weights:
            seen.add(name)
        finalized += 1
        return seen

    model.language_model = SimpleNamespace(load_weights=load_decoder)
    monkeypatch.setattr(vl_model, "AutoWeightsLoader", Loader)
    loaded = model.load_weights(weights())
    assert len(loaded) == 64
    assert finalized == 1
    assert peak <= 3


def test_diagnostic_cache_check_detects_changed_scale_and_ignores_invalid_slots():
    """Corrupting a scale byte must fail even when KV data bytes match."""
    from vllm.models.deepseek_v41.nvidia.diagnostics import check_received_rows

    expected = torch.arange(4 * 133, dtype=torch.int32).to(torch.uint8).view(4, 133)
    expected[:, -1] = 1
    slots = torch.tensor([7, -1, 9, 10])
    positions = torch.tensor([1, 3, 4, 5])
    actual = expected.clone()
    actual[1:3].zero_()
    assert check_received_rows(expected, actual, slots, positions, 2) == 2
    actual[3, 130] ^= 1
    with pytest.raises(RuntimeError, match="cache bytes or scales differ"):
        check_received_rows(expected, actual, slots, positions, 2)


def test_diagnostic_statistics_check_all_values_and_preserve_float8_bytes():
    from vllm.models.deepseek_v41.nvidia.diagnostics import (
        tensor_digest,
        tensor_statistics,
    )

    values = torch.tensor([[1.0, -3.0], [5.0, float("nan")]])
    stats = tensor_statistics(values)
    assert stats["nonfinite"] == 1
    assert stats["mean"] == 1.0
    assert stats["rms"] == pytest.approx((35 / 3) ** 0.5)
    fp8 = values.nan_to_num().to(torch.float8_e4m3fn)
    assert tensor_digest(fp8) == tensor_digest(fp8.view(torch.uint8))
    assert tensor_digest(fp8.T) != tensor_digest(fp8)


def test_diagnostic_probe_fails_at_first_nonfinite_operation():
    from vllm.models.deepseek_v41.nvidia.diagnostics import DiagnosticProbe

    probe = DiagnosticProbe(stage=10, rank=0)
    probe.active = True
    with pytest.raises(RuntimeError, match="first non-finite activation: attention"):
        probe.record("attention", torch.tensor([1.0, float("inf")]))


def test_diagnostic_projection_reference_uses_original_mxfp8_scale_bytes():
    """Compare real matrix multiplication with checkpoint-row reference math."""
    from vllm.models.deepseek_v41.nvidia.diagnostics import DiagnosticProbe

    torch.manual_seed(9)
    module = nn.Module()
    module.wq_b = nn.Linear(64, 8, bias=False)
    packed = torch.randn(8, 64).to(torch.float8_e4m3fn)
    module.wq_b.weight = nn.Parameter(packed, requires_grad=False)
    scales = torch.tensor([[125, 129]] * 8, dtype=torch.uint8)
    module.wq_b.weight_scale = nn.Parameter(scales, requires_grad=False)
    probe = DiagnosticProbe(stage=0, rank=0)
    records = []
    probe.emit = lambda event, **values: records.append(dict(event=event, **values))
    probe.capture_weights(module, set(dict(module.named_parameters())))
    x = torch.randn(2, 64)
    reference = (
        x
        @ (packed.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, 1)).T
    )
    # Simulate post-load packing overwriting the live scales.
    module.wq_b.weight_scale.data.zero_()
    probe.active = True
    probe.compare_linear(module.wq_b, (x,), reference, "wq_b")
    assert records[-1]["relative_l2"] == 0
    probe.compare_linear(module.wq_b, (x,), reference + 10, "wq_b")
    assert records[-1]["max_abs_error"] == pytest.approx(10)


def test_diagnostic_hooks_skip_profile_and_stop_after_two_real_forwards(monkeypatch):
    from vllm.models.deepseek_v41.nvidia import diagnostics

    context = SimpleNamespace(attn_metadata=None, is_padding=None)
    monkeypatch.setattr(diagnostics, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(diagnostics, "get_forward_context", lambda: context)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4)])

        def forward(self, input_ids, positions, intermediate_tensors=None):
            return self.layers[0](input_ids)

    model = Model()
    probe = diagnostics.DiagnosticProbe(stage=0, rank=0)
    records = []
    probe.emit = lambda event, **values: records.append((event, values))
    probe.install(model)
    values, positions = torch.ones(2, 4), torch.arange(2)
    model(values, positions)
    assert probe.step == 0
    assert not records
    context.attn_metadata = {"attention": object()}
    context.is_padding = torch.ones(2, dtype=torch.bool)
    model(values, positions)
    assert probe.step == 0
    context.is_padding.zero_()
    model(values, positions)
    model(values, positions)
    assert probe.step == 2
    count = len(records)
    model(values, positions)
    assert len(records) == count
    assert not probe.active
