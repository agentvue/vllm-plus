# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from vllm.config import ParallelConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import _get_dspark_parallel_config
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


@dataclass
class _FakeEPLBConfig:
    num_redundant_experts: int = 0


@dataclass
class _FakeParallelConfig:
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 8
    enable_eplb: bool = True
    eplb_config: _FakeEPLBConfig = field(
        default_factory=lambda: _FakeEPLBConfig(num_redundant_experts=32)
    )
    enable_elastic_ep: bool = True

    def __post_init__(self) -> None:
        if not self.enable_eplb and self.eplb_config.num_redundant_experts:
            raise ValueError("redundant experts require EPLB")
        if self.enable_elastic_ep and not self.enable_eplb:
            raise ValueError("elastic EP requires EPLB")


def test_dspark_parallel_config_disables_eplb_atomically():
    target_config = _FakeParallelConfig()

    draft_config = _get_dspark_parallel_config(
        target_config,
        tensor_parallel_size=4,
    )

    assert target_config.pipeline_parallel_size == 2
    assert target_config.tensor_parallel_size == 8
    assert target_config.enable_eplb
    assert target_config.eplb_config.num_redundant_experts == 32
    assert target_config.enable_elastic_ep

    assert draft_config is not target_config
    assert draft_config.pipeline_parallel_size == 1
    assert draft_config.tensor_parallel_size == 4
    assert not draft_config.enable_eplb
    assert draft_config.eplb_config.num_redundant_experts == 0
    assert not draft_config.enable_elastic_ep
    assert draft_config.eplb_config is not target_config.eplb_config


@pytest.mark.parametrize("pcp_size", [1, 4])
@pytest.mark.parametrize("dcp_size", [1, 4])
@pytest.mark.parametrize("use_mla", [False, True])
def test_draft_context_parallelism_without_changing_target(
    monkeypatch, pcp_size, dcp_size, use_mla
):
    target_parallel = ParallelConfig(
        tensor_parallel_size=4,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=16,
        distributed_executor_backend="mp",
    )
    target_config = SimpleNamespace(
        parallel_config=target_parallel,
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(use_mla=use_mla)
        ),
    )

    class CapturedConfig(Exception):
        pass

    def capture_init(self, config, device):
        raise CapturedConfig(config)

    monkeypatch.setattr(DraftModelSpeculator, "__init__", capture_init)
    with pytest.raises(CapturedConfig) as captured:
        DFlashSpeculator(target_config, device=None)
    draft_parallel = captured.value.args[0].parallel_config
    assert draft_parallel.tensor_parallel_size == 4
    assert draft_parallel.prefill_context_parallel_size == 1
    assert draft_parallel.decode_context_parallel_size == (dcp_size if use_mla else 1)
    assert draft_parallel.cp_kv_cache_interleave_size == 16
    assert target_parallel.prefill_context_parallel_size == pcp_size
    assert target_parallel.decode_context_parallel_size == dcp_size
    assert target_parallel.cp_kv_cache_interleave_size == 16


@pytest.mark.parametrize("configured_dcp", [1, 4])
def test_attention_uses_draft_dcp_setting_inside_target_process_group(
    monkeypatch, configured_dcp
):
    import vllm.config as config_module
    from vllm.distributed import parallel_state
    from vllm.v1.attention.backend import AttentionImplBase

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=configured_dcp)
    )
    monkeypatch.setattr(
        config_module, "get_current_vllm_config_or_none", lambda: config
    )

    monkeypatch.setattr(
        parallel_state, "_DCP", SimpleNamespace(world_size=4, rank_in_group=2)
    )
    impl = AttentionImplBase()
    assert impl.dcp_world_size == configured_dcp
    assert impl.dcp_rank == (0 if configured_dcp == 1 else 2)


@pytest.mark.parametrize("draft_dcp_size", [1, 4])
def test_runner_marks_only_replicated_draft_caches(monkeypatch, draft_dcp_size):
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    monkeypatch.setattr(
        model_runner, "get_kv_cache_spec", lambda _: {"target": spec, "draft": spec}
    )
    runner = object.__new__(model_runner.GPUModelRunner)
    runner.vllm_config = None
    runner.dcp_size = 4
    runner.speculator = object.__new__(DFlashSpeculator)
    runner.speculator.dcp_size = draft_dcp_size
    runner.speculator.draft_attn_layer_names = {"draft"}

    specs = runner.get_kv_cache_spec()

    assert specs["target"].dcp_sharded
    assert specs["draft"].dcp_sharded == (draft_dcp_size == 4)


def test_pp_dspark_loads_only_the_missing_checkpoint_embedding(monkeypatch):
    from torch import nn

    from vllm.model_executor.models.utils import PPMissingLayer
    from vllm.v1.worker.gpu.spec_decode.dflash import utils

    monkeypatch.setattr(utils, "get_pp_group", lambda: SimpleNamespace(world_size=5))
    seen = []

    class Loader:
        Source = staticmethod(lambda *args: args)

        def __init__(self, config):
            pass

        def _get_weights_iterator(self, source):
            yield "vision.weight", torch.full((1,), 7.0)
            yield "embed.weight", torch.arange(16.0).view(4, 4)
            raise AssertionError(
                "The embedding loader must stop after finding its weight"
            )

    monkeypatch.setattr(utils, "DefaultModelLoader", Loader)
    weight = nn.Parameter(torch.zeros(2, 4), requires_grad=False)

    def load_embedding(param, value):
        seen.append(value.shape)
        param.copy_(value[:2])

    weight.weight_loader = load_embedding
    draft = SimpleNamespace(has_own_embed_tokens=False)
    inner = SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight))
    target = SimpleNamespace(embed_tokens=PPMissingLayer())
    config = SimpleNamespace(
        load_config=None,
        model_config=SimpleNamespace(
            model_weights=None, model="checkpoint", revision="pinned"
        ),
    )
    utils._load_pp_target_embedding(draft, inner, target, config)
    assert draft.has_own_embed_tokens
    assert seen == [torch.Size([4, 4])]
    torch.testing.assert_close(weight, torch.arange(8.0).view(2, 4))


@pytest.mark.parametrize("projection", ["w1", "w2", "w3"])
@pytest.mark.parametrize(
    "algo,block,dtype",
    [
        ("MXFP4", 32, torch.uint8),
        ("MXFP4", 32, torch.float8_e8m0fnu),
        ("NVFP4", 16, torch.float8_e4m3fn),
    ],
)
def test_dspark_expert_scales_match_the_selected_format(projection, algo, block, dtype):
    from vllm.models.deepseek_v41.nvidia.dspark import validate_dspark_expert_tensor

    config = SimpleNamespace(hidden_size=64, moe_intermediate_size=32)
    n, k = (64, 32) if projection == "w2" else (32, 64)
    name = f"model.layers.0.ffn.experts.0.{projection}.scale"
    weight = torch.empty(n, k // block, dtype=dtype)
    validate_dspark_expert_tensor(name, weight, config, algo)
    other = "MXFP4" if algo == "NVFP4" else "NVFP4"
    with pytest.raises(ValueError, match="Invalid .* DSpark tensor"):
        validate_dspark_expert_tensor(name, weight, config, other)


def _dspark_loader(monkeypatch, algo="MXFP4"):
    from torch import nn

    from vllm.models.deepseek_v41.nvidia import dspark

    model = object.__new__(dspark.DSparkDeepseekV4ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=32,
        moe_intermediate_size=32,
        expert_dtype="fp4",
        num_hidden_layers=40,
        n_routed_experts=2,
        dspark_n_routed_experts=2,
        num_attention_heads=1,
        quantization_config={
            "moe_quant_algo": "NVFP4",
            "ignore": ["mtp.*"],
            "quantized_layers": {"mtp.0.ffn.experts": {"quant_algo": algo}},
        },
    )
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

    model.quant_config = object.__new__(DeepseekV4FP8Config)
    model.pad_shared_expert = False
    model.linear_scale_name = "weight_scale"
    model.model = nn.Module()
    model.model.confidence_head = None
    layer = nn.Module()
    layer.ffn = nn.Module()
    layer.ffn.use_mega_moe = False
    layer.ffn.experts = nn.Module()
    experts = layer.ffn.experts.routed_experts = nn.Module()
    scale_dtype = torch.float8_e4m3fn if algo == "NVFP4" else torch.uint8
    scale_width = 2 if algo == "NVFP4" else 1
    shapes: dict[str, tuple[int, ...]] = {
        "w13_weight": (2, 64, 16),
        "w2_weight": (2, 32, 16),
        "w13_weight_scale": (2, 64, scale_width),
        "w2_weight_scale": (2, 32, scale_width),
    }
    if algo == "NVFP4":
        shapes.update(
            w13_weight_scale_2=(2, 2),
            w2_weight_scale_2=(2,),
            w13_input_scale=(2, 2),
            w2_input_scale=(2,),
        )
    for name, shape in shapes.items():
        dtype = scale_dtype if name.endswith("weight_scale") else torch.uint8
        if name.endswith(("input_scale", "weight_scale_2")):
            dtype = torch.float32
        param = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)

        def load(param, weight, name, shard_id, expert_id, **kw):
            out = param[expert_id]
            if shard_id in ("w1", "w3"):
                if out.ndim == 1:
                    out = out[:1] if shard_id == "w1" else out[1:]
                else:
                    out = out[:32] if shard_id == "w1" else out[32:]
            out.copy_(weight.reshape_as(out))
            return True

        param.weight_loader = load
        experts.register_parameter(name, param)
    model.model.layers = nn.ModuleList([layer])
    model.process_weights_after_loading = lambda: None
    mapping = [
        (
            f"ffn.experts.routed_experts.{target}_",
            f"ffn.experts.{expert}.{projection}.",
            expert,
            projection,
        )
        for expert in range(2)
        for projection, target in (("w1", "w13"), ("w3", "w13"), ("w2", "w2"))
    ]
    monkeypatch.setattr(
        dspark, "fused_moe_make_expert_params_mapping", lambda *a, **k: mapping
    )
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 0)
    suffixes = [("weight", 16), ("scale", scale_width)]
    weights = [
        (
            f"mtp.0.ffn.experts.{expert}.{projection}.{suffix}",
            torch.ones(32, width).to(scale_dtype if suffix == "scale" else torch.uint8),
        )
        for expert in range(2)
        for projection in ("w1", "w2", "w3")
        for suffix, width in suffixes
    ]
    if algo == "NVFP4":
        weights.extend(
            (f"mtp.0.ffn.experts.{expert}.{projection}.{suffix}", torch.ones(1))
            for expert in range(2)
            for projection in ("w1", "w2", "w3")
            for suffix in ("weight_scale_2", "input_scale")
        )
    return model, weights


@pytest.mark.parametrize("algo", ["MXFP4", "NVFP4"])
def test_dspark_loader_requires_every_expert_projection(monkeypatch, algo):
    model, weights = _dspark_loader(monkeypatch, algo)
    assert len(model.load_weights(iter(weights))) == (8 if algo == "NVFP4" else 4)
    with pytest.raises(ValueError, match="Incomplete DSpark expert tensor"):
        model.load_weights(iter(weights[:-1]))


def test_dspark_loader_rejects_missing_required_scale_parameter(monkeypatch):
    model, weights = _dspark_loader(monkeypatch)
    with pytest.raises(ValueError, match="missing parameters"):
        model.load_weights(
            iter((n, w) for n, w in weights if not n.endswith("w2.scale"))
        )


def test_dspark_loader_requires_nvfp4_global_scales(monkeypatch):
    model, weights = _dspark_loader(monkeypatch, "NVFP4")
    with pytest.raises(ValueError, match="missing parameters"):
        model.load_weights(
            iter((n, w) for n, w in weights if not n.endswith("input_scale"))
        )
