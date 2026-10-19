# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run isolated CPU contract tests when the vLLM native runtime is unavailable.

Compiles selected definitions from the working tree unchanged. External runtime
services are replaced by mocks. The optional C++ owner test mocks CUDA and,
on Windows, adapts POSIX mappings. This does not validate package imports,
native extension ABI, actual CUDA/Triton kernels, checkpoints, or serving.
Run in a fresh process: this deliberately installs temporary module stubs.
"""

from __future__ import annotations
import __future__

import ast
import copy
import dataclasses
import enum
import gc
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import weakref
from bisect import bisect_right
from collections import Counter, deque, namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from itertools import islice
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from torch import nn

# These names are globals of the unchanged source definitions compiled below.
RUNTIME_GLOBALS = dict(
    bisect_right=bisect_right,
    islice=islice,
    copy=copy,
    enum=enum,
    gc=gc,
    importlib=importlib,
    shutil=shutil,
    subprocess=subprocess,
    json=json,
    logging=logging,
    weakref=weakref,
    Counter=Counter,
    deque=deque,
    ExitStack=ExitStack,
    contextmanager=contextmanager,
    nullcontext=nullcontext,
    Any=Any,
    cast=cast,
    Mock=Mock,
    patch=patch,
    np=np,
    torch=torch,
    nn=nn,
)
ROOT = Path(__file__).resolve().parents[2]


def module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    obj = ModuleType(name)
    obj.__path__ = []
    sys.modules[name] = obj
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(module(parent), child, obj)
    return obj


def load(name, path, functions=(), classes=None, **extra):
    obj = module(name)
    obj.__dict__.update(
        {key: value for key, value in globals().items() if not key.startswith("__")}
    )
    obj.__dict__.update(extra)
    nodes = []
    for node in ast.parse((ROOT / path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in (classes or {}):
            bases, methods = classes[node.name]
            if bases is not None:
                node.bases = [ast.parse(base, mode="eval").body for base in bases]
            if methods is not None:
                node.body = [
                    entry
                    for entry in node.body
                    if isinstance(entry, ast.FunctionDef) and entry.name in methods
                ]
            nodes.append(node)
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(
        compile(
            tree, str(ROOT / path), "exec", flags=__future__.annotations.compiler_flag
        ),
        obj.__dict__,
    )
    return obj


def build_tests():
    sequence = load(
        "vllm.sequence",
        "vllm/sequence.py",
        classes={"IntermediateTensors": (None, None)},
        dataclass=dataclasses.dataclass,
    )
    common = {
        "IntermediateTensors": sequence.IntermediateTensors,
        "logger": MagicMock(),
    }
    interfaces = load(
        "vllm.model_executor.models.interfaces",
        "vllm/model_executor/models/interfaces.py",
        functions=("requires_raw_input_tokens",),
        classes={"EagleModelMixin": (None, None)},
    )
    ray_utils = load(
        "vllm.v1.executor.ray_utils",
        "vllm/v1/executor/ray_utils.py",
        functions=(
            "parse_ray_ordered_node_ips",
            "parse_ray_node_env_vars_json",
            "_validate_ray_ordered_node_ips",
            "get_bundles_for_node_ips",
        ),
        ray=None,
        placement_group_table=None,
        available_resources_per_node=None,
        current_platform=SimpleNamespace(ray_device_key="GPU"),
        **common,
    )
    ray_executor = load(
        "vllm.v1.executor.ray_executor_v2",
        "vllm/v1/executor/ray_executor_v2.py",
        classes={
            "RayExecutorV2": (
                [],
                (
                    "_build_actor_runtime_env",
                    "_get_driver_local_reader_ranks",
                    "_validate_node_env_vars",
                ),
            )
        },
        **common,
    )
    ray_env = load(
        "vllm.v1.executor.ray_env_utils",
        "vllm/v1/executor/ray_env_utils.py",
        functions=("sanitize_node_local_path_env_vars",),
        NODE_LOCAL_PATH_ENV_VARS=("CUTE_DSL_LIBS", "TVM_LIBRARY_PATH"),
    )
    pp = load(
        "vllm.v1.worker.gpu.pp_utils",
        "vllm/v1/worker/gpu/pp_utils.py",
        functions=("compute_need_sampled_mask",),
        classes={
            "PendingRecv": (None, None),
            "PPHandler": (
                [],
                (
                    "broadcast_drafts",
                    "get_prev_sampled_outputs",
                    "on_req_idx_freed",
                    "relay_aux_hidden_states",
                ),
            ),
        },
        dataclass=dataclasses.dataclass,
        async_tensor_h2d=None,
        **common,
    )
    glm = load(
        "vllm.models.glm5next.common.model",
        "vllm/models/glm5next/common/model.py",
        classes={
            "Glm5NextModel": (
                ["nn.Module", "EagleModelMixin"],
                (
                    "forward",
                    "make_empty_intermediate_tensors",
                    "load_weights",
                    "_set_aux_hidden_state_layers",
                    "embed_input_ids",
                ),
            )
        },
        EagleModelMixin=interfaces.EagleModelMixin,
        Glm5NextDecoderLayer=nn.Module,
        get_pp_group=None,
        hc_contract=None,
        is_pp_missing_parameter=None,
        get_spec_layer_idx_from_weight_name=None,
        _try_load_fp8_indexer_wk=None,
        **common,
    )
    mtp = load(
        "vllm.models.glm5next.common.mtp",
        "vllm/models/glm5next/common/mtp.py",
        classes={"Glm5NextMTP": (["nn.Module"], ("forward",))},
        **common,
    )
    load(
        "vllm.models.qwen4_exp.nvidia.mtp",
        "vllm/models/qwen4_exp/nvidia/mtp.py",
        classes={"Qwen4ExpMultiTokenPredictor": (["nn.Module"], ("forward",))},
        get_pp_group=None,
        **common,
    )
    qwen_config = load(
        "vllm.model_executor.models.config",
        "vllm/model_executor/models/config.py",
        classes={
            "Qwen3_5ForConditionalGenerationConfig": (
                [],
                ("verify_and_update_config",),
            ),
            "Qwen4ExpForConditionalGenerationConfig": (
                [],
                ("verify_and_update_config",),
            ),
        },
        **common,
    )
    hybrid = module("vllm.v1.worker.gpu.model_states.mamba_hybrid")
    hybrid.MambaHybridModelState = type(
        "MambaHybridModelState",
        (),
        {"prepare_inputs": Mock(), "prepare_dummy_inputs": Mock()},
    )
    qwen_states = {}
    for backend in ("amd", "nvidia"):
        qwen_states[backend] = load(
            f"vllm.models.qwen4_exp.{backend}.model_state",
            f"vllm/models/qwen4_exp/{backend}/model_state.py",
            classes={"Qwen4ExpModelState": (["MambaHybridModelState"], None)},
            MambaHybridModelState=hybrid.MambaHybridModelState,
            **common,
        )
    compilation = load(
        "vllm.config",
        "vllm/config/compilation.py",
        classes={"CUDAGraphMode": (None, None)},
    )
    graph = load(
        "vllm.v1.worker.gpu.cudagraph_utils",
        "vllm/v1/worker/gpu/cudagraph_utils.py",
        classes={
            "CudaGraphManager": ([], ("capture",)),
            "ModelCudaGraphManager": (["CudaGraphManager"], ("capture",)),
            "BatchExecutionDescriptor": (None, None),
        },
        CUDAGraphMode=compilation.CUDAGraphMode,
        dataclass=dataclasses.dataclass,
        graph_capture=None,
        is_global_first_rank=None,
        prepare_inputs_to_capture=None,
        set_forward_context=None,
        **common,
    )
    runner = load(
        "vllm.v1.worker.gpu.model_runner",
        "vllm/v1/worker/gpu/model_runner.py",
        classes={"GPUModelRunner": ([], ("execute_model",))},
        CUDAGraphMode=compilation.CUDAGraphMode,
        requires_raw_input_tokens=interfaces.requires_raw_input_tokens,
        dispatch_cg_and_sync_dp=None,
        InputBatch=None,
        build_slot_mappings_by_layer=None,
        set_forward_context=None,
        **common,
    )
    descriptor = load(
        "vllm.forward_context",
        "vllm/forward_context.py",
        classes={"BatchDescriptor": (None, None)},
        dataclass=dataclasses.dataclass,
    )
    runner.BatchDescriptor = descriptor.BatchDescriptor
    bcg = load(
        "vllm.compilation.breakable_cudagraph",
        "vllm/compilation/breakable_cudagraph.py",
        classes={
            "_BreakableEntry": (None, None),
            "BreakableCUDAGraphWrapper": (None, None),
        },
        BatchDescriptor=descriptor.BatchDescriptor,
        CUDAGraphMode=compilation.CUDAGraphMode,
        current_platform=MagicMock(),
        set_graph_pool_id=None,
        get_offloader=None,
        is_forward_context_available=None,
        get_forward_context=None,
        BreakableCUDAGraphCapture=None,
        validate_cudagraph_capturing_enabled=lambda: None,
        weak_ref_tensors=lambda value: value,
        **common,
    )
    bcg.logger.isEnabledFor.return_value = False
    envs = module("vllm.envs")
    envs.VLLM_USE_V2_MODEL_RUNNER = None

    # The property reads the real environment after pytest's monkeypatch changes it.
    class Env:
        @property
        def VLLM_PP_LAYER_PARTITION(self):
            return os.environ.get("VLLM_PP_LAYER_PARTITION")

    load(
        "vllm.distributed.utils",
        "vllm/distributed/utils.py",
        functions=("get_pp_indices",),
        envs=Env(),
        **common,
    )
    load(
        "vllm.model_executor.models.utils",
        "vllm/model_executor/models/utils.py",
        functions=("extract_layer_index",),
    )
    layout = load(
        "vllm.v1.kv_cache_layout",
        "vllm/v1/kv_cache_layout.py",
        classes={"KVCacheLayout": (None, None)},
        Enum=enum.Enum,
        _DIM_L=0,
        _DIM_B=1,
        _DIM_H=2,
        _DIM_N=3,
        _DIM_C=4,
    ).KVCacheLayout
    spec_tree = ast.parse((ROOT / "vllm/v1/kv_cache_interface.py").read_text())
    specs_module = load(
        "vllm.v1.kv_cache_interface",
        "vllm/v1/kv_cache_interface.py",
        classes={
            n.name: (None, None) for n in spec_tree.body if isinstance(n, ast.ClassDef)
        },
        functions=(
            "_apply_alignment_padding",
            "compute_layer_kv_cache_shape_bytes",
            "compute_layout_strides",
            "create_kv_cache_views",
        ),
        KVCacheLayout=layout,
        _DIM_L=0,
        _DIM_B=1,
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        fields=dataclasses.fields,
        replace=dataclasses.replace,
        Enum=enum.Enum,
        IntEnum=enum.IntEnum,
        Fraction=__import__("fractions").Fraction,
        cached_property=__import__("functools").cached_property,
        prod=__import__("math").prod,
        get_dtype_size=lambda dtype: torch.empty((), dtype=dtype).element_size(),
        cdiv=lambda a, b: (a + b - 1) // b,
        round_up=lambda value, alignment: (value + alignment - 1)
        // alignment
        * alignment,
        KVCacheSpecRegistry=SimpleNamespace(get_uniform_type_base_spec=type),
        MambaAttentionBackendEnum=SimpleNamespace(MAMBA2="mamba2"),
    )
    spec_globals = {
        n.name: getattr(specs_module, n.name)
        for n in spec_tree.body
        if isinstance(n, ast.ClassDef)
    }
    kv = load(
        "vllm.v1.core.kv_cache_utils",
        "vllm/v1/core/kv_cache_utils.py",
        functions=(
            "_pp_balanced_mamba_group_count",
            "_get_kv_cache_groups_glm5_next",
            "_glm5_next_tensor_layout",
            "create_kv_cache_group_specs",
            "_pool_bytes_per_block",
            "_get_kv_cache_bytes_per_block",
            "get_kv_cache_groups",
            "is_kv_cache_spec_uniform",
            "is_kv_cache_type_attention_free",
            "_get_per_layer_spec",
            "validate_kv_cache_layout",
            "get_kv_cache_config_from_groups",
            "_project_kv_cache_groups_to_worker",
            "may_override_num_blocks",
        ),
        compute_layout_strides=specs_module.compute_layout_strides,
        math=__import__("math"),
        defaultdict=__import__("collections").defaultdict,
        round_up=specs_module.round_up,
        cdiv=lambda a, b: (a + b - 1) // b,
        replace=dataclasses.replace,
        get_hisparse_kv_cache_groups=lambda *args: None,
        **spec_globals,
    )
    multiple_of = load(
        "vllm.v1.attention.backend",
        "vllm/v1/attention/backend.py",
        classes={"MultipleOf": (None, None)},
    ).MultipleOf
    worker_utils = load(
        "vllm.v1.worker.utils",
        "vllm/v1/worker/utils.py",
        functions=(
            "allocate_kv_cache",
            "select_common_block_size",
            "prepare_kernel_block_sizes",
        ),
        create_kv_cache_views=specs_module.create_kv_cache_views,
        current_platform=SimpleNamespace(is_rocm=lambda: False),
        MultipleOf=multiple_of,
        **spec_globals,
    )
    config_class = load(
        "vllm.config.vllm",
        "vllm/config/vllm.py",
        classes={
            "VllmConfig": ([], ("use_v2_model_runner", "_is_dflash_candidate_draft"))
        },
        envs=envs,
        config=lambda **kw: lambda cls: cls,
        ConfigDict=dict,
    ).VllmConfig
    module("vllm.config").VllmConfig = config_class

    fi = module("vllm.utils.flashinfer")
    sm120 = load(
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120",
        "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
        classes={
            "FlashInferMLASparseSM120Impl": (
                [],
                ("_fit_topk_indices", "_run_mqa_kernel"),
            )
        },
        F=torch.nn.functional,
    )
    sm120.FlashInferMLASparseSM120Impl._packed_rope_head_dim = 64
    module(
        "vllm.model_executor.determinism.batch_invariant"
    ).init_batch_invariance = Mock()
    gpu_worker = load(
        "vllm.v1.worker.gpu_worker",
        "vllm/v1/worker/gpu_worker.py",
        functions=("init_worker_distributed_environment",),
        envs=envs,
        override_envs_for_eplb=None,
        set_custom_all_reduce=None,
        ensure_ec_transfer_initialized=None,
        init_distributed_environment=None,
        ensure_model_parallel_initialized=None,
        **common,
    )
    envs.VLLM_RAY_CPU_WORLD_GROUP = False
    comm = load(
        "vllm.distributed.parallel_state",
        "vllm/distributed/parallel_state.py",
        functions=("_split_tensor_dict",),
        classes={
            "_CudaEventHandle": (None, None),
            "GroupCoordinator": (
                [],
                (
                    "isend_tensor_dict",
                    "_reap_completed_isends",
                    "_should_use_all_gather",
                ),
            ),
        },
        TensorMetadata=namedtuple("TensorMetadata", "device dtype size"),
    )
    dflash_utils = load(
        "vllm.v1.worker.gpu.spec_decode.dflash.utils",
        "vllm/v1/worker/gpu/spec_decode/dflash/utils.py",
        functions=("_load_pp_target_embedding",),
        get_pp_group=None,
        DefaultModelLoader=None,
        PPMissingLayer=type("Missing", (), {}),
    )
    module(
        "vllm.model_executor.models.utils"
    ).PPMissingLayer = dflash_utils.PPMissingLayer
    quant_helpers = load(
        "vllm.model_executor.layers.quantization.utils.flashinfer_utils",
        "vllm/model_executor/layers/quantization/utils/flashinfer_utils.py",
        functions=(
            "align_fp4_moe_weights_for_fi",
            "align_trtllm_fp4_moe_hidden_dim_for_fi",
        ),
        round_up=lambda value, alignment: (value + alignment - 1)
        // alignment
        * alignment,
        logger=MagicMock(),
    )
    quant_utils = load(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        "vllm/model_executor/layers/quantization/utils/quant_utils.py",
        functions=("amax_for_moe_activation_quant",),
    )
    nvfp4 = load(
        "vllm.model_executor.layers.fused_moe.oracle.nvfp4",
        "vllm/model_executor/layers/fused_moe/oracle/nvfp4.py",
        functions=("is_global_sf_supported_for_nvfp4_backend",),
        classes={"NvFp4MoeBackend": (None, None)},
        Enum=enum.Enum,
    )
    nvfp4.FLASHINFER_NVFP4_MOE_BACKENDS = [
        backend
        for backend in nvfp4.NvFp4MoeBackend
        if backend.name.startswith("FLASHINFER")
    ]
    quant = load(
        "vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe",
        "vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py",
        functions=("reorder_w1w3_to_w3w1", "prepare_nvfp4_moe_layer_for_fi_or_cutlass"),
        align_fp4_moe_weights_for_fi=quant_helpers.align_fp4_moe_weights_for_fi,
        amax_for_moe_activation_quant=quant_utils.amax_for_moe_activation_quant,
        swizzle_blockscale=None,
    )
    ple = load(
        "vllm.models.qwen4_exp.common.ngram_embedding",
        "vllm/models/qwen4_exp/common/ngram_embedding.py",
        classes={
            "Qwen4ExpPLEPinnedHostEmbedding": ([], ("allocate_embedding_weight",))
        },
        _PLE_HOST_REGISTER_CHUNK_BYTES=512 * 1024**2,
        current_platform=SimpleNamespace(is_cuda=lambda: True),
    )
    workspace = load(
        "vllm.v1.worker.workspace",
        "vllm/v1/worker/workspace.py",
        functions=("_compute_bytes", "use_workspace_lane"),
        classes={"WorkspaceManager": (None, None)},
        prod=__import__("math").prod,
        accumulate=__import__("itertools").accumulate,
        _workspace_lane=__import__("contextvars").ContextVar(
            "qsa_test_lane", default=0
        ),
        dbo_current_ubatch_id=lambda: 0,
        round_up=specs_module.round_up,
        envs=SimpleNamespace(VLLM_DEBUG_WORKSPACE=False),
    )
    qsa_ops = load(
        "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer",
        "vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py",
        functions=("get_qsa_prefill_workspace", "qsa_select_paged_prefill"),
        triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        envs=type(
            "QSAEnv",
            (),
            {
                "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": property(
                    lambda _: int(
                        os.environ.get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 512)
                    )
                )
            },
        )(),
        _TOPK_WORKSPACE_BYTES=1024**2,
        current_workspace_manager=None,
        _prefill_logits=None,
        _topk=None,
    )
    qsa_indexer = load(
        "vllm.models.qwen4_exp.nvidia.indexer_qsa",
        "vllm/models/qwen4_exp/nvidia/indexer_qsa.py",
        classes={"QSAIndexer": ([], ("reserve_prefill_workspace",))},
        get_qsa_prefill_workspace=qsa_ops.get_qsa_prefill_workspace,
    )
    qsa_attention = load(
        "vllm.models.qwen4_exp.nvidia.ops.qsa",
        "vllm/models/qwen4_exp/nvidia/ops/qsa.py",
        functions=(
            "_get_qsa_attention_workspace",
            "_qsa_max_split_rows",
            "reserve_qsa_attention_workspace",
            "_select_config",
            "_select_sm120_config",
            "_select_sm90_config",
        ),
        lru_cache=__import__("functools").lru_cache,
        _is_sm120=lambda: True,
        _is_sm90=lambda: False,
        triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        current_workspace_manager=None,
    )
    activation = load(
        "vllm.model_executor.layers.fused_moe.activation",
        "vllm/model_executor/layers/fused_moe/activation.py",
        classes={"MoEActivation": (None, None)},
        Enum=enum.Enum,
    )
    fi_experts = load(
        "vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe",
        "vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py",
        classes={
            "FlashInferExperts": ([], ("workspace_shapes", "_workspace_size", "apply"))
        },
        functools=__import__("functools"),
        logger=MagicMock(),
        flashinfer_cutlass_fused_moe_workspace_size=None,
        flashinfer_cutlass_fused_moe=None,
        activation_to_flashinfer_type=lambda value: value,
        MoEActivation=activation.MoEActivation,
    )
    gdn = load(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        classes={"QwenGatedDeltaNetAttention": ([], ("_warmup_prefill_kernels",))},
        FLA_CHUNK_SIZE=64,
        fused_post_conv_prep=None,
        logger=MagicMock(),
        PluggableLayer=SimpleNamespace(register=lambda _: lambda cls: cls),
    )
    qsa_owner = load(
        "vllm.models.qwen4_exp.nvidia.qsa",
        "vllm/models/qwen4_exp/nvidia/qsa.py",
        classes={"Qwen4ExpQSAAttention": ([], ("_run_qsa",))},
        get_forward_context=None,
        eager_break_during_capture=lambda fn: fn,
        __package__="vllm.models.qwen4_exp.nvidia",
    )
    specs = [
        (
            "tests/models/qwen4_exp/test_qsa_reference.py",
            (
                "test_qsa_prefill_scratch",
                "test_qsa_profile_reserves",
                "test_qsa_split_scratch",
            ),
            {
                "qsa_indexer_ops": qsa_ops,
                "indexer_qsa": qsa_indexer,
                "qsa_owner": qsa_owner,
                "qsa_ops": qsa_attention,
                "WorkspaceManager": workspace.WorkspaceManager,
                "use_workspace_lane": workspace.use_workspace_lane,
            },
        ),
        (
            "tests/kernels/moe/test_flashinfer_moe.py",
            ("test_flashinfer_workspace",),
            {
                "flashinfer_cutlass_moe": fi_experts,
                "FlashInferExperts": fi_experts.FlashInferExperts,
                "MoEActivation": activation.MoEActivation,
            },
        ),
        (
            "tests/kernels/mamba/test_gdn_prefill_flashinfer.py",
            ("test_gdn_warmup_profiles",),
            {"QwenGatedDeltaNetAttention": gdn.QwenGatedDeltaNetAttention},
        ),
        (
            "tests/utils_/test_torch_utils.py",
            ("test_registered_host_owner",),
            {"__file__": str(ROOT / "tests/utils_/test_torch_utils.py")},
        ),
        (
            "tests/quantization/test_trtllm_nvfp4_hidden_dim_padding.py",
            ("test_cutlass_padding", "test_shared_nvfp4", "test_align_trtllm"),
            {
                "flashinfer_fp4_moe": quant,
                "NvFp4MoeBackend": nvfp4.NvFp4MoeBackend,
                "prepare_nvfp4_moe_layer_for_fi_or_cutlass": (
                    quant.prepare_nvfp4_moe_layer_for_fi_or_cutlass
                ),
                "align_fp4_moe_weights_for_fi": (
                    quant_helpers.align_fp4_moe_weights_for_fi
                ),
                "align_trtllm_fp4_moe_hidden_dim_for_fi": (
                    quant_helpers.align_trtllm_fp4_moe_hidden_dim_for_fi
                ),
            },
        ),
        (
            "tests/models/qwen4_exp/test_ple.py",
            ("test_large_cuda_ple", "test_small_or_non_cuda_ple"),
            {
                "ngram_embedding_module": ple,
                "Qwen4ExpPLEPinnedHostEmbedding": ple.Qwen4ExpPLEPinnedHostEmbedding,
            },
        ),
        ("tests/test_config.py", ("test_dflash_candidate_rejects_v1",), {}),
        (
            "tests/v1/spec_decode/test_dflash_profile.py",
            ("test_pp_dflash_loads_missing_embedding",),
            {},
        ),
        (
            "tests/v1/executor/test_ray_utils.py",
            ("test_ordered", "test_node_env"),
            {"ray_utils": ray_utils},
        ),
        (
            "tests/test_ray_env_utils.py",
            ("test_sanitize_node_paths",),
            {
                "sanitize_node_local_path_env_vars": (
                    ray_env.sanitize_node_local_path_env_vars
                )
            },
        ),
        (
            "tests/distributed/test_ray_v2_executor.py",
            (
                "test_per_node_runtime_env",
                "test_driver_local_reader",
                "test_node_environment",
            ),
            {"RayExecutorV2": ray_executor.RayExecutorV2},
        ),
        ("tests/v1/worker/test_pp_utils.py", ("test_", "_batch"), {"pp_utils": pp}),
        (
            "tests/models/glm5next/test_pipeline_parallel.py",
            ("test_", "_model"),
            {
                "glm_model": glm,
                "PPHandler": pp.PPHandler,
                "glm_mtp": mtp,
                "IntermediateTensors": sequence.IntermediateTensors,
            },
        ),
        (
            "tests/models/qwen4_exp/test_config.py",
            (
                "test_qwen4_exp_mtp_returns_sample",
                "test_local_mtp_on_last",
                "test_qwen4_exp_accepts_pipeline",
                "test_qwen4_exp_pipeline_state",
                "test_qwen4_exp_model_state_prepares",
            ),
            {
                "_text_config": lambda **kwargs: SimpleNamespace(hc_count=2, **kwargs),
                "Qwen3_5ForConditionalGenerationConfig": (
                    qwen_config.Qwen3_5ForConditionalGenerationConfig
                ),
                "Qwen4ExpForConditionalGenerationConfig": (
                    qwen_config.Qwen4ExpForConditionalGenerationConfig
                ),
                "MambaHybridModelState": hybrid.MambaHybridModelState,
                "Qwen4ExpModelState": qwen_states["nvidia"].Qwen4ExpModelState,
                "import_module": importlib.import_module,
            },
        ),
        (
            "tests/v1/worker/test_gpu_model_runner_v2.py",
            ("test_pipeline_forward_preserves",),
            {
                "model_runner_module": runner,
                "GPUModelRunner": runner.GPUModelRunner,
                "CUDAGraphMode": compilation.CUDAGraphMode,
                "IntermediateTensors": sequence.IntermediateTensors,
                "contextlib": SimpleNamespace(nullcontext=nullcontext),
            },
        ),
        (
            "tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py",
            ("test_pipeline_capture_preserves",),
            {
                "cgu": graph,
                "CUDAGraphMode": compilation.CUDAGraphMode,
                "IntermediateTensors": sequence.IntermediateTensors,
                "contextlib": SimpleNamespace(nullcontext=nullcontext),
            },
        ),
        (
            "tests/v1/core/test_kv_cache_utils.py",
            (
                "test_glm5_draft_states_are_counted",
                "test_glm5_empty_projected_groups",
                "test_glm5_window_draft",
                "test_projected_empty_wrapped_group",
                "new_mamba_spec",
                "new_mla_spec",
                "_glm5_like_kv_cache_spec",
            ),
            {
                "kv_cache_utils": kv,
                "KVCacheLayout": layout,
                **spec_globals,
                "ModelConfig": lambda **kwargs: SimpleNamespace(**kwargs),
                "VllmConfig": lambda **kwargs: SimpleNamespace(
                    **kwargs,
                    parallel_config=SimpleNamespace(pipeline_parallel_size=1),
                    scheduler_config=SimpleNamespace(
                        disable_hybrid_kv_cache_manager=False
                    ),
                ),
            },
        ),
        (
            "tests/v1/cudagraph/test_cudagraph_manager.py",
            ("test_capture_completion",),
            {"gpu_cudagraph_utils": graph},
        ),
        (
            "tests/v1/cudagraph/test_breakable_cudagraph.py",
            ("test_wrapper_synchronizes_before_cleanup",),
            {},
        ),
        (
            "tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py",
            ("test_sm120_topk_keeps", "test_sm120_nope_query"),
            {"fi_utils": fi},
        ),
        (
            "tests/v1/worker/test_gpu_worker.py",
            ("test_ray_cpu_world_group",),
            {"gpu_worker": gpu_worker},
        ),
        (
            "tests/distributed/test_comm_ops.py",
            ("test_pynccl_tensor_dict_keeps", "_make_group_for_unit_test"),
            {"GroupCoordinator": comm.GroupCoordinator},
        ),
    ]
    dsv_dependencies = load(
        "vllm.models.deepseek_v41.common.pp",
        "vllm/models/deepseek_v41/common/pp.py",
        functions=("shared_cache_dependency",),
        classes={"SharedCacheDependency": (None, None)},
        dataclass=dataclasses.dataclass,
    )
    mla_indexer = load(
        "vllm.v1.attention.backends.mla.indexer",
        "vllm/v1/attention/backends/mla/indexer.py",
        functions=("get_max_prefill_buffer_size",),
        classes={
            "DeepseekV32IndexerMetadataBuilder": (
                [],
                ("_split_indexer_prefill_chunks",),
            ),
            "DeepseekV41IndexerBackend": (
                [],
                ("get_supported_kernel_block_sizes", "get_name"),
            ),
        },
        current_platform=SimpleNamespace(
            is_device_capability_family=lambda family: family == 120
        ),
    )
    dsv_engram = load(
        "vllm.models.deepseek_v41.nvidia.engram",
        "vllm/models/deepseek_v41/nvidia/engram.py",
        functions=("allocate_engram_cpu_tensor",),
    )
    dsv_model = load(
        "vllm.models.deepseek_v41.nvidia.model",
        "vllm/models/deepseek_v41/nvidia/model.py",
        classes={
            "DeepseekV4Model": (
                ["nn.Module", "EagleModelMixin"],
                ("forward", "_set_aux_hidden_state_layers"),
            )
        },
        EagleModelMixin=interfaces.EagleModelMixin,
        get_pp_group=None,
        mhc_post_tilelang=None,
        hc_collapse_triton=None,
        **common,
    )
    dsv_vl = load(
        "vllm.models.deepseek_v41.nvidia.vl_model",
        "vllm/models/deepseek_v41/nvidia/vl_model.py",
        classes={"DeepseekV41ForCausalLM": (["nn.Module"], ("load_weights",))},
        MULTIMODAL_REGISTRY=SimpleNamespace(
            register_processor=lambda *a, **kw: lambda cls: cls
        ),
        DeepseekV4VLMultiModalProcessor=None,
        DeepseekV4VLProcessingInfo=None,
        DeepseekV4VLDummyInputsBuilder=None,
        AutoWeightsLoader=None,
    )
    specs.extend(
        [
            (
                "tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py",
                (
                    "test_indexer_gather_workspace_covers_scheduled_contexts",
                    "test_sm120_indexer_cache_setup",
                ),
                {
                    "current_platform": mla_indexer.current_platform,
                    "DeepseekV41IndexerBackend": mla_indexer.DeepseekV41IndexerBackend,
                    "prepare_kernel_block_sizes": (
                        worker_utils.prepare_kernel_block_sizes
                    ),
                    "MLAAttentionSpec": specs_module.MLAAttentionSpec,
                    "compute_layer_kv_cache_shape_bytes": (
                        specs_module.compute_layer_kv_cache_shape_bytes
                    ),
                    "get_max_prefill_buffer_size": (
                        mla_indexer.get_max_prefill_buffer_size
                    ),
                    "DeepseekV32IndexerMetadataBuilder": (
                        mla_indexer.DeepseekV32IndexerMetadataBuilder
                    ),
                },
            ),
            (
                "tests/v1/worker/test_gpu_model_runner.py",
                (
                    "_make_mock_backend_for_kernel_block_size",
                    "test_select_common_block_size_prefers_manager_block_size",
                    "test_select_common_block_size_uses_largest_shared_int",
                    "test_select_common_block_size_without_active_backends",
                    "test_select_common_block_size_no_valid_option",
                ),
                {
                    "MultipleOf": multiple_of,
                    "select_common_block_size": worker_utils.select_common_block_size,
                },
            ),
            (
                "tests/v1/attention/test_sparse_mla_backends.py",
                ("test_split_indexer_prefill_chunks",),
                {
                    "DeepseekV32IndexerMetadataBuilder": (
                        mla_indexer.DeepseekV32IndexerMetadataBuilder
                    )
                },
            ),
            (
                "tests/kernels/test_engram.py",
                ("test_engram_scales_also_use_registered_host_allocator",),
                {"nvidia_engram_ops": dsv_engram},
            ),
            (
                "tests/models/test_deepseek_v41_pipeline_parallel.py",
                ("test_", "config", "_model"),
                {
                    "dsv41": dsv_model,
                    "vl_model": dsv_vl,
                    "shared_cache_dependency": dsv_dependencies.shared_cache_dependency,
                },
            ),
            (
                "tests/v1/worker/test_dspark_utils.py",
                ("test_pp_dspark_loads_only",),
                {},
            ),
        ]
    )
    output = {}
    for index, (path, prefixes, extra) in enumerate(specs):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        functions = [
            n.name
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith(prefixes)
        ]
        classes = {
            n.name: (None, None)
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name in ("_Layer", "_DummyWork")
        }
        obj = load(
            f"agentvue_cpu.tests.case_{index}", path, functions, classes, **extra
        )
        output.update(
            {
                key: value
                for key, value in obj.__dict__.items()
                if key.startswith("test_")
            }
        )
    return output


if __name__ == "__main__":
    scratch = Path(
        os.environ.get("AGENTVUE_CPU_TEST_DIR", ROOT / ".cache" / "agentvue-cpu-tests")
    )
    scratch.mkdir(parents=True, exist_ok=True)
    test_file = scratch / "test_contracts.py"
    test_file.write_text(
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('agentvue_validate', "
        f"{str(Path(__file__).resolve())!r})\n"
        "runner = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(runner)\n"
        "globals().update(runner.build_tests())\n",
        encoding="utf-8",
    )
    print("Isolated CPU contracts only: runtime imports and GPU services are mocked.")
    raise SystemExit(
        pytest.main(
            [
                str(test_file),
                "--confcutdir=" + str(scratch),
                "--import-mode=importlib",
                "-q",
                *sys.argv[1:],
            ]
        )
    )
