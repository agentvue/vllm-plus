# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run isolated CPU contract tests when the vLLM native runtime is unavailable.

Compiles selected definitions from the working tree unchanged. External runtime
services are replaced by mocks. This checks Python behavior only; it does not
validate package imports, CUDA/Triton kernels, checkpoints, or serving.
Run in a fresh process: this deliberately installs temporary module stubs.
"""

from __future__ import annotations
import __future__

import ast
import copy
import dataclasses
import enum
import gc
import json
import logging
import os
import sys
import weakref
from collections import Counter, deque, namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
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
    copy=copy,
    enum=enum,
    gc=gc,
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
            "BatchExecutionDescriptor": (None, None),
        },
        CUDAGraphMode=compilation.CUDAGraphMode,
        dataclass=dataclasses.dataclass,
        graph_capture=None,
        is_global_first_rank=None,
        **common,
    )
    descriptor = load(
        "vllm.forward_context",
        "vllm/forward_context.py",
        classes={"BatchDescriptor": (None, None)},
        dataclass=dataclasses.dataclass,
    )
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
    spec_tree = ast.parse((ROOT / "vllm/v1/kv_cache_interface.py").read_text())
    specs_module = load(
        "vllm.v1.kv_cache_interface",
        "vllm/v1/kv_cache_interface.py",
        classes={
            n.name: (None, None) for n in spec_tree.body if isinstance(n, ast.ClassDef)
        },
        functions=("_apply_alignment_padding",),
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
        ),
        cdiv=lambda a, b: (a + b - 1) // b,
        replace=dataclasses.replace,
        get_hisparse_kv_cache_groups=lambda *args: None,
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
    load(
        "vllm.v1.worker.gpu.spec_decode.dflash.utils",
        "vllm/v1/worker/gpu/spec_decode/dflash/utils.py",
        functions=("_load_pp_target_embedding",),
        get_pp_group=None,
        DefaultModelLoader=None,
        PPMissingLayer=type("Missing", (), {}),
    )
    specs = [
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
            ("test_qwen4_exp_mtp_returns_sample", "test_local_mtp_on_last"),
            {},
        ),
        (
            "tests/v1/core/test_kv_cache_utils.py",
            (
                "test_glm5_draft_states_are_counted",
                "test_glm5_empty_projected_groups",
                "test_glm5_window_draft",
                "new_mamba_spec",
                "new_mla_spec",
                "_glm5_like_kv_cache_spec",
            ),
            {
                "kv_cache_utils": kv,
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
    scratch = ROOT / ".cache" / "agentvue-cpu-tests"
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
