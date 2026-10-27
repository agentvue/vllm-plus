# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.distributed.aux_output_connector.connector import (
    AuxRequestOutput,
)
from vllm.v1.executor import ray_utils
from vllm.v1.executor.ray_utils import detach_zero_copy_from_model_runner_output
from vllm.v1.outputs import (
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
)


def _make_readonly(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def test_detach_zero_copy_from_model_runner_output_copies_only_numpy_views():
    cu_num_generated_tokens = [0, 2]
    prompt_logprobs = LogprobsTensors.empty_cpu(1, 2)
    output = ModelRunnerOutput(
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        logprobs=LogprobsLists(
            logprob_token_ids=_make_readonly(
                np.array([[1, 2], [3, 4]], dtype=np.int32)
            ),
            logprobs=_make_readonly(
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
            ),
            sampled_token_ranks=_make_readonly(np.array([1, 2], dtype=np.int32)),
            cu_num_generated_tokens=cu_num_generated_tokens,
        ),
        prompt_logprobs_dict={"req-0": prompt_logprobs},
    )

    original_logprobs = output.logprobs
    assert original_logprobs is not None

    detach_zero_copy_from_model_runner_output(output)

    detached_logprobs = output.logprobs
    assert detached_logprobs is not None
    assert detached_logprobs is not original_logprobs
    assert (
        detached_logprobs.logprob_token_ids is not original_logprobs.logprob_token_ids
    )
    assert detached_logprobs.logprobs is not original_logprobs.logprobs
    assert (
        detached_logprobs.sampled_token_ranks
        is not original_logprobs.sampled_token_ranks
    )
    assert detached_logprobs.logprob_token_ids.flags.writeable
    assert detached_logprobs.logprobs.flags.writeable
    assert detached_logprobs.sampled_token_ranks.flags.writeable
    assert detached_logprobs.cu_num_generated_tokens is cu_num_generated_tokens
    assert output.prompt_logprobs_dict["req-0"] is prompt_logprobs


def test_detach_zero_copy_aux_output_without_logprobs():
    rows = _make_readonly(np.arange(12, dtype=np.uint8).reshape(2, 3, 2))
    output = ModelRunnerOutput(
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        aux_output_connector_output={"req-0": AuxRequestOutput(0, rows)},
    )

    detach_zero_copy_from_model_runner_output(output)

    aux_output = output.aux_output_connector_output
    assert aux_output is not None
    detached = aux_output["req-0"].rows
    assert detached is not rows
    assert detached.flags.writeable
    np.testing.assert_array_equal(detached, rows)


def test_ordered_node_ips_preserve_multiple_gpus_per_node():
    assert ray_utils.parse_ray_ordered_node_ips(" a, a, ,b ") == ["a", "a", "b"]


@pytest.mark.parametrize(
    "value",
    ["{", "[]", '{"a": []}', '{"": {}}', '{"a": {"": "x"}}', '{"a": {"NCCL_NET": 1}}'],
)
def test_node_env_rejects_invalid_schema(value):
    with pytest.raises(ValueError, match="VLLM_RAY_NODE_ENV_VARS_JSON"):
        ray_utils.parse_ray_node_env_vars_json(value)


def test_node_env_retains_node_specific_network_settings():
    assert ray_utils.parse_ray_node_env_vars_json(
        '{"a": {"NCCL_IB_HCA": "mlx5_0"}, "b": {"NCCL_IB_HCA": "mlx5_1"}}'
    ) == {"a": {"NCCL_IB_HCA": "mlx5_0"}, "b": {"NCCL_IB_HCA": "mlx5_1"}}


@pytest.mark.parametrize(
    "ips,world_size,error",
    [
        (["a"], 2, "same size"),
        (["a", "dead"], 2, "not alive"),
        (["a", "a", "a"], 3, "only has 2"),
        (["b", "a", "a"], 3, None),
    ],
)
def test_ordered_nodes_validate_capacity(monkeypatch, ips, world_size, error):
    nodes = [
        {"Alive": True, "NodeID": ip, "NodeManagerAddress": ip} for ip in ["a", "b"]
    ]
    monkeypatch.setattr(ray_utils, "ray", SimpleNamespace(nodes=lambda: nodes))
    monkeypatch.setattr(
        ray_utils,
        "available_resources_per_node",
        lambda: {"a": {"GPU": 2}, "b": {"GPU": 1}},
        raising=False,
    )
    if error:
        with pytest.raises(ValueError, match=error):
            ray_utils._validate_ray_ordered_node_ips(ips, world_size, "GPU")
    else:
        ray_utils._validate_ray_ordered_node_ips(ips, world_size, "GPU")


def test_ordered_bundles_reject_wrong_rank_placement(monkeypatch):
    nodes = [
        {"Alive": True, "NodeID": ip, "NodeManagerAddress": ip} for ip in ["a", "b"]
    ]
    monkeypatch.setattr(ray_utils, "ray", SimpleNamespace(nodes=lambda: nodes))
    monkeypatch.setattr(
        ray_utils, "current_platform", SimpleNamespace(ray_device_key="GPU")
    )
    monkeypatch.setattr(
        ray_utils,
        "placement_group_table",
        lambda _: {"bundles_to_node_id": {1: "b", 2: "a", 3: "a"}},
        raising=False,
    )
    pg = SimpleNamespace(bundle_specs=[{}, {"GPU": 1}, {"GPU": 1}, {"GPU": 1}])
    assert ray_utils.get_bundles_for_node_ips(pg, ["b", "a", "a"], 3) == [
        (1, "b", "b"),
        (2, "a", "a"),
        (3, "a", "a"),
    ]
    with pytest.raises(RuntimeError, match="rank 0"):
        ray_utils.get_bundles_for_node_ips(pg, ["a", "b", "a"], 3)
