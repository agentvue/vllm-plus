# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.executor.ray_utils import (
    get_bundles_for_node_ips,
    get_bundles_sorted_by_node,
    parse_ray_node_env_vars_json,
    parse_ray_ordered_node_ips,
)

NODE_A = "node_a"
NODE_B = "node_b"
NODE_C = "node_c"

IP_A = "10.0.0.1"
IP_B = "10.0.0.2"
IP_C = "10.0.0.3"

NODE_ID_TO_IP = {NODE_A: IP_A, NODE_B: IP_B, NODE_C: IP_C}

MOCK_RAY_NODES = [
    {"NodeID": NODE_A, "NodeManagerAddress": IP_A, "Alive": True},
    {"NodeID": NODE_B, "NodeManagerAddress": IP_B, "Alive": True},
    {"NodeID": NODE_C, "NodeManagerAddress": IP_C, "Alive": True},
]


@pytest.mark.parametrize(
    "bundles_to_node_id,bundle_specs,expected",
    [
        pytest.param(
            {0: NODE_C, 1: NODE_A, 2: NODE_B, 3: NODE_C, 4: NODE_A, 5: NODE_B},
            [{"GPU": 1}] * 6,
            [
                (1, NODE_A, IP_A),
                (4, NODE_A, IP_A),
                (2, NODE_B, IP_B),
                (5, NODE_B, IP_B),
                (0, NODE_C, IP_C),
                (3, NODE_C, IP_C),
            ],
        ),
        pytest.param(
            {0: NODE_B, 1: NODE_B, 2: NODE_A, 3: NODE_A},
            [{"GPU": 1}] * 4,
            [
                (2, NODE_A, IP_A),
                (3, NODE_A, IP_A),
                (0, NODE_B, IP_B),
                (1, NODE_B, IP_B),
            ],
        ),
        pytest.param(
            {0: NODE_C, 1: NODE_B, 2: NODE_C, 3: NODE_B},
            [{"GPU": 1}] * 4,
            [
                (1, NODE_B, IP_B),
                (3, NODE_B, IP_B),
                (0, NODE_C, IP_C),
                (2, NODE_C, IP_C),
            ],
        ),
        pytest.param(
            {0: NODE_A, 1: NODE_A, 2: NODE_A},
            [{"GPU": 1}] * 3,
            [(0, NODE_A, IP_A), (1, NODE_A, IP_A), (2, NODE_A, IP_A)],
        ),
        pytest.param(
            {},
            [],
            [],
        ),
        pytest.param(
            {0: NODE_A, 1: NODE_B, 2: NODE_A},
            [{"CPU": 1}, {"GPU": 1}, {"GPU": 1}],
            [(2, NODE_A, IP_A), (1, NODE_B, IP_B)],
        ),
    ],
)
def test_get_bundles_sorted_by_node(bundles_to_node_id, bundle_specs, expected):
    mock_pg = MagicMock()
    mock_pg.bundle_specs = bundle_specs

    mock_ctx = MagicMock()
    mock_ctx.get_node_id.return_value = NODE_A

    with (
        patch(
            "vllm.v1.executor.ray_utils.placement_group_table",
            return_value={"bundles_to_node_id": bundles_to_node_id},
        ),
        patch("vllm.v1.executor.ray_utils.ray") as mock_ray,
        patch("vllm.v1.executor.ray_utils.current_platform") as mock_platform,
    ):
        mock_ray.get_runtime_context.return_value = mock_ctx
        mock_ray.nodes.return_value = MOCK_RAY_NODES
        mock_platform.ray_device_key = "GPU"

        result = get_bundles_sorted_by_node(mock_pg)

    assert result == expected


def test_parse_ray_ordered_node_ips():
    result = parse_ray_ordered_node_ips(" 10.0.0.1,10.0.0.2,, 10.0.0.3 ")

    assert result == [IP_A, IP_B, IP_C]


def test_parse_ray_node_env_vars_json():
    result = parse_ray_node_env_vars_json(
        '{"10.0.0.1":{"VLLM_HOST_IP":"10.0.0.1","NCCL_IB_HCA":"mlx5_0:1"}}'
    )

    assert result == {
        IP_A: {
            "VLLM_HOST_IP": IP_A,
            "NCCL_IB_HCA": "mlx5_0:1",
        }
    }


@pytest.mark.parametrize(
    "value",
    [
        "[1, 2, 3]",
        '{"10.0.0.1":["NCCL_IB_HCA"]}',
        '{"10.0.0.1":{"NCCL_IB_HCA":1}}',
    ],
)
def test_parse_ray_node_env_vars_json_rejects_invalid_shapes(value):
    with pytest.raises(ValueError, match="VLLM_RAY_NODE_ENV_VARS_JSON"):
        parse_ray_node_env_vars_json(value)


def test_get_bundles_for_node_ips():
    mock_pg = MagicMock()
    mock_pg.bundle_specs = [{"CPU": 1}, {"GPU": 1}, {"GPU": 1}, {"GPU": 1}]

    with (
        patch(
            "vllm.v1.executor.ray_utils.placement_group_table",
            return_value={"bundles_to_node_id": {1: NODE_B, 2: NODE_A, 3: NODE_C}},
        ),
        patch("vllm.v1.executor.ray_utils.ray") as mock_ray,
        patch("vllm.v1.executor.ray_utils.current_platform") as mock_platform,
    ):
        mock_ray.nodes.return_value = MOCK_RAY_NODES
        mock_platform.ray_device_key = "GPU"

        result = get_bundles_for_node_ips(
            mock_pg, [IP_B, IP_A, IP_C], world_size=3
        )

    assert result == [(1, NODE_B, IP_B), (2, NODE_A, IP_A), (3, NODE_C, IP_C)]


def test_get_bundles_for_node_ips_rejects_mismatch():
    mock_pg = MagicMock()
    mock_pg.bundle_specs = [{"GPU": 1}]

    with (
        patch(
            "vllm.v1.executor.ray_utils.placement_group_table",
            return_value={"bundles_to_node_id": {0: NODE_A}},
        ),
        patch("vllm.v1.executor.ray_utils.ray") as mock_ray,
        patch("vllm.v1.executor.ray_utils.current_platform") as mock_platform,
    ):
        mock_ray.nodes.return_value = MOCK_RAY_NODES
        mock_platform.ray_device_key = "GPU"

        with pytest.raises(RuntimeError, match="placement mismatch"):
            get_bundles_for_node_ips(mock_pg, [IP_B], world_size=1)
