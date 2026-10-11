# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for automatic and explicitly ordered Ray DP placement."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import vllm.v1.engine.utils as utils
from vllm.v1.engine.utils import CoreEngineActorManager
from vllm.v1.executor.ray_utils import parse_ray_dp_ordered_node_ips_json


@pytest.fixture(autouse=True)
def isolate_placement_env(monkeypatch):
    for name in (
        "VLLM_RAY_DP_ORDERED_NODE_IPS_JSON",
        "VLLM_RAY_ORDERED_NODE_IPS",
        "VLLM_RAY_BUNDLE_INDICES",
        "VLLM_RAY_DP_PLACEMENT_NODE_IPS",
    ):
        monkeypatch.delenv(name, raising=False)


def _vllm_config(
    *, dp_size, dp_local, master_ip, world_size=1, all2all_backend="naive"
):
    parallel = SimpleNamespace(
        data_parallel_master_ip=master_ip,
        data_parallel_size=dp_size,
        data_parallel_size_local=dp_local,
        world_size=world_size,
        all2all_backend=all2all_backend,
        data_parallel_backend="ray",
        enable_elastic_ep=False,
    )
    return SimpleNamespace(parallel_config=parallel)


def _resources(node_gpus: dict[str, int]):
    # node_gpus: {ip: gpu_count}; plus a CPU-only head node.
    res = {
        f"id-{ip}": {"GPU": float(g), "CPU": 128.0, f"node:{ip}": 1.0}
        for ip, g in node_gpus.items()
    }
    res["id-head"] = {
        "CPU": 8.0,
        "node:__internal_head__": 1.0,
        "node:10.9.9.9": 1.0,
    }
    return res


def _run(cfg, resources):
    created = []

    def fake_pg(name, strategy, bundles):
        created.append({"name": name, "strategy": strategy, "bundles": bundles})
        return object()

    with (
        patch(
            "ray._private.state.available_resources_per_node",
            return_value=resources,
        ),
        patch.object(utils, "current_platform", SimpleNamespace(ray_device_key="GPU")),
        patch("ray.util.placement_group", side_effect=fake_pg),
    ):
        pgs, local_ranks = CoreEngineActorManager.create_dp_placement_groups(cfg)
    return pgs, local_ranks, created


def _pinned_ips(created):
    return {
        key.split(":", 1)[1]
        for pg in created
        for bundle in pg["bundles"]
        for key in bundle
        if key.startswith("node:")
    }


def test_allowlist_confines_dp_to_listed_nodes(monkeypatch):
    monkeypatch.setenv("VLLM_RAY_DP_PLACEMENT_NODE_IPS", "10.0.0.1,10.0.0.3")
    resources = _resources({"10.0.0.1": 8, "10.0.0.2": 8, "10.0.0.3": 8, "10.0.0.4": 8})
    cfg = _vllm_config(dp_size=16, dp_local=8, master_ip="10.0.0.1")

    pgs, _, created = _run(cfg, resources)

    assert len(pgs) == 16  # 8 on .1 (master) + 8 on .3
    assert _pinned_ips(created) <= {"10.0.0.1", "10.0.0.3"}


def test_empty_allowlist_is_noop(monkeypatch):
    monkeypatch.delenv("VLLM_RAY_DP_PLACEMENT_NODE_IPS", raising=False)
    resources = _resources({"10.0.0.1": 8, "10.0.0.2": 8})
    cfg = _vllm_config(dp_size=16, dp_local=8, master_ip="10.0.0.1")

    pgs, _, created = _run(cfg, resources)

    assert len(pgs) == 16
    assert _pinned_ips(created) == {"10.0.0.1", "10.0.0.2"}  # all nodes used


def test_master_auto_added_with_warning(monkeypatch):
    # Allowlist omits the master; vLLM must still keep it and warn.
    monkeypatch.setenv("VLLM_RAY_DP_PLACEMENT_NODE_IPS", "10.0.0.3")
    resources = _resources({"10.0.0.1": 8, "10.0.0.3": 8})
    cfg = _vllm_config(dp_size=16, dp_local=8, master_ip="10.0.0.1")
    _, _, created = _run(cfg, resources)

    assert _pinned_ips(created) == {"10.0.0.1", "10.0.0.3"}


def test_allowlist_isolates_two_engines(monkeypatch):
    # Engine B is confined to .2/.4, so it can never touch engine A's master .1.
    monkeypatch.setenv("VLLM_RAY_DP_PLACEMENT_NODE_IPS", "10.0.0.2,10.0.0.4")
    resources = _resources({"10.0.0.1": 8, "10.0.0.2": 8, "10.0.0.3": 8, "10.0.0.4": 8})
    cfg = _vllm_config(dp_size=16, dp_local=8, master_ip="10.0.0.2")

    _, _, created = _run(cfg, resources)

    assert _pinned_ips(created) <= {"10.0.0.2", "10.0.0.4"}


def test_allowlist_too_small_raises(monkeypatch):
    # Master alone can't hold all ranks and no other node is allowed.
    monkeypatch.setenv("VLLM_RAY_DP_PLACEMENT_NODE_IPS", "10.0.0.1")
    resources = _resources({"10.0.0.1": 8, "10.0.0.2": 8})
    cfg = _vllm_config(dp_size=16, dp_local=8, master_ip="10.0.0.1")

    with pytest.raises(ValueError):  # not enough placement groups created
        _run(cfg, resources)


def _ordered_placements():
    return {
        "0": ["192.0.2.1"] * 4
        + ["192.0.2.2"] * 4
        + ["192.0.2.3"] * 4
        + ["192.0.2.2"] * 3
        + ["192.0.2.1"]
        + ["192.0.2.4"] * 4,
        "1": ["192.0.2.1"] * 4
        + ["192.0.2.2"] * 4
        + ["192.0.2.3"] * 4
        + ["192.0.2.1"]
        + ["192.0.2.3"] * 3
        + ["192.0.2.4"] * 4,
    }


def test_ordered_dp_uses_all_40_gpus_with_exact_pipeline_rank_order(monkeypatch):
    placements = _ordered_placements()
    monkeypatch.setenv("VLLM_RAY_DP_ORDERED_NODE_IPS_JSON", json.dumps(placements))
    monkeypatch.setenv("VLLM_RAY_DP_PACK_STRATEGY", "span")
    cfg = _vllm_config(dp_size=2, dp_local=2, master_ip="192.0.2.1", world_size=20)
    resources = _resources(
        {"192.0.2.1": 10, "192.0.2.2": 11, "192.0.2.3": 11, "192.0.2.4": 8}
    )

    groups, local_ranks, created = _run(cfg, resources)

    assert len(groups) == 2
    assert local_ranks == [0, 1]
    for rank, pg in enumerate(created):
        assert pg["strategy"] == "PACK"
        assert pg["bundles"][:-1] == [
            {"GPU": 1.0, f"node:{ip}": 0.001} for ip in placements[str(rank)]
        ]
        assert pg["bundles"][-1] == {"CPU": 1.0, "node:192.0.2.1": 0.001}


def test_ordered_dp_checks_combined_demand_before_reserving(monkeypatch):
    monkeypatch.setenv(
        "VLLM_RAY_DP_ORDERED_NODE_IPS_JSON",
        json.dumps(
            {
                "0": ["192.0.2.1"] * 4,
                "1": ["192.0.2.1"] * 4,
            }
        ),
    )
    cfg = _vllm_config(dp_size=2, dp_local=2, master_ip="192.0.2.1", world_size=4)
    with (
        patch("ray.util.placement_group") as create,
        patch(
            "ray._private.state.available_resources_per_node",
            return_value=_resources({"192.0.2.1": 7}),
        ),
        patch.object(utils, "current_platform", SimpleNamespace(ray_device_key="GPU")),
    ):
        with pytest.raises(ValueError, match="requests 8 GPUs"):
            CoreEngineActorManager.create_dp_placement_groups(cfg)
        create.assert_not_called()


@pytest.mark.parametrize(
    "value",
    [
        "{",
        "[]",
        "{}",
        '{"0": ["a", "b"]}',
        '{"0": ["a"], "1": ["b"]}',
        '{"0": ["a", "b"], "1": ["a", " "]}',
    ],
)
def test_ordered_dp_rejects_incomplete_or_malformed_maps(value):
    with pytest.raises(ValueError, match="VLLM_RAY_DP_ORDERED_NODE_IPS_JSON"):
        parse_ray_dp_ordered_node_ips_json(value, dp_size=2, world_size=2)


@pytest.mark.parametrize(
    "name", ["VLLM_RAY_ORDERED_NODE_IPS", "VLLM_RAY_BUNDLE_INDICES"]
)
def test_ordered_dp_rejects_conflicting_rank_placement(monkeypatch, name):
    monkeypatch.setenv(
        "VLLM_RAY_DP_ORDERED_NODE_IPS_JSON", json.dumps(_ordered_placements())
    )
    monkeypatch.setenv(name, "conflict")
    cfg = _vllm_config(dp_size=2, dp_local=2, master_ip="192.0.2.1", world_size=20)
    with pytest.raises(ValueError, match="cannot be combined"):
        _run(cfg, _resources({"192.0.2.1": 40}))


def test_ordered_dp_cleans_up_if_second_reservation_fails():
    first_pg = object()
    with (
        patch(
            "ray.util.placement_group", side_effect=[first_pg, RuntimeError("failed")]
        ),
        patch("ray.util.remove_placement_group") as remove,
    ):
        with pytest.raises(RuntimeError, match="failed"):
            utils._create_pinned_dp_placement_groups(
                [["192.0.2.1"], ["192.0.2.1"]],
                _resources({"192.0.2.1": 2}),
                "GPU",
            )
        remove.assert_called_once_with(first_pg)


def test_dp_actor_gets_its_own_rank_order_and_node_transport_env():
    base = {
        "LD_PRELOAD": "/head/nccl.so",
        "GLOO_SOCKET_IFNAME": "head_ib",
        "VLLM_RAY_DP_ORDERED_NODE_IPS_JSON": "global-map",
    }
    overrides = {
        "remote": {"LD_PRELOAD": "/remote/nccl.so", "GLOO_SOCKET_IFNAME": "remote_ib"}
    }
    pg = SimpleNamespace(bundle_specs=[{"GPU": 1}, {"node:remote": 0.001, "CPU": 1}])
    first = utils._get_dp_actor_env_vars(base, pg, 1, overrides, ["remote"])
    second = utils._get_dp_actor_env_vars(base, pg, 1, overrides, ["other"])
    assert first["LD_PRELOAD"] == "/remote/nccl.so"
    assert first["GLOO_SOCKET_IFNAME"] == "remote_ib"
    assert first["VLLM_RAY_ORDERED_NODE_IPS"] == "remote"
    assert second["VLLM_RAY_ORDERED_NODE_IPS"] == "other"
    assert "VLLM_RAY_DP_ORDERED_NODE_IPS_JSON" not in first
    assert "VLLM_RAY_ORDERED_NODE_IPS" not in base


def test_ray_v2_cpu_engine_does_not_slice_a_20_gpu_local_range():
    from vllm.v1.engine.core import EngineCoreActorMixin
    from vllm.v1.executor import Executor
    from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(assigned_physical_gpu_ids=[0])
    )
    actor = EngineCoreActorMixin.__new__(EngineCoreActorMixin)
    with (
        patch.object(Executor, "get_class", return_value=RayExecutorV2),
        patch.object(actor, "_set_assigned_physical_gpu_ids") as assign,
    ):
        actor._set_visible_devices(cfg, local_dp_rank=1)
        assign.assert_not_called()
    assert cfg.parallel_config.assigned_physical_gpu_ids is None


def test_dp_startup_failure_releases_all_engine_actors_and_reservations():
    from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

    cfg = _vllm_config(dp_size=2, dp_local=2, master_ip="192.0.2.1")
    cfg.model_config = SimpleNamespace(is_moe=True)
    cfg.instance_id = "test-dp"
    cfg.kv_transfer_config = None
    groups = [
        SimpleNamespace(bundle_specs=[{"GPU": 1}, {"CPU": 1, "node:192.0.2.1": 0.001}])
        for _ in range(2)
    ]
    actors = [MagicMock(), MagicMock()]
    factory = MagicMock()
    factory.options.return_value = factory
    factory.remote.side_effect = actors
    with (
        patch("ray.is_initialized", return_value=True),
        patch("ray.remote", return_value=factory),
        patch("ray.get", side_effect=RuntimeError("engine startup failed")),
        patch("ray.kill") as kill,
        patch("ray.util.remove_placement_group") as remove,
        patch.object(
            CoreEngineActorManager,
            "create_dp_placement_groups",
            return_value=(groups, [0, 1]),
        ),
        patch.object(utils, "get_env_vars_to_copy", return_value=set()),
        patch.object(utils, "current_platform", SimpleNamespace(is_xpu=lambda: False)),
    ):
        with pytest.raises(RuntimeError, match="engine startup failed"):
            CoreEngineActorManager(cfg, SimpleNamespace(), RayExecutorV2, False)
        assert [call.args[0] for call in kill.call_args_list] == actors
        assert [call.args[0] for call in remove.call_args_list] == groups
        configs = [call.kwargs["vllm_config"] for call in factory.remote.call_args_list]
        assert [config.instance_id for config in configs] == [
            "test-dp_dp0",
            "test-dp_dp1",
        ]
