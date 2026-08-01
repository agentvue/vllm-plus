from __future__ import annotations

import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    process_started = time.perf_counter()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_started = time.perf_counter()
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    init_seconds = time.perf_counter() - init_started

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(
        "PEER_RAIL_NCCL_INIT_OK "
        f"host={socket.gethostname()} rank={rank}/{world_size} "
        f"seconds={init_seconds:.3f}",
        flush=True,
    )
    expected = world_size * (world_size + 1) / 2

    value = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(value)
    torch.cuda.synchronize()
    if value.item() != expected:
        raise RuntimeError(
            f"all-reduce mismatch on rank {rank}: {value.item()} != {expected}"
        )

    element_count = 4 * 1024 * 1024
    payload = torch.ones(element_count, dtype=torch.float32, device="cuda")
    for _ in range(1):
        payload.fill_(1)
        dist.all_reduce(payload)
    torch.cuda.synchronize()

    iterations = 3
    started = time.perf_counter()
    for _ in range(iterations):
        payload.fill_(1)
        dist.all_reduce(payload)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if not torch.all(payload == world_size):
        raise RuntimeError(f"payload validation failed on rank {rank}")

    seconds_per_iteration = elapsed / iterations
    payload_bytes = payload.numel() * payload.element_size()
    algorithm_gbps = payload_bytes / seconds_per_iteration / 1e9
    print(
        "PEER_RAIL_SMOKE_OK "
        f"host={socket.gethostname()} rank={rank}/{world_size} "
        f"total_seconds={time.perf_counter() - process_started:.3f} "
        f"seconds_per_allreduce={seconds_per_iteration:.6f} "
        f"algorithm_GBps={algorithm_gbps:.3f}",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
