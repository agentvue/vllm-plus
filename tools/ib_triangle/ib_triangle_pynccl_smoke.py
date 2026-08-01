from __future__ import annotations

import argparse
import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_worker(local_rank: int, args: argparse.Namespace) -> None:
    rank = args.rank_offset + local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
        timeout=timedelta(seconds=180),
    )

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    communicator = PyNcclCommunicator(
        group=dist.group.WORLD,
        device=torch.device(f"cuda:{local_rank}"),
    )
    if not communicator.available or communicator.disabled:
        raise RuntimeError("PyNccl communicator was unexpectedly disabled")

    expected = args.world_size * (args.world_size + 1) / 2
    value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
    reduced = communicator.all_reduce(value)
    torch.cuda.synchronize(local_rank)
    if reduced.item() != expected:
        raise RuntimeError(
            f"PyNccl all-reduce mismatch on rank {rank}: "
            f"{reduced.item()} != {expected}"
        )

    payload = torch.ones(
        16 * 1024 * 1024,
        dtype=torch.float32,
        device=f"cuda:{local_rank}",
    )
    output = torch.empty_like(payload)
    for _ in range(3):
        communicator.all_reduce(payload, output)
    torch.cuda.synchronize(local_rank)

    iterations = 10
    started = time.perf_counter()
    for _ in range(iterations):
        communicator.all_reduce(payload, output)
    torch.cuda.synchronize(local_rank)
    elapsed = time.perf_counter() - started

    if not torch.all(output == args.world_size):
        raise RuntimeError(f"PyNccl payload validation failed on rank {rank}")

    payload_bytes = payload.numel() * payload.element_size()
    seconds_per_iteration = elapsed / iterations
    algorithm_gbps = payload_bytes / seconds_per_iteration / 1e9
    print(
        "PEER_RAIL_PYNCCL_SMOKE_OK "
        f"host={socket.gethostname()} rank={rank}/{args.world_size} "
        f"local_rank={local_rank} "
        f"seconds_per_allreduce={seconds_per_iteration:.6f} "
        f"algorithm_GBps={algorithm_gbps:.3f}",
        flush=True,
    )

    dist.barrier()
    communicator.destroy()
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--local-world-size", type=int, required=True)
    parser.add_argument("--rank-offset", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_world_size < 1:
        raise ValueError("local-world-size must be positive")
    if args.rank_offset < 0:
        raise ValueError("rank-offset must be non-negative")
    if args.rank_offset + args.local_world_size > args.world_size:
        raise ValueError("local rank range exceeds world-size")

    os.environ["VLLM_DISABLE_PYNCCL"] = "0"
    mp.spawn(run_worker, args=(args,), nprocs=args.local_world_size, join=True)


if __name__ == "__main__":
    main()
