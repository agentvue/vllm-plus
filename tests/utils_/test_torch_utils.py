# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.utils.torch_utils import (
    OMP_NUM_THREADS_SET_BY_VLLM,
    async_tensor_h2d,
    available_cpu_count,
    common_broadcastable_dtype,
    current_stream,
    get_kv_cache_torch_dtype,
    is_lossless_cast,
    is_quantized_kv_cache,
    set_default_torch_dtype,
    set_torch_threads_for_runtime,
    startup_omp_num_threads,
)


def test_nvfp4_4over6_cache_dtype() -> None:
    from vllm.config.cache import CacheConfig
    from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

    cache_config = CacheConfig(cache_dtype="nvfp4_4over6")

    assert cache_config.cache_dtype == "nvfp4_4over6"
    assert get_kv_cache_torch_dtype(cache_config.cache_dtype) == torch.uint8
    assert is_quantized_kv_cache(cache_config.cache_dtype)
    assert get_kv_quant_mode(cache_config.cache_dtype) == KVQuantMode.NVFP4


def test_set_default_torch_dtype_restores_dtype_after_exception() -> None:
    original_dtype = torch.get_default_dtype()

    with (
        pytest.raises(RuntimeError, match="expected failure"),
        set_default_torch_dtype(torch.bfloat16),
    ):
        assert torch.get_default_dtype() == torch.bfloat16
        raise RuntimeError("expected failure")

    assert torch.get_default_dtype() == original_dtype


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(),
    reason="requires Linux/CUDA native registered-host allocator",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("device_index", [0, 1])
def test_registered_cpu_storage_survives_aliases_and_cuda_views(dtype, device_index):
    """Parameter wrapping and reloads retain registered storage at chunk boundaries."""
    if device_index >= torch.cuda.device_count():
        pytest.skip("requires a second CUDA device")
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    current_platform.import_kernels()
    chunk_bytes = 4096
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        for _ in range(3):
            storage = torch.ops._C.allocate_registered_cpu_tensor(
                [12288], dtype, chunk_bytes
            )
            assert storage.is_pinned()
            expected = (torch.arange(12288) % 7).to(dtype)
            storage.copy_(expected)
            parameter = torch.nn.Parameter(storage, requires_grad=False)
            alias = parameter[1:]
            view = get_accelerator_view_from_cpu_tensor(parameter)
            assert view.device == device
            pointer = parameter.data_ptr()
            del storage, parameter
            gc.collect()
            # Reload in place, with the same operation used by PLE loaders.
            replacement = (torch.arange(12287) % 5).to(dtype)
            alias.copy_(replacement)
            expected[1:].copy_(replacement)
            assert alias.data_ptr() == pointer + alias.element_size()
            indices = torch.tensor(
                [0, 2047, 2048, 4095, 4096, 8191, 8192, 12287], device=device
            )
            del alias
            gc.collect()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = view.float().clone()
            graph.replay()
            torch.testing.assert_close(
                captured[indices].cpu(),
                expected[indices.cpu()].float(),
                rtol=0,
                atol=0,
            )
            del graph, captured, view
            gc.collect()


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(),
    reason="requires Linux/CUDA native registered-host allocator",
)
def test_registered_cpu_storage_rejects_invalid_registration_sizes():
    from vllm.platforms import current_platform

    current_platform.import_kernels()
    for chunk_bytes in [0, 1, 512 * 1024**2 + 4096]:
        with pytest.raises((RuntimeError, ValueError)):
            torch.ops._C.allocate_registered_cpu_tensor(
                [8192], torch.bfloat16, chunk_bytes
            )


@pytest.mark.skipif(
    sys.platform not in ("linux", "win32"), reason="requires Linux or Windows"
)
def test_registered_host_owner_rolls_back_and_drains_before_unmapping(tmp_path):
    """Test the C++ owner with mocked CUDA; Windows also adapts POSIX mapping."""
    if sys.platform == "win32":
        if importlib.util.find_spec("ziglang") is None:
            pytest.skip("Windows mock-owner check requires optional ziglang compiler")
        compiler = [sys.executable, "-m", "ziglang", "c++"]
        (tmp_path / "sys").mkdir()
        (tmp_path / "unistd.h").write_text(
            "#pragma once\n#define _SC_PAGESIZE 1\n"
            "inline long sysconf(int) { return 4096; }\n",
            encoding="utf-8",
        )
        (tmp_path / "sys/mman.h").write_text(
            "#pragma once\n#include <windows.h>\n"
            "#define PROT_READ 1\n#define PROT_WRITE 2\n"
            "#define MAP_PRIVATE 1\n#define MAP_ANONYMOUS 2\n"
            "#define MADV_NOHUGEPAGE 1\n"
            "#define MAP_FAILED reinterpret_cast<void*>(-1)\n"
            "inline void* mmap(void*, size_t n, int, int, int, int) {\n"
            "  void* p = VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE,\n"
            "                         PAGE_READWRITE);\n"
            "  return p ? p : MAP_FAILED;\n}\n"
            "inline int munmap(void* p, size_t) {\n"
            "  return VirtualFree(p, 0, MEM_RELEASE) ? 0 : -1;\n}\n"
            "inline int madvise(void*, size_t, int) { return 0; }\n",
            encoding="utf-8",
        )
    else:
        compiler_path = shutil.which("c++")
        if compiler_path is None:
            pytest.skip("requires a C++ compiler")
        compiler = [compiler_path]
    root = Path(__file__).resolve().parents[2]
    # The harness defines the CUDA API and its failure controls itself.
    (tmp_path / "cuda_runtime.h").write_text("#pragma once\n", encoding="utf-8")
    executable = tmp_path / (
        "registered_host_test.exe"
        if sys.platform == "win32"
        else "registered_host_test"
    )
    environment = dict(
        os.environ,
        ZIG_GLOBAL_CACHE_DIR=str(tmp_path / "zig-global"),
        ZIG_LOCAL_CACHE_DIR=str(tmp_path / "zig-local"),
    )
    subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-pthread",
            "-I",
            str(tmp_path),
            "-I",
            str(root / "csrc/libtorch_stable"),
            str(Path(__file__).with_name("registered_host_memory_test.cpp")),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    for scenario in [
        "rollback",
        "mapping",
        "alias",
        "retry_mapping",
        "sync_failure",
        "unregister_failure",
        "adaptive_invalid",
        "adaptive_memory",
        "adaptive_rollback",
        "adaptive_no_retry",
        "adaptive_mapping",
    ]:
        subprocess.run(
            [str(executable), scenario],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )


@pytest.mark.parametrize(
    ("src_dtype", "tgt_dtype", "expected_result"),
    [
        # Different precision_levels
        (torch.bool, torch.int8, True),
        (torch.bool, torch.float16, True),
        (torch.bool, torch.complex32, True),
        (torch.int64, torch.bool, False),
        (torch.int64, torch.float16, True),
        (torch.int64, torch.complex32, True),
        (torch.float64, torch.bool, False),
        (torch.float64, torch.int8, False),
        (torch.float64, torch.complex32, True),
        (torch.complex128, torch.bool, False),
        (torch.complex128, torch.int8, False),
        (torch.complex128, torch.float16, False),
        # precision_level=0
        (torch.bool, torch.bool, True),
        # precision_level=1
        (torch.int8, torch.int16, True),
        (torch.int16, torch.int8, False),
        (torch.uint8, torch.int8, False),
        (torch.int8, torch.uint8, False),
        # precision_level=2
        (torch.float16, torch.float32, True),
        (torch.float32, torch.float16, False),
        (torch.bfloat16, torch.float32, True),
        (torch.float32, torch.bfloat16, False),
        # precision_level=3
        (torch.complex32, torch.complex64, True),
        (torch.complex64, torch.complex32, False),
    ],
)
def test_is_lossless_cast(src_dtype, tgt_dtype, expected_result):
    assert is_lossless_cast(src_dtype, tgt_dtype) == expected_result


@pytest.mark.parametrize(
    ("dtypes", "expected_result"),
    [
        ([torch.bool], torch.bool),
        ([torch.bool, torch.int8], torch.int8),
        ([torch.bool, torch.int8, torch.float16], torch.float16),
        ([torch.bool, torch.int8, torch.float16, torch.complex32], torch.complex32),  # noqa: E501
    ],
)
def test_common_broadcastable_dtype(dtypes, expected_result):
    assert common_broadcastable_dtype(dtypes) == expected_result


def _test_stream_thread(main_expected_stream: torch.cuda.Stream):
    import threading

    child_stream = torch.cuda.Stream()
    thread_stream_ready = threading.Event()
    thread_can_exit = threading.Event()

    def child_thread_func():
        with torch.cuda.stream(child_stream):
            thread_stream_ready.set()
            thread_can_exit.wait(timeout=10)

    child_thread = threading.Thread(target=child_thread_func)
    child_thread.start()

    try:
        assert thread_stream_ready.wait(timeout=5), (
            "Child thread failed to enter stream context in time"
        )

        main_current_stream = current_stream()

        assert main_current_stream != child_stream, (
            "Main thread's current_stream was contaminated by child thread"
        )
        assert main_current_stream == main_expected_stream, (
            f"Main thread's stream changed unexpectedly. "
            f"Expected {main_expected_stream}, got {main_current_stream}"
        )

        thread_can_exit.set()

    finally:
        child_thread.join(timeout=5)
        if child_thread.is_alive():
            pytest.fail("Child thread failed to exit properly")


def test_current_stream_multithread():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    main_dedicated_stream = current_stream()

    assert main_dedicated_stream.cuda_stream != 0, (
        "ROCm/CUDA should create a dedicated stream, not use default stream (0x0)"
    )

    main_stream_again = current_stream()
    assert main_stream_again == main_dedicated_stream, (
        "Multiple calls to current_stream should return the same dedicated stream"
    )

    _test_stream_thread(main_dedicated_stream)


@pytest.fixture
def restore_torch_threads(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    original = torch.get_num_threads()
    yield
    torch.set_num_threads(original)


def test_startup_omp_num_threads_divides_between_local_workers():
    """Workers share the node's usable CPUs rather than each taking them all."""
    available = available_cpu_count()
    if available < 4:
        pytest.skip("needs at least 4 usable CPUs")
    assert startup_omp_num_threads(1) == available
    assert startup_omp_num_threads(2) == available // 2
    # Never zero, however many workers share the node.
    assert startup_omp_num_threads(available * 4) == 1


def test_set_torch_threads_for_runtime(restore_torch_threads):
    torch.set_num_threads(max(2, available_cpu_count()))
    set_torch_threads_for_runtime()
    assert torch.get_num_threads() == 1


def test_runtime_threads_respect_user_omp_num_threads(
    restore_torch_threads, monkeypatch: pytest.MonkeyPatch
):
    """An externally-set OMP_NUM_THREADS is the user's choice; leave it alone."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    torch.set_num_threads(3)
    set_torch_threads_for_runtime()
    assert torch.get_num_threads() == 3


def test_runtime_threads_override_vllm_set_omp_num_threads(
    restore_torch_threads, monkeypatch: pytest.MonkeyPatch
):
    """The value vLLM picked for worker startup is dropped once serving starts."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv(OMP_NUM_THREADS_SET_BY_VLLM, "1")
    torch.set_num_threads(3)
    set_torch_threads_for_runtime()
    assert torch.get_num_threads() == 1


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_async_tensor_h2d_staging(device):
    """Staging must preserve values while normalizing dtype and layout."""
    # Non-contiguous source with a dtype conversion: would previously sync.
    src = torch.arange(12, dtype=torch.int64).reshape(3, 4).T
    result = async_tensor_h2d(src, device=device, dtype=torch.int32)
    assert result.dtype == torch.int32
    assert result.is_contiguous()
    assert torch.equal(result.cpu(), src.to(torch.int32))

    # numpy and list sources still work.
    assert torch.equal(
        async_tensor_h2d(np.arange(4, dtype=np.int64), device=device),
        torch.arange(4, device=device),
    )
    assert torch.equal(
        async_tensor_h2d([1, 2, 3], device=device, dtype=torch.int32),
        torch.tensor([1, 2, 3], dtype=torch.int32, device=device),
    )
