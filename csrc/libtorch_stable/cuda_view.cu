#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/version.h>
#include <cuda_runtime.h>

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "registered_host_memory.h"

#if defined(__linux__) && !defined(USE_ROCM)
torch::stable::Tensor allocate_registered_cpu_tensor(
    const std::vector<int64_t>& shape, torch::headeronly::ScalarType dtype,
    int64_t chunk_bytes) {
  STD_TORCH_CHECK(chunk_bytes > 0, "chunk_bytes must be positive");
  int64_t elements = 1;
  for (int64_t size : shape) {
    STD_TORCH_CHECK(size >= 0, "Negative PLE tensor dimension");
    STD_TORCH_CHECK(
        size == 0 || elements <= std::numeric_limits<int64_t>::max() / size,
        "PLE tensor size overflow");
    elements *= size;
  }
  const torch::stable::Device cpu_dev(torch::headeronly::DeviceType::CPU);
  auto empty = torch::stable::empty(std::vector<int64_t>{0}, dtype,
                                    std::nullopt, cpu_dev);
  size_t element_size = empty.element_size();
  STD_TORCH_CHECK(
      elements > 0 && static_cast<uint64_t>(elements) <=
                          std::numeric_limits<int64_t>::max() / element_size,
      "Empty or overflowing PLE allocation");
  auto owner = vllm::RegisteredHostMemory::allocate(elements * element_size,
                                                    chunk_bytes);
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (size_t dim = shape.size(); dim-- > 0;) {
    strides[dim] = stride;
    stride *= shape[dim];
  }
  return torch::stable::from_blob(owner->data(), shape, strides, cpu_dev, dtype,
                                  [owner](void*) {});
}
#endif

// This function assumes that `cpu_tensor` is a CPU tensor,
// and that UVA (Unified Virtual Addressing) is enabled.
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor) {
  STD_TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  const auto dtype = cpu_tensor.scalar_type();
  const auto layout = cpu_tensor.layout();
  const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA);

  // handle empty tensor
  if (cpu_tensor.numel() == 0) {
    return torch::stable::empty(cpu_tensor.sizes(), dtype, layout, cuda_dev);
  }

  std::array<StableIValue, 2> is_pinned_stack{
      torch::stable::detail::from(cpu_tensor),
      torch::stable::detail::from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::is_pinned", "", is_pinned_stack.data(), TORCH_ABI_VERSION));
  bool is_pinned = torch::stable::detail::to<bool>(is_pinned_stack[0]);
#if defined(__linux__) && !defined(USE_ROCM)
  if (auto owner =
          vllm::find_registered_host_memory(cpu_tensor.mutable_data_ptr())) {
    STD_TORCH_CHECK(is_pinned,
                    "Registered PLE storage is not pinned; "
                    "a full-table copy is forbidden");
    owner->track_current_device();
  }
#endif
  if (is_pinned) {
    // If CPU tensor is pinned, directly get the device pointer.
    void* host_ptr = const_cast<void*>(cpu_tensor.mutable_data_ptr());
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    STD_TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed: ",
                    cudaGetErrorString(err));

    return torch::stable::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(), cuda_dev, dtype,
        [base = cpu_tensor](void*) {});  // keep cpu tensor alive
  }

  // If CPU tensor is not pinned, allocate a new pinned memory buffer.
  torch::stable::Tensor contiguous_cpu = torch::stable::contiguous(cpu_tensor);
  size_t nbytes = contiguous_cpu.numel() * contiguous_cpu.element_size();

  void* host_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaHostAlloc failed: ", cudaGetErrorString(err));
  }

  err = cudaMemcpy(host_ptr, contiguous_cpu.const_data_ptr(), nbytes,
                   cudaMemcpyDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(false, "cudaMemcpy failed: ", cudaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(
        false, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

  auto deleter = [host_ptr](void*) { cudaFreeHost(host_ptr); };

  return torch::stable::from_blob(device_ptr, contiguous_cpu.sizes(),
                                  contiguous_cpu.strides(), cuda_dev,
                                  contiguous_cpu.scalar_type(), deleter);
}
