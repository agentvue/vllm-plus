// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#if defined(__linux__) && !defined(USE_ROCM)
  #include <cuda_runtime.h>
  #include <sys/mman.h>
  #include <unistd.h>

  #include <algorithm>
  #include <cerrno>
  #include <cstdint>
  #include <cstdio>
  #include <cstring>
  #include <limits>
  #include <map>
  #include <memory>
  #include <mutex>
  #include <stdexcept>
  #include <string>
  #include <vector>

namespace vllm {

// CUDA views retain the CPU storage; this registry holds only weak references.
class RegisteredHostMemory;
struct RegisteredHostRegistry {
  std::mutex mutex;
  std::map<uintptr_t, std::weak_ptr<RegisteredHostMemory>> allocations;
};

inline RegisteredHostRegistry& registered_host_registry() {
  // Storage deleters may run during static teardown.
  static auto* registry = new RegisteredHostRegistry;
  return *registry;
}

inline void check_host_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

class RegisteredHostMemory {
 public:
  static constexpr size_t max_chunk_bytes = 512ULL * 1024 * 1024;

  static std::shared_ptr<RegisteredHostMemory> allocate(size_t bytes,
                                                        size_t chunk_bytes) {
    auto owner =
        std::shared_ptr<RegisteredHostMemory>(new RegisteredHostMemory);
    owner->initialize(bytes, chunk_bytes);
    auto& registry = registered_host_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.allocations.emplace(reinterpret_cast<uintptr_t>(owner->data_),
                                 owner);
    return owner;
  }

  ~RegisteredHostMemory() noexcept {
    if (data_ == MAP_FAILED) return;
    auto& registry = registered_host_registry();
    {
      std::lock_guard<std::mutex> lock(registry.mutex);
      registry.allocations.erase(reinterpret_cast<uintptr_t>(data_));
    }
    int previous_device;
    if (cudaGetDevice(&previous_device) != cudaSuccess) {
      retain_mapping("cannot obtain current CUDA device");
      return;
    }
    bool safe = true;
    for (int device : devices_) {
      if (cudaSetDevice(device) != cudaSuccess ||
          cudaDeviceSynchronize() != cudaSuccess) {
        safe = false;
        break;
      }
    }
    if (safe) {
      for (auto it = addresses_.rbegin(); it != addresses_.rend(); ++it) {
        if (cudaHostUnregister(*it) != cudaSuccess) {
          safe = false;
          break;
        }
      }
    }
    if (cudaSetDevice(previous_device) != cudaSuccess) {
      std::fprintf(stderr,
                   "vLLM PLE: cannot restore CUDA device at teardown\n");
    }
    if (safe) {
      if (munmap(data_, mapped_bytes_) != 0) {
        retain_mapping("munmap failed");
      }
    } else {
      retain_mapping("CUDA synchronization or host unregistration failed");
    }
  }

  void* data() const { return data_; }
  size_t bytes() const { return bytes_; }

  void track_current_device() {
    int device;
    check_host_cuda(cudaGetDevice(&device), "cudaGetDevice");
    std::lock_guard<std::mutex> lock(devices_mutex_);
    if (std::find(validated_devices_.begin(), validated_devices_.end(),
                  device) != validated_devices_.end()) {
      return;
    }
    // Record before validation so teardown drains this device on exceptions.
    if (std::find(devices_.begin(), devices_.end(), device) == devices_.end()) {
      devices_.push_back(device);
    }
    void* base_device_ptr = nullptr;
    check_host_cuda(cudaHostGetDevicePointer(&base_device_ptr, data_, 0),
                    "cudaHostGetDevicePointer");
    for (size_t offset = 0; offset < mapped_bytes_; offset += chunk_bytes_) {
      size_t length = std::min(chunk_bytes_, mapped_bytes_ - offset);
      for (size_t position : {offset, offset + length - 1}) {
        void* device_ptr = nullptr;
        check_host_cuda(
            cudaHostGetDevicePointer(&device_ptr,
                                     static_cast<char*>(data_) + position, 0),
            "cudaHostGetDevicePointer at PLE chunk boundary");
        if (reinterpret_cast<uintptr_t>(device_ptr) !=
            reinterpret_cast<uintptr_t>(base_device_ptr) + position) {
          throw std::runtime_error(
              "PLE chunks do not have a contiguous CUDA "
              "mapping; a full-table copy is forbidden");
        }
      }
    }
    validated_devices_.push_back(device);
  }

 private:
  RegisteredHostMemory() = default;

  void initialize(size_t bytes, size_t chunk_bytes) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (bytes == 0 || page_size <= 0 || chunk_bytes == 0 ||
        chunk_bytes > max_chunk_bytes || chunk_bytes % page_size != 0 ||
        bytes > std::numeric_limits<size_t>::max() - (page_size - 1)) {
      throw std::invalid_argument(
          "Invalid PLE allocation or registration size");
    }
    bytes_ = bytes;
    chunk_bytes_ = chunk_bytes;
    mapped_bytes_ = ((bytes + page_size - 1) / page_size) * page_size;
    // Reserve metadata before registering anything: no allocation may fail
    // between a successful registration and recording its address.
    addresses_.reserve((mapped_bytes_ - 1) / chunk_bytes_ + 1);
    int device;
    check_host_cuda(cudaGetDevice(&device), "cudaGetDevice");
    data_ = mmap(nullptr, mapped_bytes_, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data_ == MAP_FAILED) {
      throw std::runtime_error(std::string("PLE mmap failed: ") +
                               std::strerror(errno));
    }
  #ifdef MADV_NOHUGEPAGE
    // This advice is allocation-local; never change system THP settings.
    madvise(data_, mapped_bytes_, MADV_NOHUGEPAGE);
  #endif
    for (size_t offset = 0; offset < mapped_bytes_; offset += chunk_bytes_) {
      size_t length = std::min(chunk_bytes_, mapped_bytes_ - offset);
      auto* pointer = static_cast<char*>(data_) + offset;
      // Anonymous pages are zero-filled. Fault each page without rewriting
      // the entire embedding table before checkpoint loading.
      auto* pages = reinterpret_cast<volatile char*>(pointer);
      for (size_t position = 0; position < length; position += page_size) {
        pages[position] = 0;
      }
      cudaError_t result = cudaHostRegister(
          pointer, length, cudaHostRegisterPortable | cudaHostRegisterMapped);
      if (result != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error("PLE cudaHostRegister failed at byte " +
                                 std::to_string(offset) + ": " +
                                 cudaGetErrorString(result));
      }
      addresses_.push_back(pointer);
    }
    track_current_device();
  }

  void retain_mapping(const char* reason) const noexcept {
    std::fprintf(
        stderr, "vLLM PLE: retaining %zu mapped bytes until process exit: %s\n",
        mapped_bytes_, reason);
  }

  void* data_ = MAP_FAILED;
  size_t bytes_ = 0;
  size_t mapped_bytes_ = 0;
  size_t chunk_bytes_ = 0;
  std::vector<void*> addresses_;
  std::vector<int> devices_;
  std::vector<int> validated_devices_;
  std::mutex devices_mutex_;
};

inline std::shared_ptr<RegisteredHostMemory> find_registered_host_memory(
    void* pointer) {
  auto& registry = registered_host_registry();
  uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
  uintptr_t start;
  std::weak_ptr<RegisteredHostMemory> allocation;
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto it = registry.allocations.upper_bound(address);
    if (it == registry.allocations.begin()) return nullptr;
    --it;
    start = it->first;
    allocation = it->second;
  }
  auto owner = allocation.lock();
  if (owner && address - start < owner->bytes()) return owner;
  return nullptr;
}

}  // namespace vllm
#endif
