// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// CPU fault injection for the production owner, without CUDA or GPU kernels.
#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <unistd.h>

using cudaError_t = int;
constexpr int cudaSuccess = 0;
constexpr unsigned cudaHostRegisterPortable = 1;
constexpr unsigned cudaHostRegisterMapped = 2;
static int current_device = 0;
static int register_calls = 0;
static int fail_register = -1;
static bool fail_sync = false;
static bool fail_unregister = false;
static bool invalid_mapping = false;
static std::map<uintptr_t, size_t> registrations;
static std::vector<int> drained_devices;
static std::vector<uintptr_t> unregistered;

inline const char* cudaGetErrorString(int) { return "injected error"; }
inline int cudaGetDevice(int* device) {
  *device = current_device;
  return 0;
}
inline int cudaSetDevice(int device) {
  current_device = device;
  return 0;
}
inline int cudaGetLastError() { return 0; }
inline int cudaDeviceSynchronize() {
  drained_devices.push_back(current_device);
  return fail_sync ? 1 : 0;
}
inline int cudaHostRegister(void* pointer, size_t bytes, unsigned flags) {
  assert(flags == (cudaHostRegisterPortable | cudaHostRegisterMapped));
  size_t page = sysconf(_SC_PAGESIZE);
  assert(reinterpret_cast<uintptr_t>(pointer) % page == 0);
  assert(bytes % page == 0);
  if (register_calls++ == fail_register) return 1;
  registrations[reinterpret_cast<uintptr_t>(pointer)] = bytes;
  return 0;
}
inline int cudaHostUnregister(void* pointer) {
  if (fail_unregister) return 1;
  auto address = reinterpret_cast<uintptr_t>(pointer);
  assert(registrations.erase(address) == 1);
  unregistered.push_back(address);
  return 0;
}
inline int cudaHostGetDevicePointer(void** device, void* host, unsigned flags) {
  assert(flags == 0);
  auto address = reinterpret_cast<uintptr_t>(host);
  auto it = registrations.upper_bound(address);
  if (it == registrations.begin()) return 1;
  --it;
  if (address - it->first >= it->second) return 1;
  *device = reinterpret_cast<void*>(
      address + (invalid_mapping && it != registrations.begin() ? 4096 : 0));
  return 0;
}

// Enable only the owner's POSIX guard after libc++ selects Windows threading.
#ifdef _WIN32
  #define __linux__
#endif
#include "registered_host_memory.h"

int main(int argc, char** argv) {
  assert(argc == 2);
  std::string scenario = argv[1];
  size_t page = sysconf(_SC_PAGESIZE);
  // Match the platform page size, including hosts with 64 KiB pages.
  size_t bytes = page * 3 - 7;
  if (scenario == "rollback") fail_register = 2;
  if (scenario == "mapping") invalid_mapping = true;
  if (scenario == "rollback" || scenario == "mapping") {
    bool rejected = false;
    try {
      vllm::RegisteredHostMemory::allocate(bytes, page);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    assert(rejected);
    assert(registrations.empty());
    assert(unregistered.size() == (scenario == "rollback" ? 2 : 3));
    assert(vllm::registered_host_registry().allocations.empty());
    assert(unregistered.front() > unregistered.back());
    return 0;
  }
  auto owner = vllm::RegisteredHostMemory::allocate(bytes, page);
  auto* data = static_cast<char*>(owner->data());
  for (size_t offset : {size_t(0), page - 1, page, bytes - 1}) {
    assert(data[offset] == 0);
    data[offset] = 42;
  }
  if (scenario == "alias") {
    auto alias = vllm::find_registered_host_memory(data + page);
    assert(alias == owner);
    current_device = 1;
    owner->track_current_device();
    owner.reset();
    assert(registrations.size() == 3);
    assert(data[page] == 42);
    alias.reset();
    assert(registrations.empty());
    assert((drained_devices == std::vector<int>{0, 1}));
    assert(current_device == 1);
  } else if (scenario == "retry_mapping") {
    current_device = 1;
    invalid_mapping = true;
    for (int attempt = 0; attempt < 2; ++attempt) {
      bool rejected = false;
      try {
        owner->track_current_device();
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      assert(rejected);
    }
    invalid_mapping = false;
    owner->track_current_device();
    owner.reset();
    assert(registrations.empty());
  } else if (scenario == "sync_failure" || scenario == "unregister_failure") {
    fail_sync = scenario == "sync_failure";
    fail_unregister = scenario == "unregister_failure";
    owner.reset();
    assert(registrations.size() == 3);
    assert(data[page] == 42);  // The unsafe mapping must remain mapped.
    assert(vllm::registered_host_registry().allocations.empty());
    registrations.clear();  // Simulate process-exit driver cleanup.
    assert(munmap(data, page * 3) == 0);
  } else {
    assert(false && "unknown scenario");
  }
  return 0;
}
