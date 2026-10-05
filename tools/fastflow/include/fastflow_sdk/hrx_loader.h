// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <hrx_runtime.h>

namespace fastflow_sdk {

using HrxStatus = hrx_status_t;
using HrxDispatchConfig = hrx_dispatch_config_t;
using HrxBufferRef = hrx_buffer_ref_t;

struct HrxFunctions {
  decltype(&hrx_runtime_version) runtime_version = nullptr;
  decltype(&hrx_status_code) status_code = nullptr;
  decltype(&hrx_status_to_string) status_to_string = nullptr;
  decltype(&hrx_status_free_message) status_free_message = nullptr;
  decltype(&hrx_status_ignore) status_ignore = nullptr;
  decltype(&hrx_gpu_initialize) gpu_initialize = nullptr;
  decltype(&hrx_gpu_shutdown) gpu_shutdown = nullptr;
  decltype(&hrx_gpu_device_count) gpu_device_count = nullptr;
  decltype(&hrx_gpu_device_get) gpu_device_get = nullptr;
  decltype(&hrx_device_release) device_release = nullptr;
  decltype(&hrx_stream_create) stream_create = nullptr;
  decltype(&hrx_stream_release) stream_release = nullptr;
  decltype(&hrx_stream_synchronize) stream_synchronize = nullptr;
  decltype(&hrx_buffer_allocate) buffer_allocate = nullptr;
  decltype(&hrx_buffer_map_with_mode) buffer_map_with_mode = nullptr;
  decltype(&hrx_buffer_unmap) buffer_unmap = nullptr;
  decltype(&hrx_buffer_flush_range) buffer_flush_range = nullptr;
  decltype(&hrx_buffer_invalidate_range) buffer_invalidate_range = nullptr;
  decltype(&hrx_buffer_release) buffer_release = nullptr;
  decltype(&hrx_executable_load_data) executable_load_data = nullptr;
  decltype(&hrx_executable_export_info) executable_export_info = nullptr;
  decltype(&hrx_executable_lookup_export_by_name)
      executable_lookup_export_by_name = nullptr;
  decltype(&hrx_executable_release) executable_release = nullptr;
  decltype(&hrx_stream_dispatch) stream_dispatch = nullptr;
};

struct HrxLoadResult;

class HrxLibrary {
 public:
  HrxLibrary() = default;
  ~HrxLibrary();
  HrxLibrary(HrxLibrary&& other) noexcept;
  HrxLibrary& operator=(HrxLibrary&& other) noexcept;
  HrxLibrary(const HrxLibrary&) = delete;
  HrxLibrary& operator=(const HrxLibrary&) = delete;

  static HrxLoadResult Load(
      const std::optional<std::filesystem::path>& explicit_path);
  static std::vector<std::filesystem::path> Candidates(
      const std::optional<std::filesystem::path>& explicit_path);

  const HrxFunctions& functions() const { return functions_; }
  const std::filesystem::path& path() const { return path_; }
  int runtime_version_major() const { return runtime_version_major_; }
  int runtime_version_minor() const { return runtime_version_minor_; }
  int runtime_version_patch() const { return runtime_version_patch_; }
  bool valid() const { return handle_ != nullptr; }
  std::string ConsumeStatus(HrxStatus status,
                            const std::string& operation) const;

 private:
  friend struct HrxLoadResult;
  void* handle_ = nullptr;
  std::filesystem::path path_;
  HrxFunctions functions_;
  int runtime_version_major_ = 0;
  int runtime_version_minor_ = 0;
  int runtime_version_patch_ = 0;
};

struct HrxLoadResult {
  std::optional<HrxLibrary> library;
  std::vector<std::filesystem::path> candidates;
  std::vector<std::string> missing_symbols;
  std::vector<std::string> errors;
};

}  // namespace fastflow_sdk
