// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm_npu/hrx_loader.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <system_error>
#include <utility>

#include "rocm_npu/platform.h"
#include "platform/hrx_library.h"

namespace rocm_npu {
namespace {

template <typename Function>
Function Lookup(void* handle, const char* name) {
  static_assert(sizeof(Function) == sizeof(void*),
                "function and object pointers must have matching sizes");
  void* raw = LookupHrxSymbol(handle, name);
  Function function = nullptr;
  std::memcpy(&function, &raw, sizeof(function));
  return function;
}

void AddCandidate(std::vector<std::filesystem::path>& candidates,
                  const std::filesystem::path& candidate) {
  if (candidate.empty()) {
    return;
  }
  for (const auto& existing : candidates) {
    if (existing == candidate) {
      return;
    }
  }
  candidates.push_back(candidate);
}

template <typename Function>
void Resolve(void* handle, const char* symbol, Function& destination,
             std::vector<std::string>& missing) {
  destination = Lookup<Function>(handle, symbol);
  if (destination == nullptr) {
    missing.emplace_back(symbol);
  }
}

}  // namespace

HrxLibrary::~HrxLibrary() {
  if (handle_ != nullptr) {
    CloseHrxLibrary(handle_);
  }
}

HrxLibrary::HrxLibrary(HrxLibrary&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      path_(std::move(other.path_)),
      functions_(other.functions_),
      runtime_version_major_(other.runtime_version_major_),
      runtime_version_minor_(other.runtime_version_minor_),
      runtime_version_patch_(other.runtime_version_patch_) {}

HrxLibrary& HrxLibrary::operator=(HrxLibrary&& other) noexcept {
  if (this != &other) {
    if (handle_ != nullptr) {
      CloseHrxLibrary(handle_);
    }
    handle_ = std::exchange(other.handle_, nullptr);
    path_ = std::move(other.path_);
    functions_ = other.functions_;
    runtime_version_major_ = other.runtime_version_major_;
    runtime_version_minor_ = other.runtime_version_minor_;
    runtime_version_patch_ = other.runtime_version_patch_;
  }
  return *this;
}

std::vector<std::filesystem::path> HrxLibrary::Candidates(
    const std::optional<std::filesystem::path>& explicit_path) {
  std::vector<std::filesystem::path> candidates;
  const auto add_path = [&](const std::filesystem::path& path) {
    std::error_code error;
    if (std::filesystem::is_directory(path, error)) {
      AddCandidate(candidates, path / HrxLibraryFilename());
    } else {
      AddCandidate(candidates, path);
    }
  };

  if (explicit_path) {
    add_path(*explicit_path);
    return candidates;
  }
  if (auto environment = HrxEnvironmentPath()) {
    add_path(*environment);
  }

  const std::filesystem::path executable = ExecutablePath();
  if (!executable.empty()) {
    const std::filesystem::path directory = executable.parent_path();
    for (const auto& candidate : PackagedHrxLibraryCandidates(directory)) {
      AddCandidate(candidates, candidate);
    }
  }
  for (const auto& candidate : HrxLibraryFallbackCandidates()) {
    AddCandidate(candidates, candidate);
  }
  return candidates;
}

HrxLoadResult HrxLibrary::Load(
    const std::optional<std::filesystem::path>& explicit_path) {
  HrxLoadResult result;
  result.candidates = Candidates(explicit_path);
  for (const auto& candidate : result.candidates) {
    std::string error;
    void* handle = OpenHrxLibrary(candidate, error);
    if (handle == nullptr) {
      result.errors.push_back(PathToUtf8(candidate) + ": " + error);
      continue;
    }

    HrxLibrary library;
    library.handle_ = handle;
    library.path_ = candidate;
    HrxFunctions& functions = library.functions_;
    std::vector<std::string> missing;
#define ROCM_NPU_RESOLVE(field, symbol) \
  Resolve(handle, symbol, functions.field, missing)
    ROCM_NPU_RESOLVE(runtime_version, "hrx_runtime_version");
    ROCM_NPU_RESOLVE(status_code, "hrx_status_code");
    ROCM_NPU_RESOLVE(status_to_string, "hrx_status_to_string");
    ROCM_NPU_RESOLVE(status_free_message, "hrx_status_free_message");
    ROCM_NPU_RESOLVE(status_ignore, "hrx_status_ignore");
    ROCM_NPU_RESOLVE(gpu_initialize, "hrx_gpu_initialize");
    ROCM_NPU_RESOLVE(gpu_shutdown, "hrx_gpu_shutdown");
    ROCM_NPU_RESOLVE(gpu_device_count, "hrx_gpu_device_count");
    ROCM_NPU_RESOLVE(gpu_device_get, "hrx_gpu_device_get");
    ROCM_NPU_RESOLVE(device_release, "hrx_device_release");
    ROCM_NPU_RESOLVE(stream_create, "hrx_stream_create");
    ROCM_NPU_RESOLVE(stream_release, "hrx_stream_release");
    ROCM_NPU_RESOLVE(stream_synchronize, "hrx_stream_synchronize");
    ROCM_NPU_RESOLVE(buffer_allocate, "hrx_buffer_allocate");
    ROCM_NPU_RESOLVE(buffer_map_with_mode, "hrx_buffer_map_with_mode");
    ROCM_NPU_RESOLVE(buffer_unmap, "hrx_buffer_unmap");
    ROCM_NPU_RESOLVE(buffer_flush_range, "hrx_buffer_flush_range");
    ROCM_NPU_RESOLVE(buffer_invalidate_range,
                     "hrx_buffer_invalidate_range");
    ROCM_NPU_RESOLVE(buffer_release, "hrx_buffer_release");
    ROCM_NPU_RESOLVE(executable_load_data, "hrx_executable_load_data");
    ROCM_NPU_RESOLVE(executable_export_info, "hrx_executable_export_info");
    ROCM_NPU_RESOLVE(executable_lookup_export_by_name,
                     "hrx_executable_lookup_export_by_name");
    ROCM_NPU_RESOLVE(executable_release, "hrx_executable_release");
    ROCM_NPU_RESOLVE(stream_dispatch, "hrx_stream_dispatch");
#undef ROCM_NPU_RESOLVE
    if (missing.empty()) {
      functions.runtime_version(&library.runtime_version_major_,
                                &library.runtime_version_minor_,
                                &library.runtime_version_patch_);
      if (library.runtime_version_major_ != HRX_VERSION_MAJOR ||
          library.runtime_version_minor_ != HRX_VERSION_MINOR) {
        std::ostringstream detail;
        detail << PathToUtf8(candidate) << ": incompatible HRX ABI "
               << library.runtime_version_major_ << '.'
               << library.runtime_version_minor_ << '.'
               << library.runtime_version_patch_ << " (requires "
               << HRX_VERSION_MAJOR << '.' << HRX_VERSION_MINOR << ".x)";
        result.errors.push_back(detail.str());
        continue;
      }
      result.missing_symbols.clear();
      result.library.emplace(std::move(library));
      return result;
    }
    std::ostringstream detail;
    detail << PathToUtf8(candidate) << ": missing required symbols:";
    for (const auto& symbol : missing) {
      detail << ' ' << symbol;
      if (std::find(result.missing_symbols.begin(),
                    result.missing_symbols.end(),
                    symbol) == result.missing_symbols.end()) {
        result.missing_symbols.push_back(symbol);
      }
    }
    result.errors.push_back(detail.str());
  }
  return result;
}

std::string HrxLibrary::ConsumeStatus(HrxStatus status,
                                      const std::string& operation) const {
  if (status == nullptr) {
    return {};
  }
  char* message = nullptr;
  std::size_t length = 0;
  HrxStatus formatting =
      functions_.status_to_string(status, &message, &length);
  const int code = functions_.status_code(status);
  std::string text =
      message == nullptr ? "unknown error" : std::string(message, length);
  if (message != nullptr) {
    functions_.status_free_message(message);
  }
  if (formatting != nullptr) {
    functions_.status_ignore(formatting);
  }
  functions_.status_ignore(status);
  std::ostringstream result;
  result << operation << " failed (status " << code << "): " << text;
  return result.str();
}

}  // namespace rocm_npu
