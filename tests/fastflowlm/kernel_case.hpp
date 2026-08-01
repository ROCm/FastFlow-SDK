// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

struct KernelValidationResult {
  size_t mismatches = 0;
  float maximum_absolute_error = 0.0f;
  float maximum_relative_error = 0.0f;
  float maximum_tolerance_ratio = 0.0f;
};

enum class KernelComparison {
  kExactOrdered,
  kExactMultiset,
  kTolerance,
};

struct KernelExecutable {
  hrx_executable_t handle = nullptr;
  uint32_t ordinal = UINT32_MAX;

  KernelExecutable() = default;
  KernelExecutable(const KernelExecutable&) = delete;
  KernelExecutable& operator=(const KernelExecutable&) = delete;
  KernelExecutable(KernelExecutable&& other) noexcept;
  ~KernelExecutable();
};

class KernelCase {
 public:
  KernelCase(
      std::string family, std::string kernel, std::string vector_id,
      std::vector<size_t> dimensions,
      const std::filesystem::path& xclbin_path,
      std::vector<uint8_t> transaction,
      std::vector<uint8_t> argument1,
      std::vector<uint8_t> argument2,
      std::vector<bf16> expected,
      std::string weight_provenance,
      std::string reference_kind,
      KernelComparison comparison = KernelComparison::kExactOrdered,
      float absolute_tolerance = 0.0f,
      float relative_tolerance = 0.0f);
  ~KernelCase();

  std::string name() const;
  const std::string& family() const;
  const std::string& kernel() const;
  const std::string& vector_id() const;
  const std::string& weight_provenance() const;
  const std::string& reference_kind() const;
  const std::string& implementation_id() const;
  const char* comparison_name() const;
  const std::vector<size_t>& dimensions() const;
  size_t validated_bytes() const;
  const KernelValidationResult& validation() const;
  float absolute_tolerance() const;
  float relative_tolerance() const;
  double device_seconds() const;

  KernelExecutable create_executable() const;
  std::vector<bf16> dispatch(const KernelExecutable& executable);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
