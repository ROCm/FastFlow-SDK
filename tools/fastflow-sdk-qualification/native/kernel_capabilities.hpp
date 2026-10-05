// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

enum class NibbleSemantics { kUnsigned, kSignedTwosComplement };
enum class ZeroPointSemantics { kAddAfterScale, kIgnored };

struct GemmPolicy {
  float full_nibble_absolute_tolerance;
  float full_affine_absolute_tolerance;
};

struct FamilyCapability {
  std::string name;
  std::filesystem::path xclbin_directory;
  std::filesystem::path dequant_xclbin;
  std::filesystem::path mm_xclbin;
  std::string dequant_implementation_group;
  std::string mm_implementation_group;
  std::string shared_subset_group;
  NibbleSemantics nibble;
  ZeroPointSemantics zero_point;
  GemmPolicy gemm_policy;
};

struct KernelCapabilities {
  std::string contract_id;
  size_t m;
  size_t k;
  size_t n;
  float relative_tolerance;
  float structured_absolute_tolerance;
  std::vector<FamilyCapability> families;
};

KernelCapabilities load_kernel_capabilities(const std::filesystem::path& path);
