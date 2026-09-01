// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "typedef.hpp"

namespace kernel_test_data {

inline constexpr const char* kTemporaryDirectoryPrefix = "rocm-npu-q4nx-";

enum class Q4Profile {
  kSharedSubset,
  kFullAffine,
  kFullNibble,
};

struct Q4Weight {
  size_t rows;
  size_t columns;
  std::vector<uint8_t> packed;
  std::vector<bf16> logical;
  std::vector<bf16> signed_nibble_logical;
  std::vector<bf16> qwen_affine_logical;
  std::vector<bf16> lfm_affine_logical;
};

Q4Weight make_q4_weight(size_t rows, size_t columns, int salt,
                        Q4Profile profile = Q4Profile::kSharedSubset);

// Runs FastFlowLM's own converter over caller-supplied quantization inputs and
// returns the packed Q4NX bytes. Exposed so a generated packing can be proven
// byte-exact against the shipping contract, including with coordinate-coded
// probes that make the packing order directly observable.
std::vector<uint8_t> pack_q4nx(size_t rows, size_t columns,
                               const std::vector<uint32_t>& quantized,
                               const std::vector<float>& scales,
                               const std::vector<int32_t>& zero_points);

std::vector<bf16> make_activation(size_t rows, size_t columns, int salt);

std::vector<bf16> matmul_reference(const std::vector<bf16>& weight,
                                   const std::vector<bf16>& activation,
                                   size_t m, size_t k, size_t n);

// Converts row-major logical weights to the physical BF16 layout consumed by
// the production 1024x1024 MM transaction.
size_t mm_layout_logical_index(size_t physical_index);

std::vector<bf16> to_mm_layout(const std::vector<bf16>& logical, size_t rows,
                               size_t columns);

// Converts a logical KxN activation to the physical layout consumed by the
// production 1024x1024x128 MM transaction.
size_t gemm_activation_logical_index(size_t physical_index);

std::vector<bf16> to_gemm_activation_layout(const std::vector<bf16>& logical,
                                            size_t k, size_t n);

}  // namespace kernel_test_data
