// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "kernel_capabilities.hpp"
#include "kernel_test_data.hpp"
#include "kernel_validation.hpp"

namespace {

size_t temporary_directory_count() {
  size_t count = 0;
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(
           std::filesystem::temp_directory_path(), error)) {
    if (entry.path().filename().string().starts_with(
            kernel_test_data::kTemporaryDirectoryPrefix)) {
      ++count;
    }
  }
  if (error) throw std::runtime_error("cannot inspect temporary directory");
  return count;
}

template <typename Callable>
void require_failure(Callable callable, const char* message) {
  try {
    callable();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error(
          "kernel-native-tests requires a capability manifest");
    }
    const auto capabilities = load_kernel_capabilities(argv[1]);
    if (capabilities.families.size() < 2 ||
        capabilities.contract_id != "fastflowlm-hrx-kernel-test-v1") {
      throw std::runtime_error(
          "kernel capability manifest lacks implementation coverage");
    }
    const size_t temporary_before = temporary_directory_count();
    const auto generated =
        kernel_test_data::make_q4_weight(1024, 1024, 3);
    const auto repeated =
        kernel_test_data::make_q4_weight(1024, 1024, 3);
    const auto full_affine = kernel_test_data::make_q4_weight(
        1024, 1024, 5, kernel_test_data::Q4Profile::kFullAffine);
    if (generated.packed != repeated.packed ||
        generated.logical.size() != repeated.logical.size() ||
        std::memcmp(
            generated.logical.data(), repeated.logical.data(),
            generated.logical.size() * sizeof(bf16)) != 0) {
      throw std::runtime_error("generated Q4NX vector is not deterministic");
    }
    if (temporary_directory_count() != temporary_before) {
      throw std::runtime_error("Q4NX generator leaked temporary files");
    }
    if (std::memcmp(
            generated.logical.data(),
            generated.logical.data() + generated.columns,
            generated.columns * sizeof(bf16)) == 0) {
      throw std::runtime_error(
          "generated Q4NX rows do not exercise output layout");
    }
    float minimum = static_cast<float>(generated.logical[0]);
    float maximum = minimum;
    size_t nonzero = 0;
    for (const bf16& element : generated.logical) {
      const float value = static_cast<float>(element);
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
      nonzero += value != 0.0f;
    }
    bool full_has_negative = false;
    bool full_exceeds_subset = false;
    for (const bf16& element : full_affine.logical) {
      const float value = static_cast<float>(element);
      full_has_negative |= value < 0.0f;
      full_exceeds_subset |= value > maximum;
    }
    if (!full_has_negative || !full_exceeds_subset) {
      throw std::runtime_error(
          "full affine Q4 profile lacks upper-nibble/zero-point coverage");
    }
    if (std::memcmp(
            full_affine.logical.data(),
            full_affine.signed_nibble_logical.data(),
            full_affine.logical.size() * sizeof(bf16)) == 0) {
      throw std::runtime_error(
          "family-specific signed-nibble reference is not distinct");
    }
    std::vector<bool> layout_visited(1024 * 1024);
    for (size_t physical = 0; physical < layout_visited.size(); ++physical) {
      const size_t logical =
          kernel_test_data::mm_layout_logical_index(physical);
      if (layout_visited[logical]) {
        throw std::runtime_error("MM layout transform is not a bijection");
      }
      layout_visited[logical] = true;
    }
    if (kernel_test_data::mm_layout_logical_index(0) != 0 ||
        kernel_test_data::mm_layout_logical_index(1) != 1024 ||
        kernel_test_data::mm_layout_logical_index(512) != 1 ||
        kernel_test_data::mm_layout_logical_index(16384) != 256) {
      throw std::runtime_error("MM layout coordinate contract changed");
    }
    std::vector<bool> activation_visited(1024 * 128);
    for (size_t physical = 0;
         physical < activation_visited.size(); ++physical) {
      const size_t logical =
          kernel_test_data::gemm_activation_logical_index(physical);
      if (activation_visited[logical]) {
        throw std::runtime_error(
            "GEMM activation transform is not a bijection");
      }
      activation_visited[logical] = true;
    }
    if (kernel_test_data::gemm_activation_logical_index(0) != 0 ||
        kernel_test_data::gemm_activation_logical_index(8) != 1024 ||
        kernel_test_data::gemm_activation_logical_index(512) != 128 ||
        kernel_test_data::gemm_activation_logical_index(4096) != 8) {
      throw std::runtime_error(
          "GEMM activation coordinate contract changed");
    }
    if (std::memcmp(
            full_affine.qwen_affine_logical.data(),
            full_affine.logical.data(),
            full_affine.logical.size() * sizeof(bf16)) == 0 ||
        std::memcmp(
            full_affine.lfm_affine_logical.data(),
            full_affine.signed_nibble_logical.data(),
            full_affine.logical.size() * sizeof(bf16)) == 0) {
      throw std::runtime_error(
          "production family zero-point references are not distinct");
    }
    const auto activation =
        kernel_test_data::make_activation(64, 32, 11);
    bool has_negative = false;
    bool has_positive = false;
    bool has_zero = false;
    for (const bf16& element : activation) {
      const float value = static_cast<float>(element);
      has_negative |= value < 0.0f;
      has_positive |= value > 0.0f;
      has_zero |= value == 0.0f;
    }
    if (!has_negative || !has_positive || !has_zero) {
      throw std::runtime_error(
          "activation generator lacks signed/nonuniform/zero coverage");
    }
    for (size_t left_column = 0; left_column < 32; ++left_column) {
      for (size_t right_column = left_column + 1;
           right_column < 32; ++right_column) {
        bool differs = false;
        for (size_t row = 0; row < 64; ++row) {
          differs |= std::memcmp(
              &activation[row * 32 + left_column],
              &activation[row * 32 + right_column],
              sizeof(bf16)) != 0;
        }
        if (!differs) {
          throw std::runtime_error(
              "activation generator repeats a logical column");
        }
      }
    }
    const std::vector<bf16> left = {
        bf16(1.0f), bf16(2.0f), bf16(-1.0f), bf16(0.5f)};
    const std::vector<bf16> right = {
        bf16(3.0f), bf16(-2.0f), bf16(4.0f), bf16(1.0f)};
    const auto product =
        kernel_test_data::matmul_reference(left, right, 2, 2, 2);
    if (static_cast<float>(product[0]) != 11.0f ||
        static_cast<float>(product[3]) != 2.5f) {
      throw std::runtime_error("CPU GEMM reference is incorrect");
    }
    if (static_cast<float>(bf16(1.003f)) != 1.0f ||
        static_cast<float>(bf16(1.006f)) != 1.0078125f) {
      throw std::runtime_error("BF16 rounding contract changed");
    }

    constexpr size_t kGuard = 4;
    std::vector<uint8_t> expected = {1, 2, 3, 4};
    std::vector<uint8_t> guarded(expected.size() + 2 * kGuard, 0x5A);
    std::memcpy(guarded.data() + kGuard, expected.data(), expected.size());
    kernel_validation::validate_immutable(
        guarded, expected, kGuard, 0x5A, "native-test");
    guarded[kGuard + 1] ^= 1;
    require_failure(
        [&] {
          kernel_validation::validate_immutable(
              guarded, expected, kGuard, 0x5A, "native-test");
        },
        "input mutation was not detected");
    guarded[kGuard + 1] ^= 1;
    guarded.front() ^= 1;
    require_failure(
        [&] {
          kernel_validation::validate_immutable(
              guarded, expected, kGuard, 0x5A, "native-test");
        },
        "canary corruption was not detected");
    std::cout << "packed=" << generated.packed.size()
              << " reference=" << generated.logical.size()
              << " min=" << minimum << " max=" << maximum
              << " nonzero=" << nonzero << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "kernel-native-tests: FAIL: " << error.what() << "\n";
    return 1;
  }
}
