// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kernel_case.hpp"
#include "kernel_capabilities.hpp"
#include "kernel_test_data.hpp"
#include "modules/dequant.hpp"
#include "modules/gemm.hpp"

namespace {

std::vector<uint8_t> copy_transaction(npu_sequence& sequence) {
  const auto [words, word_count] = sequence.dump();
  const auto* bytes = reinterpret_cast<const uint8_t*>(words);
  return {bytes, bytes + word_count * sizeof(uint32_t)};
}

template <typename T>
std::vector<uint8_t> to_bytes(const std::vector<T>& values) {
  const auto* begin = reinterpret_cast<const uint8_t*>(values.data());
  return {begin, begin + values.size() * sizeof(T)};
}

bool files_equal(
    const std::filesystem::path& left,
    const std::filesystem::path& right) {
  std::error_code error;
  const auto left_size = std::filesystem::file_size(left, error);
  if (error) throw std::runtime_error("cannot inspect " + left.string());
  const auto right_size = std::filesystem::file_size(right, error);
  if (error) throw std::runtime_error("cannot inspect " + right.string());
  if (left_size != right_size) return false;
  std::ifstream left_stream(left, std::ios::binary);
  std::ifstream right_stream(right, std::ios::binary);
  if (!left_stream || !right_stream) {
    throw std::runtime_error("cannot compare xclbin implementations");
  }
  constexpr size_t kChunk = 64 * 1024;
  std::vector<char> left_bytes(kChunk);
  std::vector<char> right_bytes(kChunk);
  while (left_stream) {
    left_stream.read(left_bytes.data(), left_bytes.size());
    right_stream.read(right_bytes.data(), right_bytes.size());
    const auto count = left_stream.gcount();
    if (count != right_stream.gcount() ||
        !std::equal(
            left_bytes.begin(), left_bytes.begin() + count,
            right_bytes.begin())) {
      return false;
    }
  }
  return true;
}

const char* nibble_name(NibbleSemantics value) {
  return value == NibbleSemantics::kUnsigned
             ? "unsigned"
             : "signed-twos-complement";
}

const char* zero_point_name(ZeroPointSemantics value) {
  return value == ZeroPointSemantics::kAddAfterScale
             ? "add-after-scale"
             : "ignored";
}

template <typename Key>
std::vector<size_t> representative_indices(
    const std::vector<FamilyCapability>& families, Key key) {
  std::set<std::string> seen;
  std::vector<size_t> result;
  for (size_t i = 0; i < families.size(); ++i) {
    if (seen.insert(key(families[i])).second) result.push_back(i);
  }
  return result;
}

void verify_implementation_groups(
    const std::filesystem::path& root,
    const std::vector<FamilyCapability>& families, bool dequant) {
  for (size_t left = 0; left < families.size(); ++left) {
    for (size_t right = left + 1; right < families.size(); ++right) {
      const auto& lhs = families[left];
      const auto& rhs = families[right];
      const std::string& left_group =
          dequant ? lhs.dequant_implementation_group
                  : lhs.mm_implementation_group;
      const std::string& right_group =
          dequant ? rhs.dequant_implementation_group
                  : rhs.mm_implementation_group;
      const auto& left_file =
          dequant ? lhs.dequant_xclbin : lhs.mm_xclbin;
      const auto& right_file =
          dequant ? rhs.dequant_xclbin : rhs.mm_xclbin;
      const bool equal = files_equal(
          root / lhs.xclbin_directory / left_file,
          root / rhs.xclbin_directory / right_file);
      if (equal != (left_group == right_group)) {
        throw std::runtime_error(
            std::string(dequant ? "dequant" : "MM") +
            " xclbin bytes disagree with capability implementation groups: " +
            lhs.name + " vs " + rhs.name);
      }
    }
  }
}

struct Options {
  std::filesystem::path xclbin_root;
  std::filesystem::path capabilities;
  int iterations = 20;
  bool json = false;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--xclbin-root" && i + 1 < argc) {
      options.xclbin_root = argv[++i];
    } else if (arg == "--capabilities" && i + 1 < argc) {
      options.capabilities = argv[++i];
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = std::stoi(argv[++i]);
    } else if (arg == "--json") {
      options.json = true;
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
  }
  if (options.xclbin_root.empty() || options.capabilities.empty()) {
    throw std::runtime_error(
        "--xclbin-root and --capabilities are required");
  }
  if (options.iterations < 1) {
    throw std::runtime_error("--iterations must be positive");
  }
  return options;
}

struct Scenario {
  std::string id;
  size_t m;
  size_t k;
  size_t n;
  kernel_test_data::Q4Weight weight;
  std::vector<bf16> activation;
  std::vector<bf16> mm_expected;
  std::vector<uint8_t> dequant_transaction;
  std::vector<uint8_t> gemm_transaction;
};

void print_json_report(
    const Options& options,
    const KernelCapabilities& capabilities,
    const std::vector<std::unique_ptr<KernelCase>>& cases,
    int lifecycle_rounds, size_t preflight_dispatches,
    double elapsed, double process_elapsed) {
  double device_region_seconds = 0.0;
  for (const auto& test_case : cases) {
    device_region_seconds += test_case->device_seconds();
  }
  std::printf(
      "{\"status\":\"pass\",\"iterations\":%d,\"failures\":0,"
      "\"capability_contract_id\":\"%s\","
      "\"implementation_id_algorithm\":\"fnv1a64\","
      "\"lifecycle_rounds\":%d,\"resident_rounds\":%d,"
      "\"preflight_dispatches\":%zu,\"dispatches\":%zu,"
      "\"elapsed_seconds\":%.6f,\"device_region_seconds\":%.6f,"
      "\"process_wall_seconds\":%.6f,\"cases\":[",
      options.iterations, capabilities.contract_id.c_str(),
      lifecycle_rounds, options.iterations,
      preflight_dispatches,
      preflight_dispatches +
          cases.size() * static_cast<size_t>(
                             options.iterations + lifecycle_rounds),
      elapsed, device_region_seconds, process_elapsed);
  for (size_t i = 0; i < cases.size(); ++i) {
    if (i) std::printf(",");
    std::printf(
        "{\"name\":\"%s\",\"family\":\"%s\","
        "\"kernel\":\"%s\",\"vector_id\":\"%s\","
        "\"implementation_id\":\"%s\","
        "\"weight_provenance\":\"%s\",\"reference_kind\":\"%s\","
        "\"comparison\":\"%s\",\"dimensions\":[",
        cases[i]->name().c_str(), cases[i]->family().c_str(),
        cases[i]->kernel().c_str(), cases[i]->vector_id().c_str(),
        cases[i]->implementation_id().c_str(),
        cases[i]->weight_provenance().c_str(),
        cases[i]->reference_kind().c_str(),
        cases[i]->comparison_name());
    for (size_t dimension = 0;
         dimension < cases[i]->dimensions().size(); ++dimension) {
      if (dimension) std::printf(",");
      std::printf("%zu", cases[i]->dimensions()[dimension]);
    }
    std::printf(
        "],\"lifecycle_rounds\":%d,\"resident_rounds\":%d,"
        "\"validated_bytes\":%zu,\"mismatches\":%zu,"
        "\"absolute_tolerance\":%.9g,\"relative_tolerance\":%.9g,"
        "\"maximum_absolute_error\":%.9g,"
        "\"maximum_relative_error\":%.9g,"
        "\"maximum_tolerance_ratio\":%.9g,"
        "\"device_seconds\":%.9g,\"failures\":0}",
        lifecycle_rounds, options.iterations, cases[i]->validated_bytes(),
        cases[i]->validation().mismatches,
        cases[i]->absolute_tolerance(),
        cases[i]->relative_tolerance(),
        cases[i]->validation().maximum_absolute_error,
        cases[i]->validation().maximum_relative_error,
        cases[i]->validation().maximum_tolerance_ratio,
        cases[i]->device_seconds());
  }
  std::printf("]}\n");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto process_started = std::chrono::steady_clock::now();
    const Options options = parse_options(argc, argv);
    const KernelCapabilities capabilities =
        load_kernel_capabilities(options.capabilities);
    LM_Config config = {};
    Dequant dequant(config);
    Gemm gemm(config);
    const auto dequant_transaction = [&](size_t m, size_t k) {
      npu_sequence sequence(device_npu2);
      dequant.generate_dequant_q4_1_seq(
          &sequence, static_cast<uint32_t>(k),
          static_cast<uint32_t>(m), 0, 0);
      return copy_transaction(sequence);
    };
    const auto gemm_transaction = [&](size_t m, size_t k, size_t n) {
      npu_sequence sequence(device_npu2);
      gemm.generate_seq(
          &sequence, static_cast<uint32_t>(m),
          static_cast<uint32_t>(k), static_cast<uint32_t>(n),
          0, false, Gemm::NO_Activation, 0);
      return copy_transaction(sequence);
    };

    const size_t kM = capabilities.m;
    const size_t kK = capabilities.k;
    const size_t kN = capabilities.n;
    constexpr int kSalt = 29;
    auto weight = kernel_test_data::make_q4_weight(kM, kK, kSalt);
    auto full_nibble_weight = kernel_test_data::make_q4_weight(
        kM, kK, kSalt + 7,
        kernel_test_data::Q4Profile::kFullNibble);
    auto full_affine_weight = kernel_test_data::make_q4_weight(
        kM, kK, kSalt + 11,
        kernel_test_data::Q4Profile::kFullAffine);
    auto logical_activation =
        kernel_test_data::make_activation(kK, kN, kSalt);
    auto mm_expected = kernel_test_data::matmul_reference(
        weight.logical, logical_activation, kM, kK, kN);
    auto alternate_logical_activation =
        kernel_test_data::make_activation(kK, kN, kSalt + 17);
    auto alternate_mm_expected = kernel_test_data::matmul_reference(
        weight.logical, alternate_logical_activation, kM, kK, kN);
    auto alternate_activation =
        kernel_test_data::to_gemm_activation_layout(
            alternate_logical_activation, kK, kN);
    auto dequant_commands = dequant_transaction(kM, kK);
    auto gemm_commands = gemm_transaction(kM, kK, kN);
    if (dequant_commands.empty() || gemm_commands.empty()) {
      throw std::runtime_error("transaction generator returned no commands");
    }
    Scenario scenario{
        "structured-baseline",
        kM,
        kK,
        kN,
        std::move(weight),
        kernel_test_data::to_gemm_activation_layout(
            logical_activation, kK, kN),
        std::move(mm_expected),
        std::move(dequant_commands),
        std::move(gemm_commands),
    };

    const auto& families = capabilities.families;
    verify_implementation_groups(
        options.xclbin_root, families, /*dequant=*/true);
    verify_implementation_groups(
        options.xclbin_root, families, /*dequant=*/false);
    const auto dequant_representatives = representative_indices(
        families, [](const auto& family) {
          return family.dequant_implementation_group;
        });
    const auto mm_representatives = representative_indices(
        families, [](const auto& family) {
          return family.mm_implementation_group;
        });
    const auto full_nibble_dequant_representatives =
        representative_indices(families, [](const auto& family) {
          return family.dequant_implementation_group + "|" +
                 nibble_name(family.nibble);
        });
    const auto full_affine_dequant_representatives =
        representative_indices(families, [](const auto& family) {
          return family.dequant_implementation_group + "|" +
                 nibble_name(family.nibble) + "|" +
                 zero_point_name(family.zero_point);
        });
    const auto structured_mm_representatives =
        representative_indices(families, [](const auto& family) {
          return family.mm_implementation_group + "|" +
                 family.shared_subset_group;
        });
    const auto full_nibble_mm_representatives =
        representative_indices(families, [](const auto& family) {
          return family.mm_implementation_group + "|" +
                 nibble_name(family.nibble);
        });
    const auto full_affine_mm_representatives =
        representative_indices(families, [](const auto& family) {
          return family.mm_implementation_group + "|" +
                 nibble_name(family.nibble) + "|" +
                 zero_point_name(family.zero_point);
        });

    std::vector<std::unique_ptr<KernelCase>> cases;
    for (size_t index : dequant_representatives) {
      const auto& family = families[index];
      cases.push_back(std::make_unique<KernelCase>(
          family.name, "dequant", "zero-baseline",
          std::vector<size_t>{kM, kK},
          options.xclbin_root / family.xclbin_directory /
              family.dequant_xclbin,
          scenario.dequant_transaction,
          std::vector<uint8_t>(scenario.weight.packed.size(), 0),
          std::vector<uint8_t>(4096, 0),
          std::vector<bf16>(kM * kK, bf16(0.0f)),
          "synthetic-zero-q4nx", "exact-zero",
          KernelComparison::kExactOrdered));
    }
    for (size_t index : mm_representatives) {
      const auto& family = families[index];
      cases.push_back(std::make_unique<KernelCase>(
          family.name, "mm", "zero-baseline",
          std::vector<size_t>{kM, kK, kN},
          options.xclbin_root / family.xclbin_directory /
              family.mm_xclbin,
          scenario.gemm_transaction,
          std::vector<uint8_t>(kM * kK * sizeof(bf16) + 4096, 0),
          std::vector<uint8_t>(kK * kN * sizeof(bf16), 0),
          std::vector<bf16>(kM * kN, bf16(0.0f)),
          "synthetic-zero-bf16", "exact-zero",
          KernelComparison::kExactOrdered));
    }

    const auto started = std::chrono::steady_clock::now();
    size_t preflight_dispatches = 0;
    const auto cpu_mm_weight = kernel_test_data::to_mm_layout(
        scenario.weight.logical, kM, kK);
    const auto integrated_mm_expected =
        kernel_test_data::matmul_reference(
            cpu_mm_weight, logical_activation, kM, kK, kN);
    const auto alternate_integrated_mm_expected =
        kernel_test_data::matmul_reference(
            cpu_mm_weight, alternate_logical_activation, kM, kK, kN);
    std::map<std::string, std::vector<bf16>> shared_npu_weights;
    for (size_t index : dequant_representatives) {
      const auto& family = families[index];
      auto test_case = std::make_unique<KernelCase>(
          family.name, "dequant", scenario.id,
          std::vector<size_t>{kM, kK},
          options.xclbin_root / family.xclbin_directory /
              family.dequant_xclbin,
          scenario.dequant_transaction, scenario.weight.packed,
          std::vector<uint8_t>(4096, 0), cpu_mm_weight,
          "generated-q4nx", "cpu-mm-layout",
          KernelComparison::kExactOrdered);
      KernelExecutable executable = test_case->create_executable();
      shared_npu_weights[family.dequant_implementation_group] =
          test_case->dispatch(executable);
      ++preflight_dispatches;
      cases.push_back(std::move(test_case));
    }
    for (size_t left = 0; left < families.size(); ++left) {
      for (size_t right = left + 1; right < families.size(); ++right) {
        if (families[left].shared_subset_group !=
            families[right].shared_subset_group) {
          continue;
        }
        const auto& lhs = shared_npu_weights.at(
            families[left].dequant_implementation_group);
        const auto& rhs = shared_npu_weights.at(
            families[right].dequant_implementation_group);
        if (lhs.size() != rhs.size() ||
            std::memcmp(
                lhs.data(), rhs.data(), lhs.size() * sizeof(bf16)) != 0) {
          throw std::runtime_error(
              "shared-subset output disagrees with capability group: " +
              families[left].name + " vs " + families[right].name);
        }
      }
    }
    std::map<std::string, std::vector<bf16>> full_nibble_npu_weights;
    for (size_t index : full_nibble_dequant_representatives) {
      const auto& family = families[index];
      const auto& family_full_nibble =
          family.nibble ==
                  NibbleSemantics::kUnsigned
              ? full_nibble_weight.logical
              : full_nibble_weight.signed_nibble_logical;
      auto test_case = std::make_unique<KernelCase>(
          family.name, "dequant", "full-nibble",
          std::vector<size_t>{kM, kK},
          options.xclbin_root / family.xclbin_directory /
              family.dequant_xclbin,
          scenario.dequant_transaction, full_nibble_weight.packed,
          std::vector<uint8_t>(4096, 0),
          kernel_test_data::to_mm_layout(
              family_full_nibble, kM, kK),
          "generated-full-nibble-q4nx",
          family.nibble ==
                  NibbleSemantics::kUnsigned
              ? "cpu-unsigned-nibble-mm-layout"
              : "cpu-signed-nibble-mm-layout",
          KernelComparison::kExactOrdered);
      KernelExecutable executable = test_case->create_executable();
      full_nibble_npu_weights[
          family.dequant_implementation_group + "|" +
          nibble_name(family.nibble)] = test_case->dispatch(executable);
      ++preflight_dispatches;
      cases.push_back(std::move(test_case));
    }
    std::map<std::string, std::vector<bf16>> full_affine_npu_weights;
    for (size_t index : full_affine_dequant_representatives) {
      const auto& family = families[index];
      const auto& family_full_affine =
          family.zero_point ==
                  ZeroPointSemantics::kAddAfterScale
              ? full_affine_weight.qwen_affine_logical
              : full_affine_weight.lfm_affine_logical;
      auto affine_case = std::make_unique<KernelCase>(
          family.name, "dequant", "full-affine",
          std::vector<size_t>{kM, kK},
          options.xclbin_root / family.xclbin_directory /
              family.dequant_xclbin,
          scenario.dequant_transaction, full_affine_weight.packed,
          std::vector<uint8_t>(4096, 0),
          kernel_test_data::to_mm_layout(
              family_full_affine, kM, kK),
          "generated-full-affine-q4nx",
          family.zero_point ==
                  ZeroPointSemantics::kAddAfterScale
              ? "cpu-qwen-q-times-scale-plus-zp-mm-layout"
              : "cpu-lfm-signed-nibble-zp-ignored-mm-layout",
          KernelComparison::kExactOrdered);
      KernelExecutable affine_executable =
          affine_case->create_executable();
      full_affine_npu_weights[
          family.dequant_implementation_group + "|" +
          nibble_name(family.nibble) + "|" +
          zero_point_name(family.zero_point)] =
          affine_case->dispatch(affine_executable);
      ++preflight_dispatches;
      cases.push_back(std::move(affine_case));
    }

    for (size_t index : mm_representatives) {
      const auto& family = families[index];
      auto weight_bytes = to_bytes(scenario.weight.logical);
      weight_bytes.resize(weight_bytes.size() + 4096, 0);
      cases.push_back(std::make_unique<KernelCase>(
          family.name, "mm", "isolated-structured",
          std::vector<size_t>{kM, kK, kN},
          options.xclbin_root / family.xclbin_directory /
              family.mm_xclbin,
          scenario.gemm_transaction, std::move(weight_bytes),
          to_bytes(scenario.activation), scenario.mm_expected,
          "cpu-generated-mm-layout", "cpu-logical-matmul",
          KernelComparison::kTolerance,
          capabilities.structured_absolute_tolerance,
          capabilities.relative_tolerance));
    }
    for (size_t index : structured_mm_representatives) {
      const auto& family = families[index];
      const auto& npu_weight = shared_npu_weights.at(
          family.dequant_implementation_group);
      auto weight_bytes = to_bytes(npu_weight);
      weight_bytes.resize(weight_bytes.size() + 4096, 0);
      cases.push_back(std::make_unique<KernelCase>(
          family.name, "mm", scenario.id,
          std::vector<size_t>{kM, kK, kN},
          options.xclbin_root / family.xclbin_directory /
              family.mm_xclbin,
          scenario.gemm_transaction, std::move(weight_bytes),
          to_bytes(scenario.activation), integrated_mm_expected,
          "npu-dequant-preflight", "cpu-logical-matmul",
          KernelComparison::kTolerance,
          capabilities.structured_absolute_tolerance,
          capabilities.relative_tolerance));
      weight_bytes = to_bytes(npu_weight);
      weight_bytes.resize(weight_bytes.size() + 4096, 0);
      cases.push_back(std::make_unique<KernelCase>(
          family.name, "mm", "structured-alternate",
          std::vector<size_t>{kM, kK, kN},
          options.xclbin_root / family.xclbin_directory /
              family.mm_xclbin,
          scenario.gemm_transaction, std::move(weight_bytes),
          to_bytes(alternate_activation), alternate_integrated_mm_expected,
          "npu-dequant-preflight", "cpu-logical-matmul",
          KernelComparison::kTolerance,
          capabilities.structured_absolute_tolerance,
          capabilities.relative_tolerance));
    }
    for (size_t index : full_nibble_mm_representatives) {
      const auto& family = families[index];
      const auto& npu_weight = full_nibble_npu_weights.at(
          family.dequant_implementation_group + "|" +
          nibble_name(family.nibble));
      auto weight_bytes = to_bytes(npu_weight);
      weight_bytes.resize(weight_bytes.size() + 4096, 0);
      cases.push_back(std::make_unique<KernelCase>(
          family.name, "mm", "full-nibble",
          std::vector<size_t>{kM, kK, kN},
          options.xclbin_root / family.xclbin_directory /
              family.mm_xclbin,
          scenario.gemm_transaction, std::move(weight_bytes),
          to_bytes(scenario.activation),
          kernel_test_data::matmul_reference(
              npu_weight, logical_activation, kM, kK, kN),
          "npu-dequant-preflight", "cpu-family-nibble-kernel-chain",
          KernelComparison::kTolerance,
          family.gemm_policy.full_nibble_absolute_tolerance,
          capabilities.relative_tolerance));
    }
    for (size_t index : full_affine_mm_representatives) {
      const auto& family = families[index];
      const auto& npu_weight = full_affine_npu_weights.at(
          family.dequant_implementation_group + "|" +
          nibble_name(family.nibble) + "|" +
          zero_point_name(family.zero_point));
      auto weight_bytes = to_bytes(npu_weight);
      weight_bytes.resize(weight_bytes.size() + 4096, 0);
      cases.push_back(std::make_unique<KernelCase>(
          family.name, "mm", "full-affine",
          std::vector<size_t>{kM, kK, kN},
          options.xclbin_root / family.xclbin_directory /
              family.mm_xclbin,
          scenario.gemm_transaction, std::move(weight_bytes),
          to_bytes(scenario.activation),
          kernel_test_data::matmul_reference(
              npu_weight, logical_activation, kM, kK, kN),
          "npu-dequant-preflight", "cpu-family-affine-kernel-chain",
          KernelComparison::kTolerance,
          family.gemm_policy.full_affine_absolute_tolerance,
          capabilities.relative_tolerance));
    }

    std::set<std::string> scheduled_names;
    for (const auto& test_case : cases) {
      if (!scheduled_names.insert(test_case->name()).second) {
        throw std::runtime_error(
            "kernel capability schedule produced a duplicate case: " +
            test_case->name());
      }
    }

    constexpr int kLifecycleRounds = 5;
    for (int round = 0; round < kLifecycleRounds; ++round) {
      for (auto& test_case : cases) {
        KernelExecutable executable = test_case->create_executable();
        test_case->dispatch(executable);
      }
    }
    std::vector<KernelExecutable> resident;
    resident.reserve(cases.size());
    for (const auto& test_case : cases) {
      resident.push_back(test_case->create_executable());
    }
    for (int round = 0; round < options.iterations; ++round) {
      for (size_t i = 0; i < cases.size(); ++i) {
        cases[i]->dispatch(resident[i]);
      }
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const double process_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - process_started).count();
    if (options.json) {
      print_json_report(
          options, capabilities, cases, kLifecycleRounds,
          preflight_dispatches,
          elapsed, process_elapsed);
    } else {
      std::printf(
          "PASS: %zu cases, %d lifecycle + %d resident rounds in %.3fs\n",
          cases.size(), kLifecycleRounds, options.iterations, elapsed);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "hrx-kernel-smoke: FAIL: %s\n", error.what());
    return 1;
  }
}
