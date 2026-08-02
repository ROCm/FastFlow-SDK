// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Checks that a host mapping stays coherent when the NPU writes an output BO
// the host is only ever reading.
//
// A buffer read back after a host-to-device publish is easy to get right: the
// publish leaves no valid cache lines over the range, so the copy out cannot
// source stale ones. hrx_kernel_smoke re-seeds every output BO from the host
// before each dispatch and therefore only ever exercises that easy case.
//
// The hard case is a BO seeded once and then alternately written by the NPU
// and read by the host, which is what an inference loop does with a reused
// output buffer. Each host read repopulates clean lines over the range that
// the next dispatch then writes behind the CPU's back, so the runtime has to
// invalidate them before copying out. When it does not, the readback returns
// the previous dispatch's result.
//
// This test drives that second pattern. It captures a reference result for two
// activation sets through the safe host-seeded path, then alternates between
// them without touching the output BO from the host again, and requires every
// readback to match the activation it was dispatched with.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernel_capabilities.hpp"
#include "kernel_case.hpp"
#include "kernel_test_data.hpp"
#include "modules/gemm.hpp"

namespace {

constexpr size_t kGuardBytes = 4096;

struct Options {
  std::filesystem::path xclbin_root;
  std::filesystem::path capabilities;
  int iterations = 64;
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
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
  }
  if (options.xclbin_root.empty() || options.capabilities.empty()) {
    throw std::runtime_error("--xclbin-root and --capabilities are required");
  }
  if (options.iterations < 2) {
    throw std::runtime_error("--iterations must be at least 2");
  }
  return options;
}

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

template <typename T>
std::vector<uint8_t> to_bytes(const std::vector<T>& values) {
  const auto* begin = reinterpret_cast<const uint8_t*>(values.data());
  return {begin, begin + values.size() * sizeof(T)};
}

std::vector<uint8_t> gemm_transaction(size_t m, size_t k, size_t n) {
  LM_Config config = {};
  Gemm gemm(config);
  npu_sequence sequence(device_npu2);
  gemm.generate_seq(
      &sequence, static_cast<uint32_t>(m), static_cast<uint32_t>(k),
      static_cast<uint32_t>(n), 0, false, Gemm::NO_Activation, 0);
  const auto [words, word_count] = sequence.dump();
  const auto* bytes = reinterpret_cast<const uint8_t*>(words);
  return {bytes, bytes + word_count * sizeof(uint32_t)};
}

// Holds the BOs for one gemm shape and runs it repeatedly. `output` is only
// ever seeded from the host on request so the caller controls whether a
// readback follows a host publish or a bare device write.
class ReadbackHarness {
 public:
  ReadbackHarness(
      const std::filesystem::path& xclbin_path, size_t m, size_t k, size_t n,
      const std::vector<bf16>& weight)
      : output_bytes_(m * n * sizeof(bf16)),
        activation_bytes_(k * n * sizeof(bf16)),
        weight_bytes_(m * k * sizeof(bf16) + kGuardBytes),
        xclbin_(read_file(xclbin_path)),
        transaction_(gemm_transaction(m, k, n)),
        output_(kernel_test_device(), output_bytes_ + 2 * kGuardBytes),
        weight_(kernel_test_device(), weight_bytes_ + 2 * kGuardBytes),
        activation_(kernel_test_device(), activation_bytes_ + 2 * kGuardBytes),
        executable_(create_kernel_executable(
            xclbin_, transaction_, "hrx-readback-smoke")) {
    if (xclbin_.empty() || transaction_.empty()) {
      throw std::runtime_error("gemm kernel case data is empty");
    }
    const auto weight_values = to_bytes(weight);
    if (weight_values.size() + kGuardBytes != weight_bytes_) {
      throw std::runtime_error("weight size does not match the gemm shape");
    }
    weight_.memset(0);
    std::memcpy(
        weight_.data() + kGuardBytes, weight_values.data(),
        weight_values.size());
    weight_.sync_to_device();
  }

  void set_activation(const std::vector<bf16>& activation) {
    const auto bytes = to_bytes(activation);
    if (bytes.size() != activation_bytes_) {
      throw std::runtime_error("activation size does not match the gemm shape");
    }
    activation_.memset(0);
    std::memcpy(activation_.data() + kGuardBytes, bytes.data(), bytes.size());
    activation_.sync_to_device();
  }

  void seed_output(uint8_t value) {
    output_.memset(value);
    output_.sync_to_device();
  }

  std::vector<bf16> dispatch() {
    hrx_buffer_ref_t bindings[] = {
        {output_.bo().handle(), kGuardBytes, output_bytes_},
        {weight_.bo().handle(), kGuardBytes, weight_bytes_},
        {activation_.bo().handle(), kGuardBytes, activation_bytes_},
    };
    hrx_dispatch_config_t config = {{1, 1, 1}, {1, 1, 1}, 0};
    if (hrx::hrx_report(
            hrx_stream_dispatch(
                hrx::rt().stream, executable_.handle, executable_.ordinal,
                &config, nullptr, 0, bindings, 3, HRX_DISPATCH_FLAG_NONE),
            "hrx_stream_dispatch") ||
        hrx::hrx_report(
            hrx_stream_synchronize(hrx::rt().stream),
            "hrx_stream_synchronize")) {
      throw std::runtime_error("gemm dispatch failed");
    }
    output_.sync_from_device();
    const auto* values =
        reinterpret_cast<const bf16*>(output_.data() + kGuardBytes);
    return {values, values + output_bytes_ / sizeof(bf16)};
  }

 private:
  size_t output_bytes_;
  size_t activation_bytes_;
  size_t weight_bytes_;
  std::vector<uint8_t> xclbin_;
  std::vector<uint8_t> transaction_;
  buffer<uint8_t> output_;
  buffer<uint8_t> weight_;
  buffer<uint8_t> activation_;
  KernelExecutable executable_;
};

size_t count_differences(
    const std::vector<bf16>& left, const std::vector<bf16>& right) {
  size_t differences = 0;
  for (size_t i = 0; i < left.size(); ++i) {
    if (std::memcmp(&left[i], &right[i], sizeof(bf16)) != 0) ++differences;
  }
  return differences;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto capabilities = load_kernel_capabilities(options.capabilities);
    if (capabilities.families.empty()) {
      throw std::runtime_error("kernel capability manifest has no families");
    }
    const size_t m = capabilities.m;
    const size_t k = capabilities.k;
    const size_t n = capabilities.n;
    const auto& family = capabilities.families.front();

    constexpr int kSalt = 41;
    ReadbackHarness harness(
        options.xclbin_root / family.xclbin_directory / family.mm_xclbin, m, k,
        n, kernel_test_data::make_q4_weight(m, k, kSalt).logical);

    const std::vector<std::vector<bf16>> activations = {
        kernel_test_data::to_gemm_activation_layout(
            kernel_test_data::make_activation(k, n, kSalt), k, n),
        kernel_test_data::to_gemm_activation_layout(
            kernel_test_data::make_activation(k, n, kSalt + 13), k, n),
    };

    // Reference capture: seeding the output from the host before each dispatch
    // drops every cache line over the range, so these readbacks are trusted
    // even on a runtime that never invalidates.
    std::vector<std::vector<bf16>> expected;
    for (size_t i = 0; i < activations.size(); ++i) {
      harness.set_activation(activations[i]);
      harness.seed_output(static_cast<uint8_t>(0xA5 + i));
      expected.push_back(harness.dispatch());
    }
    const size_t reference_differences =
        count_differences(expected[0], expected[1]);
    if (reference_differences * 8 < expected[0].size()) {
      throw std::runtime_error(
          "the two activation sets produce nearly identical results (" +
          std::to_string(reference_differences) + " of " +
          std::to_string(expected[0].size()) +
          " elements differ), so a stale readback would go unnoticed");
    }

    // Coherence loop: the output BO is never written by the host again, so
    // each dispatch overwrites the clean lines the previous readback left
    // behind.
    const auto started = std::chrono::steady_clock::now();
    for (int round = 0; round < options.iterations; ++round) {
      const size_t current = static_cast<size_t>(round) % activations.size();
      const size_t previous =
          (current + activations.size() - 1) % activations.size();
      harness.set_activation(activations[current]);
      const auto actual = harness.dispatch();
      const size_t differences = count_differences(actual, expected[current]);
      if (!differences) continue;
      // Elements that carry the previous dispatch's value came from cache
      // lines the runtime left valid over a range the device had rewritten.
      size_t stale = 0;
      for (size_t i = 0; i < actual.size(); ++i) {
        if (std::memcmp(&actual[i], &expected[current][i], sizeof(bf16)) != 0 &&
            std::memcmp(&actual[i], &expected[previous][i], sizeof(bf16)) ==
                0) {
          ++stale;
        }
      }
      throw std::runtime_error(
          "round " + std::to_string(round) + " read " +
          std::to_string(differences) + " of " +
          std::to_string(actual.size()) +
          " elements that do not match activation set " +
          std::to_string(current) + ", of which " + std::to_string(stale) +
          " still hold the previous dispatch's result");
    }
    const double elapsed =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started)
            .count();
    std::printf(
        "PASS: %s mm %zux%zux%zu, %d coherent readbacks in %.3fs\n",
        family.name.c_str(), m, k, n, options.iterations, elapsed);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "hrx-readback-smoke: FAIL: %s\n", error.what());
    return 1;
  }
}
