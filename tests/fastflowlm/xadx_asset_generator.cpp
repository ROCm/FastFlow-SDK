// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Generates compiler-free qualification inputs from the same production
// FastFlowLM transaction generators and numeric oracle as the native smoke.

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "hrx_amdxdna.h"
#include "kernel_capabilities.hpp"
#include "kernel_test_data.hpp"
#include "modules/dequant.hpp"
#include "modules/gemm.hpp"

namespace {

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

template <typename T>
void write_values(
    const std::filesystem::path& path, const std::vector<T>& values) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(
      reinterpret_cast<const char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!stream) throw std::runtime_error("cannot write " + path.string());
}

void write_bytes(
    const std::filesystem::path& path, const std::vector<uint8_t>& values) {
  write_values(path, values);
}

std::vector<uint8_t> transaction(npu_sequence& sequence) {
  const auto [words, count] = sequence.dump();
  const auto* bytes = reinterpret_cast<const uint8_t*>(words);
  return {bytes, bytes + count * sizeof(uint32_t)};
}

std::vector<uint8_t> serialize(
    const std::vector<uint8_t>& xclbin,
    const std::vector<uint8_t>& commands) {
  const hrx_const_byte_span_t xclbin_span = {xclbin.data(), xclbin.size()};
  hrx_amdxdna_executable_run_t run = hrx_amdxdna_executable_run_default();
  run.transaction = {commands.data(), commands.size()};
  hrx_amdxdna_executable_entry_point_t entry =
      hrx_amdxdna_executable_entry_point_default();
  entry.name = {"MLIR_AIE", 8};
  entry.runs = &run;
  entry.run_count = 1;
  hrx_amdxdna_executable_create_params_t params =
      hrx_amdxdna_executable_create_params_default();
  params.xclbins = &xclbin_span;
  params.xclbin_count = 1;
  params.entry_points = &entry;
  params.entry_point_count = 1;
  const hrx_host_allocator_t allocator = hrx_host_allocator_system();
  uint8_t* output = nullptr;
  size_t output_size = 0;
  hrx_status_t status = hrx_amdxdna_xadx_serialize(
      &params, allocator, &output, &output_size);
  if (status) {
    hrx_status_ignore(status);
    throw std::runtime_error(
        "hrx_amdxdna_xadx_serialize failed; installed HRX does not provide "
        "a usable public XADX serializer");
  }
  std::vector<uint8_t> result(output, output + output_size);
  hrx_host_allocator_free(allocator, output);
  if (result.empty()) throw std::runtime_error("HRX serialized an empty XADX");
  return result;
}

std::string safe(std::string value) {
  for (char& character : value) {
    if (!std::isalnum(static_cast<unsigned char>(character)) &&
        character != '-' && character != '_') {
      character = '_';
    }
  }
  return value;
}

void json_string(std::ostream& stream, const std::string& value) {
  stream << '"';
  for (char character : value) {
    if (character == '"' || character == '\\') stream << '\\';
    stream << character;
  }
  stream << '"';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      throw std::runtime_error(
          "usage: fastflowlm-xadx-asset-generator "
          "<xclbin-root> <capabilities.json> <output-directory>");
    }
    const std::filesystem::path xclbin_root = argv[1];
    const auto capabilities = load_kernel_capabilities(argv[2]);
    const std::filesystem::path output = argv[3];
    std::filesystem::remove_all(output);
    std::filesystem::create_directories(output / "executables");
    std::filesystem::create_directories(output / "data");
    std::filesystem::create_directories(output / ".transactions");

    LM_Config config = {};
    Dequant dequant(config);
    Gemm gemm(config);
    npu_sequence dequant_sequence(device_npu2);
    dequant.generate_dequant_q4_1_seq(
        &dequant_sequence, static_cast<uint32_t>(capabilities.k),
        static_cast<uint32_t>(capabilities.m), 0, 0);
    npu_sequence mm_sequence(device_npu2);
    gemm.generate_seq(
        &mm_sequence, static_cast<uint32_t>(capabilities.m),
        static_cast<uint32_t>(capabilities.k),
        static_cast<uint32_t>(capabilities.n), 0, false,
        Gemm::NO_Activation, 0);
    const auto dequant_transaction = transaction(dequant_sequence);
    const auto mm_transaction = transaction(mm_sequence);
    if (dequant_transaction.empty() || mm_transaction.empty()) {
      throw std::runtime_error("FastFlowLM transaction generator returned no commands");
    }
    write_bytes(output / ".transactions/dequant.bin", dequant_transaction);
    write_bytes(output / ".transactions/mm.bin", mm_transaction);

    constexpr int kSalt = 41;
    const auto weight = kernel_test_data::make_q4_weight(
        capabilities.m, capabilities.k, kSalt);
    auto weight_bytes = std::vector<uint8_t>(
        reinterpret_cast<const uint8_t*>(weight.logical.data()),
        reinterpret_cast<const uint8_t*>(weight.logical.data()) +
            weight.logical.size() * sizeof(bf16));
    weight_bytes.resize(weight_bytes.size() + 4096, 0);
    const auto activation_a = kernel_test_data::make_activation(
        capabilities.k, capabilities.n, kSalt);
    const auto activation_b = kernel_test_data::make_activation(
        capabilities.k, capabilities.n, kSalt + 13);
    const auto physical_a = kernel_test_data::to_gemm_activation_layout(
        activation_a, capabilities.k, capabilities.n);
    const auto physical_b = kernel_test_data::to_gemm_activation_layout(
        activation_b, capabilities.k, capabilities.n);
    const auto expected_a = kernel_test_data::matmul_reference(
        weight.logical, activation_a, capabilities.m, capabilities.k,
        capabilities.n);
    const auto expected_b = kernel_test_data::matmul_reference(
        weight.logical, activation_b, capabilities.m, capabilities.k,
        capabilities.n);
    write_bytes(output / "data/mm-weight.bin", weight_bytes);
    write_values(output / "data/mm-activation-a.bin", physical_a);
    write_values(output / "data/mm-activation-b.bin", physical_b);
    write_values(output / "data/mm-expected-a.bin", expected_a);
    write_values(output / "data/mm-expected-b.bin", expected_b);
    write_bytes(output / "data/dequant-weight-zero.bin",
                std::vector<uint8_t>(weight.packed.size(), 0));
    write_bytes(output / "data/dequant-parameter-zero.bin",
                std::vector<uint8_t>(4096, 0));
    write_values(output / "data/dequant-expected-zero.bin",
                 std::vector<bf16>(capabilities.m * capabilities.k, bf16(0.0f)));

    std::ofstream draft(output / ".generation.json");
    draft << "{\"serializer_api\":\"hrx_amdxdna_xadx_serialize\","
             "\"serializer_revision\":\"hrx-amdxdna-v0\","
             "\"transactions\":{\"dequant\":\".transactions/dequant.bin\","
             "\"mm\":\".transactions/mm.bin\"},\"executables\":[";
    bool first = true;
    for (const auto& family : capabilities.families) {
      for (const std::string kernel : {"dequant", "mm"}) {
        if (!first) draft << ',';
        first = false;
        const std::string id = safe(family.name) + "-" + kernel;
        const std::filesystem::path source_relative_path =
            family.xclbin_directory /
            (kernel == "dequant" ? family.dequant_xclbin : family.mm_xclbin);
        const std::string source_relative =
            source_relative_path.generic_string();
        const auto xclbin = read_file(xclbin_root / source_relative_path);
        const auto& commands =
            kernel == "dequant" ? dequant_transaction : mm_transaction;
        write_bytes(output / "executables" / (id + ".xadx"),
                    serialize(xclbin, commands));
        draft << "{\"id\":";
        json_string(draft, id);
        draft << ",\"family\":";
        json_string(draft, family.name);
        draft << ",\"kernel\":";
        json_string(draft, kernel);
        draft << ",\"source_xclbin\":";
        json_string(draft, source_relative);
        draft << ",\"xadx\":";
        json_string(draft, "executables/" + id + ".xadx");
        draft << '}';
      }
    }
    draft << "],\"shape\":{\"m\":" << capabilities.m
          << ",\"k\":" << capabilities.k
          << ",\"n\":" << capabilities.n
          << "},\"tolerances\":{\"atol\":"
          << capabilities.structured_absolute_tolerance
          << ",\"rtol\":" << capabilities.relative_tolerance << "}}";
    if (!draft) throw std::runtime_error("cannot write generation metadata");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "fastflowlm-xadx-asset-generator: FAIL: %s\n", error.what());
    return 1;
  }
}
