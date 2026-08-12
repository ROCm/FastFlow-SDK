// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Emits one XADX per unique shipped kernel implementation so qualification
// coverage can be indexed by implementation rather than by model family.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hrx_amdxdna.h"
#include "models/gemma_text/gemma_text_dequant.hpp"
#include "models/gemma_text/gemma_text_gemm.hpp"
#include "modules/dequant.hpp"
#include "modules/gemm.hpp"

namespace {

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
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
  const hrx_status_t status = hrx_amdxdna_xadx_serialize(
      &params, allocator, &output, &output_size);
  if (status) {
    hrx_status_ignore(status);
    throw std::runtime_error("hrx_amdxdna_xadx_serialize failed");
  }
  std::vector<uint8_t> result(output, output + output_size);
  hrx_host_allocator_free(allocator, output);
  if (result.empty()) throw std::runtime_error("HRX serialized an empty XADX");
  return result;
}

struct Implementation {
  std::string kernel;
  std::string representative;
  std::vector<std::string> families;
  size_t xclbin_bytes = 0;
};

struct Shape {
  uint32_t m;
  uint32_t k;
  uint32_t n;

  std::string label() const {
    return std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
  }
};

// The shipped overlays are not interchangeable, so an implementation is only
// reachable through the builder that targets it, and every mm builder except
// the generic one needs N >= 512. Builders abort the process on a shape they
// reject rather than throwing, so one invocation emits exactly one builder at
// one shape and the caller sweeps the grid across processes.
std::string build_transaction(
    const std::string& builder, const Shape& shape,
    std::vector<uint8_t>& commands) {
  LM_Config config = {};
  npu_sequence sequence(device_npu2);
  std::string kernel;
  if (builder == "q4_1") {
    kernel = "dequant";
    Dequant(config).generate_dequant_q4_1_seq(
        &sequence, shape.k, shape.m, 0, 0);
  } else if (builder == "q80nx") {
    kernel = "dequant";
    Dequant(config).generate_dequant_q80_packed_in_q4nx_seq(
        &sequence, shape.k, shape.m, 0, 0);
  } else if (builder == "gemma-dequant") {
    kernel = "dequant";
    GemmaTextDequant(config).generate_seq(&sequence, shape.k, shape.m, 0);
  } else if (builder == "gemm") {
    kernel = "mm";
    Gemm(config).generate_seq(
        &sequence, shape.m, shape.k, shape.n, 0, false, Gemm::NO_Activation, 0);
  } else if (builder == "gemma-gemm") {
    kernel = "mm";
    GemmaTextGemm(config).generate_seq(
        &sequence, shape.m, shape.k, shape.n, 0);
  } else {
    throw std::runtime_error("unknown builder: " + builder);
  }
  commands = transaction(sequence);
  if (commands.empty()) throw std::runtime_error("builder emitted no commands");
  return kernel;
}

// "m,k,n" -> shape.
Shape parse_shape(const std::string& item) {
  const size_t first = item.find(',');
  const size_t second = item.find(',', first + 1);
  if (first == std::string::npos || second == std::string::npos) {
    throw std::runtime_error("shape must be m,k,n: " + item);
  }
  return {
      static_cast<uint32_t>(std::stoul(item.substr(0, first))),
      static_cast<uint32_t>(std::stoul(item.substr(first + 1, second - first - 1))),
      static_cast<uint32_t>(std::stoul(item.substr(second + 1)))};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      throw std::runtime_error(
          "usage: fastflowlm-xadx-implementation-probe <xclbin-root> <output> "
          "<builder> <m,k,n>");
    }
    const std::filesystem::path root = argv[1];
    const std::filesystem::path output = argv[2];
    const std::string builder = argv[3];
    const Shape shape = parse_shape(argv[4]);
    const std::string variant = builder + "-" + shape.label();
    std::filesystem::create_directories(output);

    std::vector<uint8_t> commands;
    const std::string target_kernel = build_transaction(builder, shape, commands);
    const std::string transaction_relative = "transaction-" + variant + ".bin";
    std::ofstream transaction_stream(
        output / transaction_relative, std::ios::binary);
    transaction_stream.write(
        reinterpret_cast<const char*>(commands.data()),
        static_cast<std::streamsize>(commands.size()));
    if (!transaction_stream) {
      throw std::runtime_error("cannot write " + transaction_relative);
    }

    // Identical xclbin bytes mean an identical kernel implementation, so the
    // qualification payload only needs one XADX per distinct blob.
    std::map<std::pair<std::string, std::vector<uint8_t>>, Implementation> unique;
    std::vector<std::filesystem::path> directories;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
      if (entry.is_directory()) directories.push_back(entry.path());
    }
    std::sort(directories.begin(), directories.end());
    for (const auto& directory : directories) {
      for (const std::string kernel : {"dequant", "mm"}) {
        const auto path = directory / (kernel + ".xclbin");
        if (!std::filesystem::is_regular_file(path)) continue;
        auto contents = read_file(path);
        const auto key = std::make_pair(kernel, contents);
        auto found = unique.find(key);
        if (found == unique.end()) {
          Implementation record;
          record.kernel = kernel;
          record.representative = directory.filename().string();
          record.xclbin_bytes = contents.size();
          record.families = {directory.filename().string()};
          unique.emplace(key, std::move(record));
        } else {
          found->second.families.push_back(directory.filename().string());
        }
      }
    }

    std::ofstream index(output / ("index-" + variant + ".json"));
    index << "[";
    bool first = true;
    int ordinal = 0;
    size_t emitted = 0;
    for (const auto& [key, record] : unique) {
      const auto& [kernel, contents] = key;
      const std::string implementation =
          kernel + "-" + std::to_string(ordinal++);
      if (kernel != target_kernel) continue;
      const std::string id = implementation + "-" + variant;
      const auto xadx = serialize(contents, commands);
      const std::string relative = id + ".xadx";
      std::ofstream stream(output / relative, std::ios::binary);
      stream.write(
          reinterpret_cast<const char*>(xadx.data()),
          static_cast<std::streamsize>(xadx.size()));
      if (!stream) throw std::runtime_error("cannot write " + relative);
      if (!first) index << ',';
      first = false;
      ++emitted;
      index << "{\"id\":\"" << id << "\",\"implementation\":\""
            << implementation << "\",\"kernel\":\"" << kernel
            << "\",\"representative\":\"" << record.representative
            << "\",\"builder\":\"" << builder
            << "\",\"transaction\":\"" << transaction_relative
            << "\",\"xclbin_bytes\":" << record.xclbin_bytes
            << ",\"m\":" << shape.m << ",\"k\":" << shape.k
            << ",\"n\":" << shape.n
            << ",\"xadx\":\"" << relative
            << "\",\"xadx_bytes\":" << xadx.size()
            << ",\"family_count\":" << record.families.size()
            << ",\"families\":[";
      for (size_t i = 0; i < record.families.size(); ++i) {
        if (i) index << ',';
        index << '"' << record.families[i] << '"';
      }
      index << "]}";
    }
    index << "]";
    if (!index) throw std::runtime_error("cannot write implementation index");
    std::printf(
        "%s: %zu of %zu %s implementations (from %zu families)\n",
        variant.c_str(), emitted, unique.size(), target_kernel.c_str(),
        directories.size());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(
        stderr, "fastflowlm-xadx-implementation-probe: FAIL: %s\n",
        error.what());
    return 1;
  }
}
