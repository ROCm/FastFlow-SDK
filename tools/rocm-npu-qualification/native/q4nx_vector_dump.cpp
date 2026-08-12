// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Emits converter-backed default Q4NX vectors and C++ interpretations of the
// explicitly measured non-default layouts. The latter prove that the C++ and
// Python interpreters agree on recorded facts; they are not a second production
// converter.

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernel_test_data.hpp"

namespace {

template <typename T>
void write_values(
    const std::filesystem::path& path, const std::vector<T>& values) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(
      reinterpret_cast<const char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!stream) throw std::runtime_error("cannot write " + path.string());
}

struct ProfileSpec {
  const char* id;
  kernel_test_data::Q4Profile profile;
  int salt;
};

std::vector<uint8_t> pack_parameterized(
    size_t rows, size_t columns, size_t row_block, size_t col_block,
    size_t parallel, const std::vector<uint32_t>& quantized,
    const std::vector<float>& scales,
    const std::vector<int32_t>& zero_points) {
  if (rows % row_block || columns % col_block ||
      row_block % parallel || col_block % 32 || parallel % 2) {
    throw std::runtime_error("invalid parameterized Q4NX geometry");
  }
  const size_t chunks = (rows / row_block) * (columns / col_block);
  const size_t fields_per_chunk = row_block * (col_block / 32);
  const size_t nibbles_per_chunk = row_block * col_block / 2;
  std::vector<uint8_t> result(
      chunks * (2 * sizeof(uint16_t) * fields_per_chunk + nibbles_per_chunk));
  size_t cursor = 0;
  for (size_t row_chunk = 0; row_chunk < rows; row_chunk += row_block) {
    for (size_t col_chunk = 0; col_chunk < columns; col_chunk += col_block) {
      for (bool zero_points_field : {false, true}) {
        for (size_t group = 0; group < col_block / 32; ++group) {
          for (size_t row = 0; row < row_block; ++row) {
            const size_t source =
                (row_chunk + row) * (columns / 32) +
                (col_chunk / 32 + group);
            const bf16 value(
                zero_points_field
                    ? static_cast<float>(zero_points[source])
                    : scales[source]);
            std::memcpy(result.data() + cursor, &value, sizeof(value));
            cursor += sizeof(value);
          }
        }
      }
      for (size_t row_group = 0; row_group < row_block; row_group += parallel) {
        for (size_t column = 0; column < col_block; ++column) {
          for (size_t pair = 0; pair < parallel; pair += 2) {
            const size_t low =
                (row_chunk + row_group + pair) * columns + col_chunk + column;
            const size_t high = low + columns;
            result[cursor++] = static_cast<uint8_t>(
                (quantized[low] & 0xF) | ((quantized[high] & 0xF) << 4));
          }
        }
      }
    }
  }
  if (cursor != result.size()) {
    throw std::runtime_error("parameterized Q4NX pack size mismatch");
  }
  return result;
}

template <size_t N>
std::vector<bf16> bit_permute(
    const std::vector<bf16>& logical, const int (&bits)[N]) {
  if (logical.size() != (size_t{1} << N)) {
    throw std::runtime_error("bit permutation size mismatch");
  }
  std::vector<bf16> physical(logical.size());
  for (size_t index = 0; index < physical.size(); ++index) {
    size_t logical_index = 0;
    for (size_t bit = 0; bit < N; ++bit) {
      logical_index |= ((index >> bit) & 1) << bits[bit];
    }
    physical[index] = logical[logical_index];
  }
  return physical;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error(
          "usage: fastflowlm-q4nx-vector-dump <output-directory>");
    }
    const std::filesystem::path output = argv[1];
    std::filesystem::create_directories(output);

    constexpr size_t kM = 1024;
    constexpr size_t kK = 1024;
    constexpr size_t kN = 128;
    const ProfileSpec profiles[] = {
        {"shared-subset", kernel_test_data::Q4Profile::kSharedSubset, 29},
        {"full-nibble", kernel_test_data::Q4Profile::kFullNibble, 36},
        {"full-affine", kernel_test_data::Q4Profile::kFullAffine, 40},
    };

    std::ofstream index(output / "index.json");
    index << "{\"shape\":{\"m\":" << kM << ",\"k\":" << kK << ",\"n\":" << kN
          << "},\"profiles\":[";
    bool first = true;
    for (const ProfileSpec& spec : profiles) {
      const auto weight =
          kernel_test_data::make_q4_weight(kM, kK, spec.salt, spec.profile);
      const std::string prefix = std::string(spec.id) + "-";
      write_values(output / (prefix + "packed.bin"), weight.packed);
      write_values(output / (prefix + "logical.bin"), weight.logical);
      write_values(
          output / (prefix + "signed-nibble.bin"), weight.signed_nibble_logical);
      write_values(
          output / (prefix + "qwen-affine.bin"), weight.qwen_affine_logical);
      write_values(
          output / (prefix + "lfm-affine.bin"), weight.lfm_affine_logical);
      write_values(
          output / (prefix + "mm-layout.bin"),
          kernel_test_data::to_mm_layout(weight.logical, kM, kK));
      constexpr int kGptOssOrder[] = {
          0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 7, 9, 8, 15, 16, 17, 18, 19};
      constexpr int kGemmaTextOrder[] = {
          0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 7, 8, 9, 15, 16, 17, 18, 19};
      constexpr int kMmOrder[] = {
          10, 11, 12, 3, 4, 5, 6, 7, 9, 0, 1, 2, 13, 14, 8, 16, 15, 17, 18, 19};
      write_values(
          output / (prefix + "gpt-oss-layout.bin"),
          bit_permute(weight.logical, kGptOssOrder));
      write_values(
          output / (prefix + "gemma-text-layout.bin"),
          bit_permute(weight.logical, kGemmaTextOrder));
      const auto activation_n128 =
          kernel_test_data::make_activation(kK, kN, 29);
      const auto activation_wide =
          kernel_test_data::make_activation(kK, 512, 29);
      write_values(
          output / (prefix + "matmul-29-128.bin"),
          kernel_test_data::matmul_reference(
              weight.logical, activation_n128, kM, kK, kN));
      write_values(
          output / (prefix + "matmul-29-512.bin"),
          kernel_test_data::matmul_reference(
              weight.logical, activation_wide, kM, kK, 512));
      {
        const auto lfm_mm = bit_permute(
            weight.lfm_affine_logical, kMmOrder);
        const auto qwen_mm = bit_permute(
            weight.qwen_affine_logical, kMmOrder);
        const auto qwen_gemma = bit_permute(
            weight.qwen_affine_logical, kGemmaTextOrder);
        write_values(
            output / (prefix + "chain-lfm-affine-mm-512.bin"),
            kernel_test_data::matmul_reference(
                lfm_mm, activation_wide, kM, kK, 512));
        write_values(
            output / (prefix + "chain-qwen-affine-mm-512.bin"),
            kernel_test_data::matmul_reference(
                qwen_mm, activation_wide, kM, kK, 512));
        write_values(
            output / (prefix + "chain-qwen-affine-gemma-text-512.bin"),
            kernel_test_data::matmul_reference(
                qwen_gemma, activation_wide, kM, kK, 512));
      }
      if (!first) index << ',';
      first = false;
      index << "{\"id\":\"" << spec.id << "\",\"salt\":" << spec.salt
            << ",\"packed_bytes\":" << weight.packed.size() << '}';
    }
    index << "],\"activations\":[";
    first = true;
    for (const int salt : {29, 46, 41, 54}) {
      const auto logical = kernel_test_data::make_activation(kK, kN, salt);
      const std::string prefix = "activation-" + std::to_string(salt) + "-";
      write_values(output / (prefix + "logical.bin"), logical);
      write_values(
          output / (prefix + "physical.bin"),
          kernel_test_data::to_gemm_activation_layout(logical, kK, kN));
      const auto logical_wide = kernel_test_data::make_activation(kK, 512, salt);
      constexpr int kGemmWide[] = {
          0, 1, 2, 12, 13, 14, 15, 16, 17, 9, 10, 11, 3, 4, 5, 6, 18, 7, 8};
      constexpr int kGemmaWide[] = {
          9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 16, 17, 18, 5, 6, 7, 8};
      write_values(output / (prefix + "gemm-512.bin"),
                   bit_permute(logical_wide, kGemmWide));
      write_values(output / (prefix + "gemma-gemm-512.bin"),
                   bit_permute(logical_wide, kGemmaWide));
      if (!first) index << ',';
      first = false;
      index << salt;
    }
    index << "]}";
    if (!index) throw std::runtime_error("cannot write vector index");

    // Coordinate-coded probes make the packing permutation directly readable.
    // BF16 cannot represent every integer block index at this scale, so paired
    // files carry exact low/high 7-bit coordinates instead of lossy integers.
    // Nibble coordinates remain split across five probes.
    constexpr size_t kBlocks = kM * kK / 32;
    std::vector<float> scale_probe(kBlocks);
    std::vector<int32_t> zero_point_probe(kBlocks);
    for (int shift = 0; shift < 5; ++shift) {
      for (size_t block = 0; block < kBlocks; ++block) {
        const size_t coordinate =
            shift % 2 == 0 ? block & 0x7F : block >> 7;
        scale_probe[block] = static_cast<float>(coordinate);
        zero_point_probe[block] = static_cast<int32_t>(coordinate);
      }
      std::vector<uint32_t> quantized(kM * kK);
      for (size_t index = 0; index < quantized.size(); ++index) {
        quantized[index] = (index >> (4 * shift)) & 0xF;
      }
      const auto probe =
          kernel_test_data::pack_q4nx(
              kM, kK, quantized, scale_probe, zero_point_probe);
      write_values(
          output / ("probe-nibble-shift" + std::to_string(shift) + ".bin"),
          probe);
      for (const auto& [row_block, col_block] :
           std::vector<std::pair<size_t, size_t>>{{32, 128}, {16, 128}}) {
        write_values(
            output / (
                "probe-" + std::to_string(row_block) + "x" +
                std::to_string(col_block) + "x16-nibble-shift" +
                std::to_string(shift) + ".bin"),
            pack_parameterized(
                kM, kK, row_block, col_block, 16, quantized,
                scale_probe, zero_point_probe));
      }
    }

    // One reference GEMM pins the accumulation contract the NumPy oracle must
    // reproduce within the shipped tolerances.
    const auto weight =
        kernel_test_data::make_q4_weight(kM, kK, 29);
    const auto activation = kernel_test_data::make_activation(kK, kN, 29);
    write_values(
        output / "matmul-shared-subset-29-29.bin",
        kernel_test_data::matmul_reference(
            weight.logical, activation, kM, kK, kN));
    write_values(
        output / "bf16-edge-values.bin",
        std::vector<bf16>{
            bf16(0.0f),
            bf16(-0.0f),
            bf16(std::numeric_limits<float>::denorm_min()),
            bf16(std::numeric_limits<float>::infinity()),
            bf16(-std::numeric_limits<float>::infinity()),
            bf16(std::numeric_limits<float>::quiet_NaN()),
            bf16(1.00390625f),
            bf16(1.01171875f),
            bf16(-1.00390625f),
            bf16(-1.01171875f)});
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "fastflowlm-q4nx-vector-dump: FAIL: %s\n", error.what());
    return 1;
  }
}
