// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kernel_test_data.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

#include "tensor_utils/q4_npu_eXpress.hpp"

namespace kernel_test_data {
namespace {

struct Tensor {
  std::string name;
  std::string dtype;
  std::vector<size_t> shape;
  std::vector<uint8_t> data;
};

template <typename T>
std::vector<uint8_t> as_bytes(const std::vector<T>& values) {
  const auto* begin = reinterpret_cast<const uint8_t*>(values.data());
  return {begin, begin + values.size() * sizeof(T)};
}

void write_u64(std::ofstream& output, uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    output.put(static_cast<char>((value >> (8 * byte)) & 0xFF));
  }
}

std::string shape_json(const std::vector<size_t>& shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) result += ",";
    result += std::to_string(shape[i]);
  }
  return result + "]";
}

void write_safetensors(
    const std::filesystem::path& path,
    const std::vector<Tensor>& tensors) {
  std::string header = "{\"__metadata__\":{\"format\":\"pt\"}";
  size_t offset = 0;
  for (const Tensor& tensor : tensors) {
    const size_t bytes = tensor.data.size();
    header += ",\"" + tensor.name + "\":{\"dtype\":\"" + tensor.dtype +
              "\",\"shape\":" + shape_json(tensor.shape) +
              ",\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";
  while (header.size() % 8) header.push_back(' ');

  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot create " + path.string());
  write_u64(output, header.size());
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const Tensor& tensor : tensors) {
    output.write(
        reinterpret_cast<const char*>(tensor.data.data()),
        static_cast<std::streamsize>(tensor.data.size()));
  }
}

Tensor make_bf16_tensor(
    std::string name, std::vector<size_t> shape, int salt) {
  size_t count = 1;
  for (size_t dimension : shape) count *= dimension;
  std::vector<bf16> values;
  values.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const int centered =
        static_cast<int>((i * 17 + (i / 7) * 13 + salt * 19) % 61) - 30;
    values.emplace_back((i % 37 == 0) ? 0.0f : centered / 32.0f);
  }
  return {std::move(name), "BF16", std::move(shape), as_bytes(values)};
}

uint32_t quantized_value(
    Q4Profile profile, size_t row, size_t column_word,
    size_t nibble, int salt) {
  const uint32_t levels =
      profile == Q4Profile::kSharedSubset ? 8 : 16;
  return static_cast<uint32_t>(
      (row * 3 + column_word * 11 + nibble * 5 + salt * 7) %
      levels);
}

float quantized_scale(
    size_t row, size_t column_block, int salt) {
  return
      (1 + static_cast<int>((row * 5 + column_block + salt) % 7)) /
      64.0f;
}

std::vector<Tensor> make_quantized_weight_tensors(
    const std::string& name, size_t rows, size_t columns, int salt,
    Q4Profile profile = Q4Profile::kSharedSubset) {
  if (rows % 32 || columns % 256) {
    throw std::runtime_error(
        "Q4NX dimensions must be row/column chunk aligned");
  }
  std::vector<int32_t> packed(rows * columns / 8);
  for (size_t word = 0; word < packed.size(); ++word) {
    uint32_t value = 0;
    const size_t row = word / (columns / 8);
    const size_t column_word = word % (columns / 8);
    for (size_t nibble = 0; nibble < 8; ++nibble) {
      const uint32_t q = quantized_value(
          profile, row, column_word, nibble, salt);
      value |= q << (nibble * 4);
    }
    packed[word] = static_cast<int32_t>(value);
  }
  std::vector<bf16> scales(rows * columns / 32);
  std::vector<int32_t> zero_points(rows * columns / 32);
  for (size_t block = 0; block < scales.size(); ++block) {
    const size_t row = block / (columns / 32);
    const size_t column_block = block % (columns / 32);
    scales[block] = bf16(
        quantized_scale(row, column_block, salt));
    zero_points[block] =
        profile == Q4Profile::kFullAffine
            ? static_cast<int32_t>(
                  (row * 3 + column_block + salt) % 5) - 2
            : 0;
  }
  return {
      {name, "I32", {rows, columns / 8}, as_bytes(packed)},
      {name + "_scale", "BF16", {rows, columns / 32}, as_bytes(scales)},
      {name + "_zero_point", "I32", {rows, columns / 32},
       as_bytes(zero_points)},
  };
}

void append(std::vector<Tensor>& destination, std::vector<Tensor> source) {
  for (Tensor& tensor : source) destination.push_back(std::move(tensor));
}

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    path = std::filesystem::temp_directory_path() /
           (std::string(kTemporaryDirectoryPrefix) + std::to_string(nonce));
    std::filesystem::create_directories(path);
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
  std::filesystem::path path;
};

}  // namespace

Q4Weight make_q4_weight(
    size_t rows, size_t columns, int salt, Q4Profile profile) {
  std::vector<Tensor> tensors;
  constexpr size_t kDummyRows = 32;
  constexpr size_t kDummyColumns = 256;
  append(tensors, make_quantized_weight_tensors(
                      "model.embed_tokens.weight",
                      kDummyRows, kDummyColumns, salt + 1));
  tensors.push_back(make_bf16_tensor(
      "model.layers.0.input_layernorm.weight", {columns}, salt + 2));
  append(tensors, make_quantized_weight_tensors(
                      "model.layers.0.self_attn.q_proj.weight",
                      rows, columns, salt, profile));
  for (const char* suffix : {
           "k_proj.weight", "v_proj.weight", "o_proj.weight"}) {
    append(tensors, make_quantized_weight_tensors(
                        std::string("model.layers.0.self_attn.") + suffix,
                        kDummyRows, kDummyColumns, salt + 3));
  }
  for (const char* suffix : {
           "gate_proj.weight", "up_proj.weight", "down_proj.weight"}) {
    append(tensors, make_quantized_weight_tensors(
                        std::string("model.layers.0.mlp.") + suffix,
                        kDummyRows, kDummyColumns, salt + 4));
  }
  tensors.push_back(make_bf16_tensor(
      "model.layers.0.post_attention_layernorm.weight",
      {columns}, salt + 5));
  tensors.push_back(
      make_bf16_tensor("model.norm.weight", {columns}, salt + 6));
  append(tensors, make_quantized_weight_tensors(
                      "lm_head.weight", kDummyRows, kDummyColumns, salt + 7));

  TemporaryDirectory temporary;
  write_safetensors(temporary.path / "model.safetensors", tensors);
  const auto q4nx_path = temporary.path / "model.q4nx";
  Q4NX converter(temporary.path.string());
  converter.convert_model(q4nx_path.string());

  SafeTensors converted(q4nx_path.string());
  const std::string name = "model.layers.0.self_attn.q_proj.weight";
  const tensor_metadata metadata = converted.get_tensor_metadata(name);
  if (metadata.dtype != "Q4NX" ||
      metadata.shape != std::vector<size_t>({rows, columns})) {
    throw std::runtime_error("unexpected generated Q4NX tensor metadata");
  }
  bytes packed_storage(metadata.byte_size);
  converted.load_weights(packed_storage, name);
  Q4Weight result{rows, columns, {}, {}, {}, {}, {}};
  result.packed.assign(
      packed_storage.data(), packed_storage.data() + packed_storage.size());

  bytes reference_storage(rows * columns * sizeof(bf16));
  Q4NX::q4nx_dequantize<bf16>(
      reference_storage, packed_storage, static_cast<int>(columns));
  const auto* exported_reference =
      reinterpret_cast<const bf16*>(reference_storage.data());
  result.logical.assign(
      exported_reference, exported_reference + rows * columns);
  result.signed_nibble_logical.resize(rows * columns);
  result.qwen_affine_logical.resize(rows * columns);
  result.lfm_affine_logical.resize(rows * columns);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t column = 0; column < columns; ++column) {
      const size_t linear = row * columns + column;
      const size_t word = row * (columns / 8) + column / 8;
      const size_t column_word = word % (columns / 8);
      const size_t nibble = column % 8;
      const uint32_t q = quantized_value(
          profile, row, column_word, nibble, salt);
      const size_t block = row * (columns / 32) + column / 32;
      const size_t column_block = block % (columns / 32);
      const float scale =
          quantized_scale(row, column_block, salt);
      const int32_t zero_point =
          profile == Q4Profile::kFullAffine
              ? static_cast<int32_t>(
                    (row * 3 + column_block + salt) % 5) - 2
              : 0;
      const bf16 expected(
          (static_cast<int32_t>(q) - zero_point) * scale);
      result.signed_nibble_logical[linear] = bf16(
          (static_cast<int32_t>(q >= 8 ? q - 16 : q) - zero_point) *
          scale);
      // These are production-xclbin contracts, intentionally distinct from
      // the generic Q4NX CPU dequantization above. Qwen adds the stored I32
      // zero point after scaling; LFM uses signed nibbles and ignores it.
      result.qwen_affine_logical[linear] =
          bf16(static_cast<int32_t>(q) * scale + zero_point);
      result.lfm_affine_logical[linear] =
          bf16(static_cast<int32_t>(q >= 8 ? q - 16 : q) * scale);
      if (std::memcmp(&expected, &result.logical[linear], sizeof(bf16)) != 0) {
        throw std::runtime_error(
            "exported Q4NX CPU reference disagrees with generated source");
      }
    }
  }
  return result;
}

std::vector<bf16> make_activation(
    size_t rows, size_t columns, int salt) {
  std::vector<bf16> result(rows * columns);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t column = 0; column < columns; ++column) {
      const size_t index = row * columns + column;
      const int centered = static_cast<int>(
          (row * 13 + column * 7 + (row ^ column) * 3 + salt * 11) % 41) - 20;
      result[index] =
          bf16((index % 53 == 0) ? 0.0f : centered / 64.0f);
    }
  }
  return result;
}

std::vector<bf16> matmul_reference(
    const std::vector<bf16>& weight,
    const std::vector<bf16>& activation,
    size_t m, size_t k, size_t n) {
  if (weight.size() != m * k || activation.size() != k * n) {
    throw std::runtime_error("invalid matrix sizes for CPU reference");
  }
  std::vector<bf16> output(m * n);
  for (size_t row = 0; row < m; ++row) {
    for (size_t column = 0; column < n; ++column) {
      float accumulator = 0.0f;
      for (size_t inner = 0; inner < k; ++inner) {
        accumulator += static_cast<float>(weight[row * k + inner]) *
                       static_cast<float>(activation[inner * n + column]);
      }
      output[row * n + column] = bf16(accumulator);
    }
  }
  return output;
}

size_t mm_layout_logical_index(size_t physical_index) {
  if (physical_index >= (1u << 20)) {
    throw std::runtime_error("MM layout physical index is out of range");
  }
  // Each entry maps one physical-address bit to the corresponding bit in the
  // row-major logical index. Coordinate-coded NPU probes independently cover
  // all 2^20 addresses and establish that this is a bijection.
  constexpr int kPhysicalToLogicalBit[] = {
      10, 11, 12, 3, 4, 5, 6, 7, 9, 0,
      1,  2,  13, 14, 8, 16, 15, 17, 18, 19};
  size_t logical_index = 0;
  for (size_t physical_bit = 0;
       physical_bit < std::size(kPhysicalToLogicalBit);
       ++physical_bit) {
    logical_index |= ((physical_index >> physical_bit) & 1)
                     << kPhysicalToLogicalBit[physical_bit];
  }
  return logical_index;
}

std::vector<bf16> to_mm_layout(
    const std::vector<bf16>& logical, size_t rows, size_t columns) {
  if (rows != 1024 || columns != 1024 ||
      logical.size() != rows * columns) {
    throw std::runtime_error(
        "MM layout contract is defined for 1024x1024 BF16 weights");
  }
  std::vector<bf16> physical(logical.size());
  for (size_t physical_index = 0;
       physical_index < physical.size(); ++physical_index) {
    physical[physical_index] =
        logical[mm_layout_logical_index(physical_index)];
  }
  return physical;
}

size_t gemm_activation_logical_index(size_t physical_index) {
  if (physical_index >= (1u << 17)) {
    throw std::runtime_error(
        "GEMM activation physical index is out of range");
  }
  // Physical-address bits map to KxN row-major logical-index bits. Sparse
  // basis probes cover every K bit and all N coordinates.
  constexpr int kPhysicalToLogicalBit[] = {
      0, 1, 2, 10, 11, 12, 13, 14, 15,
      7, 8, 9, 3,  4,  5,  6,  16};
  size_t logical_index = 0;
  for (size_t physical_bit = 0;
       physical_bit < std::size(kPhysicalToLogicalBit);
       ++physical_bit) {
    logical_index |= ((physical_index >> physical_bit) & 1)
                     << kPhysicalToLogicalBit[physical_bit];
  }
  return logical_index;
}

std::vector<bf16> to_gemm_activation_layout(
    const std::vector<bf16>& logical, size_t k, size_t n) {
  if (k != 1024 || n != 128 || logical.size() != k * n) {
    throw std::runtime_error(
        "GEMM activation layout is defined for 1024x128 BF16 inputs");
  }
  std::vector<bf16> physical(logical.size());
  for (size_t physical_index = 0;
       physical_index < physical.size(); ++physical_index) {
    physical[physical_index] =
        logical[gemm_activation_logical_index(physical_index)];
  }
  return physical;
}

}  // namespace kernel_test_data
