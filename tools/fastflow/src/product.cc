// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "fastflow_sdk/product.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <vector>

#include "fastflow_sdk/platform.h"

namespace fastflow_sdk {
namespace {

constexpr std::array<std::uint32_t, 64> kSha256Constants = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

std::uint32_t RotateRight(std::uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32 - bits));
}

class Sha256 {
 public:
  void Update(const std::uint8_t* data, std::size_t size) {
    total_size_ += size;
    while (size != 0) {
      const std::size_t count = std::min(size, block_.size() - block_size_);
      std::copy_n(data, count, block_.begin() + block_size_);
      block_size_ += count;
      data += count;
      size -= count;
      if (block_size_ == block_.size()) {
        Transform(block_.data());
        block_size_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> Finish() {
    const std::uint64_t bit_size = total_size_ * 8;
    block_[block_size_++] = 0x80;
    if (block_size_ > 56) {
      std::fill(block_.begin() + block_size_, block_.end(), std::uint8_t{0});
      Transform(block_.data());
      block_size_ = 0;
    }
    std::fill(block_.begin() + block_size_, block_.begin() + 56,
              std::uint8_t{0});
    for (unsigned index = 0; index < 8; ++index) {
      block_[63 - index] = static_cast<std::uint8_t>(bit_size >> (index * 8));
    }
    Transform(block_.data());
    std::array<std::uint8_t, 32> digest = {};
    for (std::size_t index = 0; index < state_.size(); ++index) {
      for (unsigned byte = 0; byte < 4; ++byte) {
        digest[index * 4 + byte] =
            static_cast<std::uint8_t>(state_[index] >> ((3 - byte) * 8));
      }
    }
    return digest;
  }

 private:
  void Transform(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> words = {};
    for (std::size_t index = 0; index < 16; ++index) {
      words[index] = (static_cast<std::uint32_t>(block[index * 4]) << 24) |
                     (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16) |
                     (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8) |
                     static_cast<std::uint32_t>(block[index * 4 + 3]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const std::uint32_t s0 = RotateRight(words[index - 15], 7) ^
                               RotateRight(words[index - 15], 18) ^
                               (words[index - 15] >> 3);
      const std::uint32_t s1 = RotateRight(words[index - 2], 17) ^
                               RotateRight(words[index - 2], 19) ^
                               (words[index - 2] >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = state_;
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t sum1 =
          RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 =
          h + sum1 + choice + kSha256Constants[index] + words[index];
      const std::uint32_t sum0 =
          RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  };
  std::array<std::uint8_t, 64> block_ = {};
  std::size_t block_size_ = 0;
  std::uint64_t total_size_ = 0;
};

std::optional<std::string> Sha256File(const std::filesystem::path& path,
                                      std::string& error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "could not open " + PathToUtf8(path);
    return std::nullopt;
  }
  Sha256 hash;
  std::array<char, 64 * 1024> buffer = {};
  while (input) {
    input.read(buffer.data(), buffer.size());
    const std::streamsize count = input.gcount();
    if (count > 0) {
      hash.Update(reinterpret_cast<const std::uint8_t*>(buffer.data()),
                  static_cast<std::size_t>(count));
    }
  }
  if (!input.eof()) {
    error = "could not read " + PathToUtf8(path);
    return std::nullopt;
  }
  std::ostringstream rendered;
  rendered << std::hex << std::setfill('0');
  for (std::uint8_t byte : hash.Finish()) {
    rendered << std::setw(2) << static_cast<unsigned>(byte);
  }
  return rendered.str();
}

bool IsSha256(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

}  // namespace

std::vector<std::filesystem::path> ProductManifestCandidates() {
  const std::filesystem::path executable = ExecutablePath();
  if (executable.empty()) {
    return {};
  }
  const std::filesystem::path prefix = executable.parent_path().parent_path();
  return {
      prefix / "manifest.json",
      prefix / "share" / "fastflow-sdk" / "manifest.json",
  };
}

std::optional<std::filesystem::path> FindProductManifest() {
  for (const auto& candidate : ProductManifestCandidates()) {
    std::error_code error;
    if (std::filesystem::is_regular_file(candidate, error)) {
      return candidate;
    }
  }
  return std::nullopt;
}

std::optional<nlohmann::json> ReadJsonFile(const std::filesystem::path& path,
                                           std::string& error) {
  std::ifstream input(path);
  if (!input) {
    error = "could not open " + PathToUtf8(path);
    return std::nullopt;
  }
  try {
    return nlohmann::json::parse(input);
  } catch (const nlohmann::json::exception& exception) {
    error = PathToUtf8(path) + ": " + exception.what();
    return std::nullopt;
  }
}

bool VerifyProductManifestFiles(const std::filesystem::path& manifest_path,
                                const nlohmann::json& manifest,
                                nlohmann::json& details, std::string& error) {
  details = nlohmann::json::object();
  error.clear();
  const auto files = manifest.find("files");
  if (files == manifest.end() || !files->is_array() || files->empty()) {
    error = "manifest file inventory is missing or empty";
    return false;
  }
  std::filesystem::path root = manifest_path.parent_path();
  if (root.filename() == "fastflow-sdk" &&
      root.parent_path().filename() == "share") {
    root = root.parent_path().parent_path();
  }
  std::error_code filesystem_error;
  const std::filesystem::path canonical_root =
      std::filesystem::weakly_canonical(root, filesystem_error);
  if (filesystem_error) {
    error = "could not resolve manifest root: " + filesystem_error.message();
    return false;
  }
  std::size_t verified = 0;
  std::set<std::string> verified_paths;
  for (const auto& record : *files) {
    if (!record.is_object() || !record.contains("path") ||
        !record["path"].is_string() || !record.contains("size") ||
        !record["size"].is_number_unsigned() || !record.contains("sha256") ||
        !record["sha256"].is_string()) {
      error = "manifest contains an invalid file record";
      return false;
    }
    const std::string relative_text = record["path"].get<std::string>();
    const std::string expected_digest = record["sha256"].get<std::string>();
    const std::filesystem::path relative =
        std::filesystem::u8path(relative_text);
    if (relative_text.empty() ||
        relative_text.find('\\') != std::string::npos ||
        relative.is_absolute() ||
        std::find(relative.begin(), relative.end(), "..") != relative.end() ||
        !IsSha256(expected_digest) ||
        !verified_paths.insert(relative_text).second) {
      error = "manifest contains an unsafe file record: " + relative_text;
      return false;
    }
    const std::filesystem::path path =
        std::filesystem::weakly_canonical(root / relative, filesystem_error);
    if (filesystem_error) {
      error = relative_text + ": " + filesystem_error.message();
      return false;
    }
    const std::filesystem::path from_root =
        std::filesystem::relative(path, canonical_root, filesystem_error);
    if (filesystem_error || from_root.empty() || from_root.is_absolute() ||
        *from_root.begin() == ".." ||
        !std::filesystem::is_regular_file(path, filesystem_error)) {
      error =
          "manifest file is missing or outside the product: " + relative_text;
      return false;
    }
    const auto expected_size = record["size"].get<std::uint64_t>();
    const auto actual_size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error || actual_size != expected_size) {
      error = "manifest file size does not match: " + relative_text;
      return false;
    }
    const auto actual_digest = Sha256File(path, error);
    if (!actual_digest || *actual_digest != expected_digest) {
      if (actual_digest) {
        error = "manifest file SHA256 does not match: " + relative_text;
      }
      return false;
    }
    ++verified;
  }
  details = {{"declared", true},
             {"root", PathToUtf8(canonical_root)},
             {"verified_file_count", verified}};
  return true;
}

std::filesystem::path SelfTestDataRoot() {
  const std::filesystem::path executable = ExecutablePath();
  if (executable.empty()) {
    return {};
  }
  return executable.parent_path().parent_path() / "share" / "fastflow-sdk" /
         "self-test";
}

}  // namespace fastflow_sdk
