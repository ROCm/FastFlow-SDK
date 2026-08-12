// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace kernel_validation {

inline std::string check_guards(
    std::span<const uint8_t> storage, size_t logical_bytes,
    size_t guard_bytes, uint8_t canary, const std::string& label) {
  if (storage.size() != logical_bytes + 2 * guard_bytes) {
    return label + " storage has an invalid size";
  }
  for (size_t i = 0; i < guard_bytes; ++i) {
    if (storage[i] != canary ||
        storage[guard_bytes + logical_bytes + i] != canary) {
      return label + " guard mismatch at byte " + std::to_string(i);
    }
  }
  return {};
}

inline std::string check_immutable(
    std::span<const uint8_t> storage, std::span<const uint8_t> expected,
    size_t guard_bytes, uint8_t canary, const std::string& label) {
  const std::string guard_error = check_guards(
      storage, expected.size(), guard_bytes, canary, label);
  if (!guard_error.empty()) return guard_error;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (storage[guard_bytes + i] != expected[i]) {
      return label + " input mutation at byte " + std::to_string(i);
    }
  }
  return {};
}

inline void validate_guards(
    std::span<const uint8_t> storage, size_t logical_bytes,
    size_t guard_bytes, uint8_t canary, const std::string& label) {
  const std::string error = check_guards(
      storage, logical_bytes, guard_bytes, canary, label);
  if (!error.empty()) throw std::runtime_error(error);
}

inline void validate_immutable(
    std::span<const uint8_t> storage, std::span<const uint8_t> expected,
    size_t guard_bytes, uint8_t canary, const std::string& label) {
  const std::string error = check_immutable(
      storage, expected, guard_bytes, canary, label);
  if (!error.empty()) throw std::runtime_error(error);
}

}  // namespace kernel_validation
