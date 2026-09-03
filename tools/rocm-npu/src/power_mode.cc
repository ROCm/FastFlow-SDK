// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm_npu/platform.h"

#include <algorithm>
#include <cctype>

namespace rocm_npu {

std::optional<PowerMode> ParsePowerMode(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  if (normalized == "default") {
    return PowerMode::kDefault;
  }
  if (normalized == "powersaver") {
    return PowerMode::kPowerSaver;
  }
  if (normalized == "balanced") {
    return PowerMode::kBalanced;
  }
  if (normalized == "performance") {
    return PowerMode::kPerformance;
  }
  if (normalized == "turbo") {
    return PowerMode::kTurbo;
  }
  return std::nullopt;
}

std::optional<PowerMode> PowerModeFromValue(std::uint32_t value) {
  switch (value) {
    case 0:
      return PowerMode::kDefault;
    case 1:
      return PowerMode::kPowerSaver;
    case 2:
      return PowerMode::kBalanced;
    case 3:
      return PowerMode::kPerformance;
    case 4:
      return PowerMode::kTurbo;
    default:
      return std::nullopt;
  }
}

std::string PowerModeName(PowerMode mode) {
  switch (mode) {
    case PowerMode::kDefault:
      return "default";
    case PowerMode::kPowerSaver:
      return "powersaver";
    case PowerMode::kBalanced:
      return "balanced";
    case PowerMode::kPerformance:
      return "performance";
    case PowerMode::kTurbo:
      return "turbo";
  }
  return "unknown";
}

}  // namespace rocm_npu
