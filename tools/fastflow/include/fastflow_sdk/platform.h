// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "fastflow_sdk/report.h"

namespace fastflow_sdk {

struct ProbeOptions {
  std::filesystem::path sys_module_root = "/sys/module";
  std::filesystem::path sys_accel_root = "/sys/class/accel";
  std::filesystem::path dev_accel_root = "/dev/accel";
  bool open_devices = true;
};

struct ProbeResult {
  nlohmann::json facts = nlohmann::json::object();
  std::vector<Check> checks;
  int detected_devices = 0;
  int usable_devices = 0;
  std::string target_family;
};

enum class PowerMode : std::uint8_t {
  kDefault = 0,
  kPowerSaver = 1,
  kBalanced = 2,
  kPerformance = 3,
  kTurbo = 4,
};

struct PowerModeResult {
  nlohmann::json devices = nlohmann::json::array();
  std::vector<Check> checks;
  int detected_devices = 0;
  int reported_devices = 0;
  int updated_devices = 0;
};

std::optional<PowerMode> ParsePowerMode(const std::string& value);
std::optional<PowerMode> PowerModeFromValue(std::uint32_t value);
std::string PowerModeName(PowerMode mode);
std::filesystem::path ExecutablePath();
inline std::string PathToUtf8(const std::filesystem::path& path) {
  return path.u8string();
}
std::string OperatingSystemName();
std::string ProcessArchitecture();
ProbeResult CollectPlatformProbe(const ProbeOptions& options = {});
ProbeResult EvaluateWindowsDeviceRecords(const nlohmann::json& records);
PowerModeResult QueryPlatformPowerMode(const ProbeOptions& options = {});
PowerModeResult SetPlatformPowerMode(PowerMode mode,
                                     const ProbeOptions& options = {});

}  // namespace fastflow_sdk
