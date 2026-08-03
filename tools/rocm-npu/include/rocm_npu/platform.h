#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rocm_npu/report.h"

namespace rocm_npu {

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

std::filesystem::path ExecutablePath();
inline std::string PathToUtf8(const std::filesystem::path& path) {
  return path.u8string();
}
std::string OperatingSystemName();
std::string ProcessArchitecture();
ProbeResult CollectPlatformProbe(const ProbeOptions& options = {});
ProbeResult EvaluateWindowsDeviceRecords(const nlohmann::json& records);

}  // namespace rocm_npu
