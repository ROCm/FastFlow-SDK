// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "fastflow_sdk/command.h"

#include "fastflow_sdk/hrx_loader.h"
#include "fastflow_sdk/platform.h"

namespace fastflow_sdk {

Report RunProbe(const Invocation& invocation) {
  Report report = MakeReport("probe");
  ProbeResult platform = CollectPlatformProbe();
  report.data = std::move(platform.facts);
  report.data["detected_device_count"] = platform.detected_devices;
  report.data["usable_device_count"] = platform.usable_devices;
  for (const Check& check : platform.checks) {
    if (check.status == Status::kError) {
      AddCheck(report, check);
    }
  }

  HrxLoadResult runtime = HrxLibrary::Load(invocation.runtime_path);
  nlohmann::json candidates = nlohmann::json::array();
  for (const auto& candidate : runtime.candidates) {
    candidates.push_back(PathToUtf8(candidate));
  }
  report.data["runtime"] = {
      {"candidates", std::move(candidates)},
      {"loadable", runtime.library.has_value()},
      {"resolved_path",
       runtime.library ? PathToUtf8(runtime.library->path()) : std::string()},
      {"abi_version",
       runtime.library
           ? nlohmann::json{{"major", runtime.library->runtime_version_major()},
                            {"minor", runtime.library->runtime_version_minor()},
                            {"patch", runtime.library->runtime_version_patch()}}
           : nlohmann::json(nullptr)},
      {"missing_symbols", runtime.missing_symbols},
      {"load_errors", runtime.errors},
  };
  return report;
}

}  // namespace fastflow_sdk
