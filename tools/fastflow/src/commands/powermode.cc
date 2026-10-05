// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "fastflow_sdk/command.h"

#include "fastflow_sdk/platform.h"

#include <utility>

namespace fastflow_sdk {
namespace {

std::optional<std::string> PowerModeArgument(const Invocation& invocation) {
  if (invocation.arguments.empty()) {
    return std::nullopt;
  }
  return invocation.arguments.front();
}

void AddPowerModeResult(Report& report, PowerModeResult result) {
  report.data["detected_device_count"] = result.detected_devices;
  report.data["reported_device_count"] = result.reported_devices;
  report.data["updated_device_count"] = result.updated_devices;
  report.data["devices"] = std::move(result.devices);
  for (const Check& check : result.checks) {
    AddCheck(report, check);
  }
}

}  // namespace

Report RunPowerMode(const Invocation& invocation) {
  Report report = MakeReport(invocation.command);
  const std::optional<std::string> requested = PowerModeArgument(invocation);
  if (!requested) {
    AddPowerModeResult(report, QueryPlatformPowerMode());
    return report;
  }

  const std::optional<PowerMode> mode = ParsePowerMode(*requested);
  if (!mode) {
    AddCheck(report, {"powermode.arguments",
                      Status::kError,
                      "invalid power mode: " + *requested,
                      {{"valid_modes", nlohmann::json::array(
                                           {"default", "powersaver", "balanced",
                                            "performance", "turbo"})}}});
    return report;
  }

  report.data["requested_power_mode"] = PowerModeName(*mode);
  AddPowerModeResult(report, SetPlatformPowerMode(*mode));
  return report;
}

}  // namespace fastflow_sdk
