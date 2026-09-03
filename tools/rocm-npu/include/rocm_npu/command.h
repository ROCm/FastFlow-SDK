// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "rocm_npu/report.h"

namespace rocm_npu {

struct Invocation {
  std::string command;
  bool json = false;
  bool verbose = false;
  bool help = false;
  std::optional<std::filesystem::path> runtime_path;
  std::vector<std::string> arguments;
};

struct ParseResult {
  std::optional<Invocation> invocation;
  std::string error;
};

struct ExternalCommandResult {
  int exit_code = 2;
  std::string error;
};

ParseResult ParseInvocation(const std::vector<std::string>& arguments);
bool IsBuiltinCommand(const std::string& command);
bool IsValidExternalCommandName(const std::string& command);
std::string HelpText();
Report RunVersion(const Invocation& invocation);
Report RunProbe(const Invocation& invocation);
Report RunPowerMode(const Invocation& invocation);
Report RunDoctor(const Invocation& invocation);
bool HasValidSelfTestFixtureTypes(const nlohmann::json& fixture);
Report RunSelfTest(const Invocation& invocation);
Report RunPluginList(const Invocation& invocation);
ExternalCommandResult RunExternalCommand(const Invocation& invocation);
Report RunCommand(const Invocation& invocation);

}  // namespace rocm_npu
