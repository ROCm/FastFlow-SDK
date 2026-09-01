// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm_npu/command.h"

#include <sstream>
#include <system_error>
#include <unordered_set>
#include <utility>

#include "rocm_npu/hrx_loader.h"
#include "rocm_npu/platform.h"

namespace rocm_npu {
namespace {

const std::unordered_set<std::string> kBuiltins = {"version", "probe", "doctor",
                                                   "self-test", "plugin"};

}  // namespace

bool IsBuiltinCommand(const std::string& command) {
  return kBuiltins.count(command) != 0;
}

bool IsValidExternalCommandName(const std::string& command) {
  if (command.empty()) {
    return false;
  }
  for (const unsigned char character : command) {
    if ((character < 'a' || character > 'z') &&
        (character < '0' || character > '9') && character != '-') {
      return false;
    }
  }
  return true;
}

ParseResult ParseInvocation(const std::vector<std::string>& arguments) {
  Invocation invocation;
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string& argument = arguments[index];
    if (argument == "--json") {
      invocation.json = true;
    } else if (argument == "--verbose") {
      invocation.verbose = true;
    } else if (argument == "--help" || argument == "-h") {
      if (!invocation.command.empty() &&
          !IsBuiltinCommand(invocation.command)) {
        invocation.arguments.push_back(argument);
      } else {
        invocation.help = true;
      }
    } else if (argument == "--runtime") {
      if (++index == arguments.size()) {
        return {std::nullopt, "--runtime requires a path"};
      }
      invocation.runtime_path = arguments[index];
    } else if (argument.rfind("--runtime=", 0) == 0) {
      if (argument.size() == 10) {
        return {std::nullopt, "--runtime requires a non-empty path"};
      }
      invocation.runtime_path = argument.substr(10);
    } else if (argument.rfind("-", 0) == 0 && invocation.command.empty()) {
      return {std::nullopt, "unknown option: " + argument};
    } else if (invocation.command.empty()) {
      invocation.command = argument;
    } else {
      invocation.arguments.push_back(argument);
    }
  }

  if (invocation.command.empty()) {
    if (invocation.help) {
      return {std::move(invocation), {}};
    }
    return {std::nullopt, "a command is required"};
  }
  if (!IsBuiltinCommand(invocation.command)) {
    if (!IsValidExternalCommandName(invocation.command)) {
      return {std::nullopt,
              "invalid external command name: " + invocation.command};
    }
    return {std::move(invocation), {}};
  }
  if (invocation.command == "plugin") {
    if (invocation.arguments != std::vector<std::string>{"list"}) {
      return {std::nullopt, "plugin requires the subcommand: list"};
    }
    return {std::move(invocation), {}};
  }
  if (!invocation.arguments.empty()) {
    return {std::nullopt,
            invocation.command + " does not accept positional arguments"};
  }
  return {std::move(invocation), {}};
}

std::string HelpText() {
  return R"(Usage: rocm-npu [global-options] <command> [global-options]

Commands:
  version    Report product, build, manifest, and HRX library information
  probe      Collect platform, driver, firmware, device, and HRX facts
  doctor     Evaluate package, runtime, and device health
  self-test  Dispatch the packaged bounded deterministic XADX test
  plugin list
             Discover external rocm-npu-* plugin executables without running them
  <name>     Execute the external rocm-npu-<name> plugin

Global options:
  --json             Emit exactly one JSON report to stdout
  --runtime <path>   Use an explicit HRX library file or directory
  --verbose          Include additional diagnostics
  --help, -h         Show this help
)";
}

Report RunCommand(const Invocation& invocation) {
  Report report;
  if (invocation.command == "version") {
    report = RunVersion(invocation);
  } else if (invocation.command == "probe") {
    report = RunProbe(invocation);
  } else if (invocation.command == "doctor") {
    report = RunDoctor(invocation);
  } else if (invocation.command == "self-test") {
    report = RunSelfTest(invocation);
  } else {
    report = RunPluginList(invocation);
  }
  if (invocation.verbose) {
    nlohmann::json candidates = nlohmann::json::array();
    for (const auto& candidate :
         HrxLibrary::Candidates(invocation.runtime_path)) {
      candidates.push_back(PathToUtf8(candidate));
    }
    std::error_code error;
    const std::filesystem::path working_directory =
        std::filesystem::current_path(error);
    report.data["diagnostics"] = {
        {"current_working_directory",
         error ? std::string() : PathToUtf8(working_directory)},
        {"explicit_runtime",
         invocation.runtime_path
             ? nlohmann::json(PathToUtf8(*invocation.runtime_path))
             : nlohmann::json(nullptr)},
        {"runtime_candidates", std::move(candidates)},
    };
  }
  return report;
}

}  // namespace rocm_npu
