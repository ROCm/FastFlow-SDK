// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "main.h"
#include "fastflow_sdk/command.h"
#include "fastflow_sdk/report.h"

namespace fastflow_sdk {

int RunMain(std::vector<std::string> arguments) {
  const bool json_requested = std::find(arguments.begin(), arguments.end(),
                                        "--json") != arguments.end();
  const fastflow_sdk::ParseResult parsed =
      fastflow_sdk::ParseInvocation(arguments);
  if (!parsed.invocation) {
    if (json_requested) {
      fastflow_sdk::Report report = fastflow_sdk::MakeReport("invocation");
      fastflow_sdk::AddCheck(report, {"invocation.arguments",
                                      fastflow_sdk::Status::kError,
                                      parsed.error,
                                      {}});
      fastflow_sdk::RenderJson(report, std::cout);
    } else {
      std::cerr << "fastflow: " << parsed.error << '\n'
                << "Try 'fastflow --help' for usage.\n";
    }
    return 2;
  }
  if (parsed.invocation->help) {
    std::cout << fastflow_sdk::HelpText();
    return 0;
  }

  try {
    if (!fastflow_sdk::IsBuiltinCommand(parsed.invocation->command)) {
      const fastflow_sdk::ExternalCommandResult result =
          fastflow_sdk::RunExternalCommand(*parsed.invocation);
      if (!result.error.empty()) {
        if (parsed.invocation->json) {
          fastflow_sdk::Report report =
              fastflow_sdk::MakeReport(parsed.invocation->command);
          fastflow_sdk::AddCheck(report, {"plugin.dispatch",
                                          fastflow_sdk::Status::kError,
                                          result.error,
                                          {}});
          fastflow_sdk::RenderJson(report, std::cout);
        } else {
          std::cerr << "fastflow: " << result.error << '\n';
        }
      }
      return result.exit_code;
    }
    const fastflow_sdk::Report report =
        fastflow_sdk::RunCommand(*parsed.invocation);
    if (parsed.invocation->json) {
      fastflow_sdk::RenderJson(report, std::cout);
    } else {
      fastflow_sdk::RenderHuman(report, std::cout);
    }
    return fastflow_sdk::ExitCode(report);
  } catch (const std::exception& error) {
    fastflow_sdk::Report report =
        fastflow_sdk::MakeReport(parsed.invocation->command);
    fastflow_sdk::AddCheck(
        report,
        {"internal.error", fastflow_sdk::Status::kError, error.what(), {}});
    if (parsed.invocation->json) {
      fastflow_sdk::RenderJson(report, std::cout);
    } else {
      fastflow_sdk::RenderHuman(report, std::cerr);
    }
    return 2;
  }
}

}  // namespace fastflow_sdk
