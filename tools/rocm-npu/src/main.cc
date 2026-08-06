// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "main.h"
#include "rocm_npu/command.h"
#include "rocm_npu/report.h"

namespace rocm_npu {

int RunMain(std::vector<std::string> arguments) {
  const bool json_requested =
      std::find(arguments.begin(), arguments.end(), "--json") !=
      arguments.end();
  const rocm_npu::ParseResult parsed = rocm_npu::ParseInvocation(arguments);
  if (!parsed.invocation) {
    if (json_requested) {
      rocm_npu::Report report = rocm_npu::MakeReport("invocation");
      rocm_npu::AddCheck(
          report,
          {"invocation.arguments", rocm_npu::Status::kError, parsed.error, {}});
      rocm_npu::RenderJson(report, std::cout);
    } else {
      std::cerr << "rocm-npu: " << parsed.error << '\n'
                << "Try 'rocm-npu --help' for usage.\n";
    }
    return 2;
  }
  if (parsed.invocation->help) {
    std::cout << rocm_npu::HelpText();
    return 0;
  }

  try {
    if (!rocm_npu::IsBuiltinCommand(parsed.invocation->command)) {
      const rocm_npu::ExternalCommandResult result =
          rocm_npu::RunExternalCommand(*parsed.invocation);
      if (!result.error.empty()) {
        if (parsed.invocation->json) {
          rocm_npu::Report report =
              rocm_npu::MakeReport(parsed.invocation->command);
          rocm_npu::AddCheck(
              report,
              {"plugin.dispatch", rocm_npu::Status::kError, result.error, {}});
          rocm_npu::RenderJson(report, std::cout);
        } else {
          std::cerr << "rocm-npu: " << result.error << '\n';
        }
      }
      return result.exit_code;
    }
    const rocm_npu::Report report = rocm_npu::RunCommand(*parsed.invocation);
    if (parsed.invocation->json) {
      rocm_npu::RenderJson(report, std::cout);
    } else {
      rocm_npu::RenderHuman(report, std::cout);
    }
    return rocm_npu::ExitCode(report);
  } catch (const std::exception& error) {
    rocm_npu::Report report =
        rocm_npu::MakeReport(parsed.invocation->command);
    rocm_npu::AddCheck(
        report, {"internal.error", rocm_npu::Status::kError, error.what(), {}});
    if (parsed.invocation->json) {
      rocm_npu::RenderJson(report, std::cout);
    } else {
      rocm_npu::RenderHuman(report, std::cerr);
    }
    return 2;
  }
}

}  // namespace rocm_npu
