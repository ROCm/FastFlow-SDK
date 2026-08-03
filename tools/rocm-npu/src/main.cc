#include <algorithm>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "rocm_npu/command.h"
#include "rocm_npu/report.h"

namespace {
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
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  std::vector<std::string> arguments;
  arguments.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  for (int index = 1; index < argc; ++index) {
    arguments.push_back(
        std::filesystem::path(std::wstring(argv[index])).u8string());
  }
  return RunMain(std::move(arguments));
}
#else
int main(int argc, char** argv) {
  std::vector<std::string> arguments;
  arguments.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  for (int index = 1; index < argc; ++index) {
    arguments.emplace_back(argv[index]);
  }
  return RunMain(std::move(arguments));
}
#endif
