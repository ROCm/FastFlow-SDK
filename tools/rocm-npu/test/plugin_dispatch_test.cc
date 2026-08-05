// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Verifies external plugin discovery, invocation, and process behavior.

#include <filesystem>
#include <iostream>
#include <string>

#include "platform.h"
#include "plugin_dispatch_platform.h"
#include "rocm_npu/command.h"
#include "rocm_npu/platform.h"

#define REQUIRE(condition)                                                   \
  do {                                                                       \
    if (!(condition)) {                                                      \
      std::cerr << "plugin dispatch check failed at line " << __LINE__       \
                << ": " #condition << std::endl;                             \
      return 1;                                                              \
    }                                                                        \
  } while (false)

int main(int argc, char** argv) {
  REQUIRE(argc == 3);
  const std::filesystem::path source = argv[1];
  const std::filesystem::path runtime = argv[2];
  const std::filesystem::path root = rocm_npu::test::TestRoot("dispatch");
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const std::filesystem::path plugin =
      rocm_npu::test::PluginExecutable(root, "echo");
  rocm_npu::test::SetEnvironmentPath("ROCM_NPU_PLUGIN_PATH", root);
  std::filesystem::copy_file(source, plugin);
  rocm_npu::test::MakeExecutable(plugin);

  rocm_npu::Invocation invocation;
  invocation.command = "echo";
  invocation.json = true;
  invocation.arguments = {
      "--require-json", "--verify-arguments", "space argument",
      "unicode-\u00e9", "quote\"argument", "trailing\\", "--exit=37"};
  const rocm_npu::ExternalCommandResult result =
      rocm_npu::RunExternalCommand(invocation);
  if (!result.error.empty() || result.exit_code != 37) {
    std::cerr << "initial dispatch result: exit=" << result.exit_code
              << " error=" << result.error << std::endl;
  }
  REQUIRE(result.error.empty());
  REQUIRE(result.exit_code == 37);

  invocation.runtime_path = root / "missing-hrx";
  const rocm_npu::ExternalCommandResult invalid_runtime =
      rocm_npu::RunExternalCommand(invocation);
  if (invalid_runtime.exit_code != 2 ||
      invalid_runtime.error.find("explicit HRX runtime") ==
          std::string::npos) {
    std::cerr << "invalid runtime result: exit=" << invalid_runtime.exit_code
              << " error=" << invalid_runtime.error << std::endl;
  }
  REQUIRE(invalid_runtime.exit_code == 2);
  REQUIRE(invalid_runtime.error.find("explicit HRX runtime") !=
          std::string::npos);
  invocation.runtime_path = runtime;
  invocation.arguments = {"--require-runtime"};
  const rocm_npu::ExternalCommandResult valid_runtime =
      rocm_npu::RunExternalCommand(invocation);
  if (!valid_runtime.error.empty() || valid_runtime.exit_code != 0) {
    std::cerr << "valid runtime result: exit=" << valid_runtime.exit_code
              << " error=" << valid_runtime.error << std::endl;
  }
  REQUIRE(valid_runtime.error.empty());
  REQUIRE(valid_runtime.exit_code == 0);
  invocation.runtime_path.reset();

  rocm_npu::test::VerifySignalForwarding(invocation);

  const std::filesystem::path recursive =
      rocm_npu::test::PluginExecutable(root, "recurse");
  std::error_code hard_link_error;
  std::filesystem::create_hard_link(rocm_npu::ExecutablePath(), recursive,
                                    hard_link_error);
  if (!hard_link_error) {
    invocation.command = "recurse";
    const rocm_npu::ExternalCommandResult recursion =
        rocm_npu::RunExternalCommand(invocation);
    if (recursion.exit_code != 2 ||
        recursion.error.find("core executable") == std::string::npos) {
      std::cerr << "hard-link recursion result: exit=" << recursion.exit_code
                << " error=" << recursion.error << std::endl;
    }
    REQUIRE(recursion.exit_code == 2);
    REQUIRE(recursion.error.find("core executable") != std::string::npos);
  } else {
    std::cerr << "hard-link recursion check skipped: "
              << hard_link_error.message() << std::endl;
  }

  invocation.command = "missing";
  const rocm_npu::ExternalCommandResult missing =
      rocm_npu::RunExternalCommand(invocation);
  if (missing.exit_code != 2 || missing.error.empty()) {
    std::cerr << "missing command result: exit=" << missing.exit_code
              << " error=" << missing.error << std::endl;
  }
  REQUIRE(missing.exit_code == 2);
  REQUIRE(!missing.error.empty());
  std::filesystem::remove_all(root);
  return 0;
}
