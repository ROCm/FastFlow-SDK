// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Verifies external plugin discovery, invocation, and process behavior.

#include <filesystem>
#include <iostream>
#include <string>

#include "platform.h"
#include "plugin_dispatch_platform.h"
#include "fastflow_sdk/command.h"
#include "fastflow_sdk/platform.h"

#define REQUIRE(condition)                                             \
  do {                                                                 \
    if (!(condition)) {                                                \
      std::cerr << "plugin dispatch check failed at line " << __LINE__ \
                << ": " #condition << std::endl;                       \
      return 1;                                                        \
    }                                                                  \
  } while (false)

int main(int argc, char** argv) {
  REQUIRE(argc == 3);
  const std::filesystem::path source = argv[1];
  const std::filesystem::path runtime = argv[2];
  const std::filesystem::path root = fastflow_sdk::test::TestRoot("dispatch");
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const std::filesystem::path plugin =
      fastflow_sdk::test::PluginExecutable(root, "echo");
  fastflow_sdk::test::SetEnvironmentPath("FASTFLOW_SDK_PLUGIN_PATH", root);
  std::filesystem::copy_file(source, plugin);
  fastflow_sdk::test::MakeExecutable(plugin);

  fastflow_sdk::Invocation invocation;
  invocation.command = "echo";
  invocation.json = true;
  invocation.arguments = {"--require-json",  "--verify-arguments",
                          "space argument",  "unicode-\u00e9",
                          "quote\"argument", "trailing\\",
                          "--exit=37"};
  const fastflow_sdk::ExternalCommandResult result =
      fastflow_sdk::RunExternalCommand(invocation);
  if (!result.error.empty() || result.exit_code != 37) {
    std::cerr << "initial dispatch result: exit=" << result.exit_code
              << " error=" << result.error << std::endl;
  }
  REQUIRE(result.error.empty());
  REQUIRE(result.exit_code == 37);

  invocation.runtime_path = root / "missing-hrx";
  const fastflow_sdk::ExternalCommandResult invalid_runtime =
      fastflow_sdk::RunExternalCommand(invocation);
  if (invalid_runtime.exit_code != 2 ||
      invalid_runtime.error.find("explicit HRX runtime") == std::string::npos) {
    std::cerr << "invalid runtime result: exit=" << invalid_runtime.exit_code
              << " error=" << invalid_runtime.error << std::endl;
  }
  REQUIRE(invalid_runtime.exit_code == 2);
  REQUIRE(invalid_runtime.error.find("explicit HRX runtime") !=
          std::string::npos);
  invocation.runtime_path = runtime;
  invocation.arguments = {"--require-runtime"};
  const fastflow_sdk::ExternalCommandResult valid_runtime =
      fastflow_sdk::RunExternalCommand(invocation);
  if (!valid_runtime.error.empty() || valid_runtime.exit_code != 0) {
    std::cerr << "valid runtime result: exit=" << valid_runtime.exit_code
              << " error=" << valid_runtime.error << std::endl;
  }
  REQUIRE(valid_runtime.error.empty());
  REQUIRE(valid_runtime.exit_code == 0);
  invocation.runtime_path.reset();

  fastflow_sdk::test::VerifySignalForwarding(invocation);

  const std::filesystem::path recursive =
      fastflow_sdk::test::PluginExecutable(root, "recurse");
  std::error_code hard_link_error;
  std::filesystem::create_hard_link(fastflow_sdk::ExecutablePath(), recursive,
                                    hard_link_error);
  if (!hard_link_error) {
    invocation.command = "recurse";
    const fastflow_sdk::ExternalCommandResult recursion =
        fastflow_sdk::RunExternalCommand(invocation);
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
  const fastflow_sdk::ExternalCommandResult missing =
      fastflow_sdk::RunExternalCommand(invocation);
  if (missing.exit_code != 2 || missing.error.empty()) {
    std::cerr << "missing command result: exit=" << missing.exit_code
              << " error=" << missing.error << std::endl;
  }
  REQUIRE(missing.exit_code == 2);
  REQUIRE(!missing.error.empty());
  std::filesystem::remove_all(root);
  return 0;
}
