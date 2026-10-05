// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

#include "platform.h"
#include "fastflow_sdk/command.h"

int main() {
  const std::filesystem::path root = fastflow_sdk::test::TestRoot("plugin");
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const std::filesystem::path system_root = root / "system";
  std::filesystem::create_directories(system_root);
  const std::filesystem::path plugin =
      fastflow_sdk::test::PluginExecutable(root, "example");
  const std::filesystem::path shadowed =
      fastflow_sdk::test::PluginExecutable(system_root, "example");
  const std::filesystem::path path_plugin =
      fastflow_sdk::test::PluginExecutable(system_root, "path-only");
  const std::filesystem::path invalid =
      fastflow_sdk::test::PluginExecutable(root, "Bad");
  fastflow_sdk::test::SetEnvironmentPath("FASTFLOW_SDK_PLUGIN_PATH", root);
  fastflow_sdk::test::SetEnvironmentPath("PATH", system_root);
  for (const auto& path : {plugin, shadowed, path_plugin, invalid}) {
    std::ofstream(path) << "fixture";
    fastflow_sdk::test::MakeExecutable(path);
  }

  fastflow_sdk::Invocation invocation;
  invocation.command = "plugin";
  invocation.arguments = {"list"};
  fastflow_sdk::Report report = fastflow_sdk::RunPluginList(invocation);
  bool found = false;
  bool found_path = false;
  bool found_invalid = false;
  for (const auto& record : report.data.at("plugins")) {
    found = found || record.at("name") == "example";
    found_path = found_path || record.at("name") == "path-only";
    found_invalid = found_invalid || record.at("name") == "Bad";
    if (record.at("name") == "example") {
      assert(record.at("shadowed").size() == 1);
    }
  }
  assert(found);
  assert(found_path);
  assert(!found_invalid);
  assert(report.status == fastflow_sdk::Status::kPass);
  if (fastflow_sdk::test::SupportsIsolatedEmptyPluginPath()) {
    fastflow_sdk::test::ClearEnvironment("FASTFLOW_SDK_PLUGIN_PATH");
    fastflow_sdk::test::ClearEnvironment("PATH");
    const std::filesystem::path previous_directory =
        std::filesystem::current_path();
    std::filesystem::current_path(root);
    fastflow_sdk::Report empty_path_report =
        fastflow_sdk::RunPluginList(invocation);
    std::filesystem::current_path(previous_directory);
    assert(empty_path_report.data.at("search_paths").size() == 1);
    for (const auto& record : empty_path_report.data.at("plugins")) {
      assert(record.at("name") != "example");
      assert(record.at("name") != "path-only");
    }
  }
  std::filesystem::remove_all(root);
  return 0;
}
