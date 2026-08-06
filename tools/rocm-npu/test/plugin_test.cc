// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

#include "platform.h"
#include "rocm_npu/command.h"

int main() {
  const std::filesystem::path root = rocm_npu::test::TestRoot("plugin");
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const std::filesystem::path system_root = root / "system";
  std::filesystem::create_directories(system_root);
  const std::filesystem::path plugin =
      rocm_npu::test::PluginExecutable(root, "example");
  const std::filesystem::path shadowed =
      rocm_npu::test::PluginExecutable(system_root, "example");
  const std::filesystem::path path_plugin =
      rocm_npu::test::PluginExecutable(system_root, "path-only");
  const std::filesystem::path invalid =
      rocm_npu::test::PluginExecutable(root, "Bad");
  rocm_npu::test::SetEnvironmentPath("ROCM_NPU_PLUGIN_PATH", root);
  rocm_npu::test::SetEnvironmentPath("PATH", system_root);
  for (const auto& path : {plugin, shadowed, path_plugin, invalid}) {
    std::ofstream(path) << "fixture";
    rocm_npu::test::MakeExecutable(path);
  }

  rocm_npu::Invocation invocation;
  invocation.command = "plugin";
  invocation.arguments = {"list"};
  rocm_npu::Report report = rocm_npu::RunPluginList(invocation);
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
  assert(report.status == rocm_npu::Status::kPass);
  if (rocm_npu::test::SupportsIsolatedEmptyPluginPath()) {
    rocm_npu::test::ClearEnvironment("ROCM_NPU_PLUGIN_PATH");
    rocm_npu::test::ClearEnvironment("PATH");
    const std::filesystem::path previous_directory =
        std::filesystem::current_path();
    std::filesystem::current_path(root);
    rocm_npu::Report empty_path_report = rocm_npu::RunPluginList(invocation);
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
