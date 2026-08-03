#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "rocm_npu/command.h"

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
#ifdef _WIN32
      L"rocm-npu-plugin-\u00e9-test";
#else
      "rocm-npu-plugin-test";
#endif
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
#ifdef _WIN32
  const std::filesystem::path plugin = root / "rocm-npu-example.exe";
  _wputenv_s(L"ROCM_NPU_PLUGIN_PATH", root.c_str());
  _wputenv_s(L"PATH", L"");
#else
  const std::filesystem::path plugin = root / "rocm-npu-example";
  setenv("ROCM_NPU_PLUGIN_PATH", root.string().c_str(), 1);
  setenv("PATH", "", 1);
#endif
  std::ofstream(plugin) << "fixture";
#ifndef _WIN32
  std::filesystem::permissions(
      plugin, std::filesystem::perms::owner_exec,
      std::filesystem::perm_options::add);
#endif

  rocm_npu::Invocation invocation;
  invocation.command = "plugin";
  invocation.arguments = {"list"};
  rocm_npu::Report report = rocm_npu::RunPluginList(invocation);
  bool found = false;
  for (const auto& record : report.data.at("plugins")) {
    found = found || record.at("name") == "example";
  }
  assert(found);
  assert(report.status == rocm_npu::Status::kPass);
#ifndef _WIN32
  unsetenv("ROCM_NPU_PLUGIN_PATH");
  const std::filesystem::path previous_directory =
      std::filesystem::current_path();
  std::filesystem::current_path(root);
  rocm_npu::Report empty_path_report = rocm_npu::RunPluginList(invocation);
  std::filesystem::current_path(previous_directory);
  assert(empty_path_report.data.at("search_paths").at(1) == ".");
  assert(empty_path_report.data.at("plugins").size() == 1);
  assert(empty_path_report.data.at("plugins").at(0).at("shadowed").empty());
#endif
  std::filesystem::remove_all(root);
  return 0;
}
