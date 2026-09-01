// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm_npu/command.h"

#include <algorithm>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "rocm_npu/platform.h"
#include "rocm_npu/hrx_loader.h"
#include "rocm_npu/build_config.h"
#include "platform/plugin.h"

namespace rocm_npu {
namespace {

using NativeString = std::filesystem::path::string_type;

struct DiscoveredPlugin {
  std::filesystem::path path;
  std::vector<std::filesystem::path> shadowed;
};

struct PluginDiscovery {
  std::map<std::string, DiscoveredPlugin> plugins;
  std::vector<std::filesystem::path> search_paths;
};

void AddSearchPaths(std::vector<std::filesystem::path>& paths,
                    const NativeString& value) {
  const auto separator = PluginPathListSeparator();
  const auto quote = PluginPathListQuote();
  std::size_t begin = 0;
  while (begin <= value.size()) {
    std::size_t end = value.find(separator, begin);
    NativeString part = value.substr(
        begin, end == NativeString::npos ? NativeString::npos : end - begin);
    if (part.size() >= 2 && part.front() == quote && part.back() == quote) {
      part = part.substr(1, part.size() - 2);
    }
    if (!part.empty()) {
      paths.emplace_back(std::move(part));
    }
    if (end == NativeString::npos) {
      break;
    }
    begin = end + 1;
  }
}

PluginDiscovery DiscoverPlugins() {
  PluginDiscovery result;
  if (auto explicit_paths = PluginSearchPathOverride()) {
    AddSearchPaths(result.search_paths, *explicit_paths);
  }
  result.search_paths.push_back(ExecutablePath().parent_path());
  if (auto system_paths = SystemExecutableSearchPath()) {
    AddSearchPaths(result.search_paths, *system_paths);
  }

  std::set<std::string> visited_directories;
  std::vector<std::filesystem::path> unique_paths;
  for (const auto& directory : result.search_paths) {
    std::error_code canonical_error;
    std::filesystem::path normalized =
        std::filesystem::weakly_canonical(directory, canonical_error);
    if (canonical_error) {
      canonical_error.clear();
      normalized = std::filesystem::absolute(directory, canonical_error);
    }
    if (canonical_error) {
      normalized = directory;
    }
    const std::string key =
        PluginDirectoryKey(PathToUtf8(normalized.lexically_normal()));
    if (!visited_directories.insert(key).second) {
      continue;
    }
    unique_paths.push_back(directory);
    std::error_code error;
    std::filesystem::directory_iterator iterator(directory, error);
    if (error) {
      continue;
    }
    std::vector<std::filesystem::path> entries;
    const std::filesystem::directory_iterator end;
    while (!error && iterator != end) {
      entries.push_back(iterator->path());
      iterator.increment(error);
    }
    std::sort(entries.begin(), entries.end(),
              [](const std::filesystem::path& left,
                 const std::filesystem::path& right) {
                return PluginEntryLess(left, right);
              });
    for (const auto& entry : entries) {
      auto name = PluginExecutableName(entry);
      if (!name || !IsValidExternalCommandName(*name)) {
        continue;
      }
      std::error_code resolve_error;
      std::filesystem::path resolved =
          std::filesystem::weakly_canonical(entry, resolve_error);
      if (resolve_error) {
        resolve_error.clear();
        resolved = std::filesystem::absolute(entry, resolve_error);
      }
      if (resolve_error) {
        resolved = entry;
      }
      auto [found, inserted] =
          result.plugins.emplace(*name, DiscoveredPlugin{resolved, {}});
      if (!inserted) {
        found->second.shadowed.push_back(resolved);
      }
    }
  }
  result.search_paths = std::move(unique_paths);
  return result;
}

struct RuntimeResolution {
  std::optional<std::filesystem::path> path;
  std::string error;
};

RuntimeResolution ResolveRuntimePath(const Invocation& invocation) {
  if (invocation.runtime_path) {
    HrxLoadResult loaded = HrxLibrary::Load(invocation.runtime_path);
    if (!loaded.library) {
      std::string detail = loaded.errors.empty()
                               ? "no compatible HRX library was found"
                               : loaded.errors.front();
      return {std::nullopt,
              "explicit HRX runtime could not be loaded: " + detail};
    }
    std::error_code error;
    std::filesystem::path resolved =
        std::filesystem::weakly_canonical(loaded.library->path(), error);
    return {error ? std::optional<std::filesystem::path>(loaded.library->path())
                  : std::optional<std::filesystem::path>(resolved),
            {}};
  }
  for (const auto& candidate :
       HrxLibrary::Candidates(invocation.runtime_path)) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(candidate, error)) {
      continue;
    }
    std::filesystem::path resolved =
        std::filesystem::weakly_canonical(candidate, error);
    return {error ? std::optional<std::filesystem::path>(candidate)
                  : std::optional<std::filesystem::path>(resolved),
            {}};
  }
  return {};
}

}  // namespace

Report RunPluginList(const Invocation&) {
  Report report = MakeReport("plugin list");
  PluginDiscovery discovery = DiscoverPlugins();
  nlohmann::json rendered_paths = nlohmann::json::array();
  for (const auto& directory : discovery.search_paths) {
    rendered_paths.push_back(PathToUtf8(directory));
  }

  nlohmann::json discovered = nlohmann::json::array();
  for (const auto& [name, plugin] : discovery.plugins) {
    nlohmann::json shadowed = nlohmann::json::array();
    for (const auto& path : plugin.shadowed) {
      shadowed.push_back(PathToUtf8(path));
    }
    discovered.push_back({{"name", name},
                          {"path", PathToUtf8(plugin.path)},
                          {"shadowed", std::move(shadowed)}});
  }
  report.data = {
      {"plugins", std::move(discovered)},
      {"search_paths", std::move(rendered_paths)},
  };
  AddCheck(report, {"plugin.discovery",
                    Status::kPass,
                    std::to_string(discovery.plugins.size()) +
                        " external plugin(s) discovered",
                    {{"count", discovery.plugins.size()}}});
  return report;
}

ExternalCommandResult RunExternalCommand(const Invocation& invocation) {
  PluginDiscovery discovery = DiscoverPlugins();
  const auto plugin = discovery.plugins.find(invocation.command);
  if (plugin == discovery.plugins.end()) {
    return {2, "external command not found: " + invocation.command};
  }

  std::error_code error;
  const std::filesystem::path core =
      std::filesystem::weakly_canonical(ExecutablePath(), error);
  error.clear();
  const std::filesystem::path executable =
      std::filesystem::weakly_canonical(plugin->second.path, error);
  error.clear();
  if ((!core.empty() && !executable.empty() && core == executable) ||
      std::filesystem::equivalent(ExecutablePath(), plugin->second.path,
                                  error)) {
    return {2, "refusing to execute the core executable as a plugin"};
  }

  std::map<std::string, std::string> environment = {
      {"ROCM_NPU_PLUGIN_API", "1"},
      {"ROCM_NPU_CORE_VERSION", ROCM_NPU_VERSION},
      {"ROCM_NPU_CORE_EXECUTABLE", PathToUtf8(core)},
      {"ROCM_NPU_OUTPUT_FORMAT", invocation.json ? "json" : "human"},
      {"ROCM_NPU_RUNTIME_PATH", ""},
  };
  RuntimeResolution runtime = ResolveRuntimePath(invocation);
  if (!runtime.error.empty()) {
    return {2, std::move(runtime.error)};
  }
  if (runtime.path) {
    environment["ROCM_NPU_RUNTIME_PATH"] = PathToUtf8(*runtime.path);
  }
  std::string launch_error;
  const int exit_code = ExecutePluginProcess(
      executable.empty() ? plugin->second.path : executable,
      invocation.arguments, environment, launch_error);
  return {exit_code, std::move(launch_error)};
}

}  // namespace rocm_npu
