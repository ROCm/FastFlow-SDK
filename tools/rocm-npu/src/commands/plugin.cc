#include "rocm_npu/command.h"

#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "rocm_npu/platform.h"
#include "platform/plugin.h"

namespace rocm_npu {
namespace {

using NativeString = std::filesystem::path::string_type;

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
    } else if (PluginEmptyPathEntryMeansCurrentDirectory()) {
      // Empty POSIX PATH components denote the current working directory.
      paths.emplace_back(".");
    }
    if (end == NativeString::npos) {
      break;
    }
    begin = end + 1;
  }
}

}  // namespace

Report RunPluginList(const Invocation&) {
  Report report = MakeReport("plugin list");
  std::vector<std::filesystem::path> search_paths;
  if (auto explicit_paths = PluginSearchPathOverride()) {
    AddSearchPaths(search_paths, *explicit_paths);
  }
  search_paths.push_back(ExecutablePath().parent_path());
  if (auto system_paths = SystemExecutableSearchPath()) {
    AddSearchPaths(search_paths, *system_paths);
  }

  std::set<std::string> visited_directories;
  std::map<std::string, nlohmann::json> plugins;
  nlohmann::json rendered_paths = nlohmann::json::array();
  for (const auto& directory : search_paths) {
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
    std::string key =
        PluginDirectoryKey(PathToUtf8(normalized.lexically_normal()));
    if (!visited_directories.insert(key).second) {
      continue;
    }
    rendered_paths.push_back(PathToUtf8(directory));
    std::error_code error;
    std::filesystem::directory_iterator iterator(directory, error);
    if (error) {
      continue;
    }
    std::vector<std::filesystem::path> entries;
    for (const auto& entry : iterator) {
      entries.push_back(entry.path());
    }
    std::sort(entries.begin(), entries.end(),
              [](const std::filesystem::path& left,
                 const std::filesystem::path& right) {
                return PluginEntryLess(left, right);
              });
    for (const auto& entry : entries) {
      auto name = PluginExecutableName(entry);
      if (!name) {
        continue;
      }
      auto found = plugins.find(*name);
      if (found == plugins.end()) {
        plugins.emplace(
            *name, nlohmann::json{{"name", *name},
                                  {"path", PathToUtf8(entry)},
                                  {"shadowed", nlohmann::json::array()}});
      } else {
        found->second["shadowed"].push_back(PathToUtf8(entry));
      }
    }
  }

  nlohmann::json discovered = nlohmann::json::array();
  for (auto& [name, plugin] : plugins) {
    (void)name;
    discovered.push_back(std::move(plugin));
  }
  report.data = {
      {"plugins", std::move(discovered)},
      {"search_paths", std::move(rendered_paths)},
  };
  AddCheck(report,
           {"plugin.discovery", Status::kPass,
            std::to_string(plugins.size()) + " external plugin(s) discovered",
            {{"count", plugins.size()}}});
  return report;
}

}  // namespace rocm_npu
