#include "rocm_npu/command.h"

#include <string>

#include "rocm_npu/hrx_loader.h"
#include "rocm_npu/platform.h"
#include "rocm_npu/product.h"

namespace rocm_npu {

Report RunVersion(const Invocation& invocation) {
  Report report = MakeReport("version");
  report.data = {
      {"os", OperatingSystemName()},
      {"architecture", ProcessArchitecture()},
      {"executable", PathToUtf8(ExecutablePath())},
      {"declared_abi",
       {
           {"xadx", 1},
           {"amdxdna_executable_run", 0},
           {"amdxdna_executable_entry_point", 0},
           {"amdxdna_executable_create_params", 0},
       }},
  };

  const auto manifest_path = FindProductManifest();
  if (manifest_path) {
    std::string error;
    auto manifest = ReadJsonFile(*manifest_path, error);
    if (manifest) {
      report.data["package_manifest"] = {
          {"path", PathToUtf8(*manifest_path)},
          {"version", manifest->value("version", "")},
      };
    } else {
      report.data["package_manifest"] = {
          {"path", PathToUtf8(*manifest_path)}, {"error", error}};
    }
  } else {
    report.data["package_manifest"] = nullptr;
  }

  HrxLoadResult load = HrxLibrary::Load(invocation.runtime_path);
  nlohmann::json candidates = nlohmann::json::array();
  for (const auto& candidate : load.candidates) {
    candidates.push_back(PathToUtf8(candidate));
  }
  report.data["runtime"] = {
      {"loadable", load.library.has_value()},
      {"path", load.library ? PathToUtf8(load.library->path()) : ""},
      {"abi_version",
       load.library
           ? nlohmann::json{
                 {"major", load.library->runtime_version_major()},
                 {"minor", load.library->runtime_version_minor()},
                 {"patch", load.library->runtime_version_patch()}}
           : nlohmann::json(nullptr)},
      {"candidates", std::move(candidates)},
      {"missing_symbols", load.missing_symbols},
      {"load_errors", load.errors},
  };
  return report;
}

}  // namespace rocm_npu
