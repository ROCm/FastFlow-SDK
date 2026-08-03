#include "rocm_npu/command.h"

#include <filesystem>
#include <string>

#include "rocm_npu/hrx_loader.h"
#include "rocm_npu/platform.h"
#include "rocm_npu/product.h"

namespace rocm_npu {

Report RunDoctor(const Invocation& invocation) {
  Report report = MakeReport("doctor");

  const auto manifest_path = FindProductManifest();
  if (!manifest_path) {
    AddCheck(report, {"package.manifest", Status::kFail,
                      "product manifest was not found", {}});
  } else {
    std::string error;
    auto manifest = ReadJsonFile(*manifest_path, error);
    if (!manifest) {
      AddCheck(report,
               {"package.manifest", Status::kFail, error,
                {{"path", PathToUtf8(*manifest_path)}}});
    } else {
      const std::string version = manifest->value("version", "");
      nlohmann::json inventory_details;
      std::string inventory_error;
      const bool development_manifest =
          !manifest->contains("files") &&
          manifest_path->parent_path().filename() == "rocm-npu" &&
          manifest_path->parent_path().parent_path().filename() == "share" &&
          manifest->value("product", "") == "rocm-npu native runtime CLI";
      const bool inventory_matches =
          development_manifest ||
          VerifyProductManifestFiles(*manifest_path, *manifest,
                                     inventory_details, inventory_error);
      if (development_manifest) {
        inventory_details = {{"declared", false}};
      }
      const bool version_matches = version == report.product.version;
      const bool matches = version_matches && inventory_matches;
      AddCheck(
          report,
          {"package.manifest", matches ? Status::kPass : Status::kFail,
           matches
               ? (development_manifest
                      ? "development manifest version matches; no packaged "
                        "file inventory is declared"
                      : "manifest version and declared file inventory match "
                        "the product")
               : (!version_matches
                      ? "manifest version does not match the native executable"
                      : inventory_error),
           {{"path", PathToUtf8(*manifest_path)},
            {"manifest_version", version},
            {"executable_version", report.product.version},
            {"inventory", std::move(inventory_details)}}});
    }
  }

  ProbeResult platform = CollectPlatformProbe();
  report.data["probe"] = std::move(platform.facts);
  for (const Check& check : platform.checks) {
    AddCheck(report, check);
  }

  HrxLoadResult load = HrxLibrary::Load(invocation.runtime_path);
  bool candidate_exists = false;
  for (const auto& candidate : load.candidates) {
    std::error_code error;
    candidate_exists |= std::filesystem::is_regular_file(candidate, error);
  }
  AddCheck(report,
           {"runtime.library.present",
            (candidate_exists || load.library) ? Status::kPass : Status::kFail,
            (candidate_exists || load.library)
                ? "an HRX runtime candidate is present"
                : "no HRX runtime candidate is present",
            {}});
  AddCheck(report,
           {"runtime.library.load",
            load.library ? Status::kPass : Status::kFail,
            load.library ? "HRX runtime loaded"
                         : "HRX runtime could not be loaded",
            {{"errors", load.errors}}});
  AddCheck(report,
           {"runtime.symbols",
            load.library && load.missing_symbols.empty() ? Status::kPass
                                                         : Status::kFail,
            load.library && load.missing_symbols.empty()
                ? "all required public HRX symbols resolved"
                : (load.missing_symbols.empty()
                       ? "symbols could not be resolved because HRX did not load"
                       : "required public HRX symbols are missing"),
            {{"missing", load.missing_symbols}}});
  AddCheck(
      report,
      {"runtime.abi",
       load.library ? Status::kPass : Status::kFail,
       load.library ? "HRX runtime ABI is compatible"
                    : "a compatible HRX runtime ABI could not be established",
       {{"required",
         {{"major", HRX_VERSION_MAJOR}, {"minor", HRX_VERSION_MINOR}}},
        {"actual",
         load.library
             ? nlohmann::json{
                   {"major", load.library->runtime_version_major()},
                   {"minor", load.library->runtime_version_minor()},
                   {"patch", load.library->runtime_version_patch()}}
             : nlohmann::json(nullptr)}}});
  if (!load.library) {
    return report;
  }

  HrxLibrary& library = *load.library;
  const HrxFunctions& functions = library.functions();
  HrxStatus initialize = functions.gpu_initialize(0);
  const std::string initialize_error =
      library.ConsumeStatus(initialize, "hrx_gpu_initialize");
  const bool initialized = initialize_error.empty();
  AddCheck(report,
           {"runtime.initialize",
            initialized ? Status::kPass : Status::kFail,
            initialized ? "HRX initialized" : initialize_error, {}});
  if (!initialized) {
    return report;
  }

  int count = 0;
  HrxStatus count_status = functions.gpu_device_count(&count);
  const std::string count_error =
      library.ConsumeStatus(count_status, "hrx_gpu_device_count");
  AddCheck(report,
           {"runtime.device_count",
            count_error.empty() && count > 0 ? Status::kPass : Status::kFail,
            !count_error.empty()
                ? count_error
                : (count > 0 ? "HRX reports one or more devices"
                             : "HRX reports zero devices"),
            {{"count", count}}});

  HrxStatus shutdown = functions.gpu_shutdown();
  const std::string shutdown_error =
      library.ConsumeStatus(shutdown, "hrx_gpu_shutdown");
  AddCheck(report,
           {"runtime.shutdown",
            shutdown_error.empty() ? Status::kPass : Status::kFail,
            shutdown_error.empty() ? "HRX shut down cleanly" : shutdown_error,
            {}});
  return report;
}

}  // namespace rocm_npu
