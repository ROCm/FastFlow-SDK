// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <cassert>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rocm_npu/build_config.h"
#include "rocm_npu/command.h"
#include "rocm_npu/platform.h"
#include "rocm_npu/report.h"
#include "rocm_npu/sha256.h"

static_assert(NLOHMANN_JSON_VERSION_MAJOR == 3);
static_assert(NLOHMANN_JSON_VERSION_MINOR == 12);
static_assert(NLOHMANN_JSON_VERSION_PATCH == 0);

int main() {
  using rocm_npu::ParseInvocation;

  auto before = ParseInvocation(
      {"--json", "--runtime", "runtime", "doctor", "--verbose"});
  assert(before.invocation);
  assert(before.invocation->command == "doctor");
  assert(before.invocation->json);
  assert(before.invocation->verbose);
  assert(before.invocation->runtime_path->string() == "runtime");
  const rocm_npu::Report verbose_report =
      rocm_npu::RunCommand(*before.invocation);
  assert(verbose_report.data.contains("diagnostics"));

  auto after = ParseInvocation({"doctor", "--json", "--runtime=runtime"});
  assert(after.invocation);
  assert(after.invocation->json);
  assert(after.invocation->runtime_path->string() == "runtime");

  assert(!ParseInvocation({"--runtime"}).invocation);
  const auto external_before =
      ParseInvocation({"--json", "qualify", "--profile", "models"});
  const auto external_after =
      ParseInvocation({"qualify", "--profile", "models", "--json"});
  assert(external_before.invocation && external_after.invocation);
  assert(external_before.invocation->json && external_after.invocation->json);
  assert(external_before.invocation->arguments ==
         external_after.invocation->arguments);
  assert(external_before.invocation->arguments ==
         std::vector<std::string>({"--profile", "models"}));
  assert(ParseInvocation({"qualify", "--help"}).invocation->arguments ==
         std::vector<std::string>({"--help"}));
  assert(!ParseInvocation({"Uppercase"}).invocation);
  assert(!ParseInvocation({"../qualify"}).invocation);
  assert(!ParseInvocation({"probe", "unexpected"}).invocation);
  assert(ParseInvocation({"plugin", "list"}).invocation);
  assert(!ParseInvocation({"plugin"}).invocation);
  assert(!ParseInvocation({"plugin", "run"}).invocation);
  assert(ParseInvocation({"powermode"}).invocation);
  assert(ParseInvocation({"powermode", "performance"}).invocation);
  assert(!ParseInvocation({"powermode", "invalid"}).invocation);
  assert(!ParseInvocation({"powermode", "performance", "turbo"}).invocation);
  assert(rocm_npu::ParsePowerMode("PERFORMANCE") ==
         rocm_npu::PowerMode::kPerformance);
  assert(rocm_npu::PowerModeName(rocm_npu::PowerMode::kPowerSaver) ==
         "powersaver");
  assert(rocm_npu::PowerModeFromValue(4) == rocm_npu::PowerMode::kTurbo);
  assert(!rocm_npu::PowerModeFromValue(5));

  rocm_npu::Report report = rocm_npu::MakeReport("doctor");
  rocm_npu::AddCheck(report, {"runtime.library.load",
                              rocm_npu::Status::kFail,
                              "quote: \" slash: \\\\ newline:\n",
                              {}});
  assert(rocm_npu::ExitCode(report) == 1);
  const nlohmann::json value = rocm_npu::ToJson(report);
  assert(value.at("schema_version") == 1);
  assert(value.at("status") == "fail");
  assert(value.at("product").at("version") == ROCM_NPU_VERSION);

  std::ostringstream output;
  rocm_npu::RenderJson(report, output);
  const nlohmann::json reparsed = nlohmann::json::parse(output.str());
  assert(reparsed == value);

  rocm_npu::Report error = rocm_npu::MakeReport("probe");
  rocm_npu::AddCheck(
      error,
      {"internal.error", rocm_npu::Status::kError, "synthetic failure", {}});
  assert(rocm_npu::ExitCode(error) == 2);
  assert(nlohmann::json::parse(rocm_npu::ToJson(error).dump()).is_object());
  assert(rocm_npu::Sha256Hex(std::vector<std::uint8_t>{'a', 'b', 'c'}) ==
         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  nlohmann::json fixture = {
      {"schema_version", 1U},
      {"xadx_version", 1U},
      {"name", "test"},
      {"target", "npu2"},
      {"export", "entry"},
      {"producer",
       {{"artifacts",
         {{"xadx_sha256", "x"},
          {"input_sha256", "i"},
          {"expected_sha256", "o"}}}}},
      {"dispatch",
       {{"workgroup_count", {1U, 1U, 1U}},
        {"workgroup_size", {1U, 1U, 1U}},
        {"subgroup_size", 0U},
        {"constant_byte_length", 0U},
        {"export_binding_count", nullptr}}},
      {"buffers",
       {{{"name", "input"},
         {"direction", "input"},
         {"dtype", "int32"},
         {"shape", {1U}},
         {"byte_length", 4U},
         {"generator", {{"kind", "iota_i32"}, {"start", 1}}}},
        {{"name", "output"},
         {"direction", "output"},
         {"dtype", "int32"},
         {"shape", {1U}},
         {"byte_length", 4U},
         {"generator",
          {{"kind", "add_i32_scalar"}, {"source", "input"}, {"scalar", 1}}}}}}};
  assert(rocm_npu::HasValidSelfTestFixtureTypes(fixture));
  assert(rocm_npu::HasValidSelfTestFixtureTypes(
      nlohmann::json::parse(fixture.dump())));
  nlohmann::json fixture_with_stale_file = fixture;
  fixture_with_stale_file["buffers"][0]["input"] = "input.bin";
  assert(!rocm_npu::HasValidSelfTestFixtureTypes(fixture_with_stale_file));
  fixture["buffers"][0]["byte_length"] = -4;
  assert(!rocm_npu::HasValidSelfTestFixtureTypes(fixture));
  fixture["buffers"][0]["byte_length"] = "4";
  assert(!rocm_npu::HasValidSelfTestFixtureTypes(fixture));
  return 0;
}
