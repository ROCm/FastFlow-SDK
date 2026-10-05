// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <cassert>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "fastflow_sdk/build_config.h"
#include "fastflow_sdk/command.h"
#include "fastflow_sdk/platform.h"
#include "fastflow_sdk/report.h"
#include "fastflow_sdk/sha256.h"

static_assert(NLOHMANN_JSON_VERSION_MAJOR == 3);
static_assert(NLOHMANN_JSON_VERSION_MINOR == 12);
static_assert(NLOHMANN_JSON_VERSION_PATCH == 0);

int main() {
  using fastflow_sdk::ParseInvocation;

  auto before = ParseInvocation(
      {"--json", "--runtime", "runtime", "doctor", "--verbose"});
  assert(before.invocation);
  assert(before.invocation->command == "doctor");
  assert(before.invocation->json);
  assert(before.invocation->verbose);
  assert(before.invocation->runtime_path->string() == "runtime");
  const fastflow_sdk::Report verbose_report =
      fastflow_sdk::RunCommand(*before.invocation);
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
  assert(fastflow_sdk::ParsePowerMode("PERFORMANCE") ==
         fastflow_sdk::PowerMode::kPerformance);
  assert(fastflow_sdk::PowerModeName(fastflow_sdk::PowerMode::kPowerSaver) ==
         "powersaver");
  assert(fastflow_sdk::PowerModeFromValue(4) ==
         fastflow_sdk::PowerMode::kTurbo);
  assert(!fastflow_sdk::PowerModeFromValue(5));

  fastflow_sdk::Report report = fastflow_sdk::MakeReport("doctor");
  fastflow_sdk::AddCheck(report, {"runtime.library.load",
                                  fastflow_sdk::Status::kFail,
                                  "quote: \" slash: \\\\ newline:\n",
                                  {}});
  assert(fastflow_sdk::ExitCode(report) == 1);
  const nlohmann::json value = fastflow_sdk::ToJson(report);
  assert(value.at("schema_version") == 1);
  assert(value.at("status") == "fail");
  assert(value.at("product").at("version") == FASTFLOW_SDK_VERSION);

  std::ostringstream output;
  fastflow_sdk::RenderJson(report, output);
  const nlohmann::json reparsed = nlohmann::json::parse(output.str());
  assert(reparsed == value);

  fastflow_sdk::Report error = fastflow_sdk::MakeReport("probe");
  fastflow_sdk::AddCheck(error, {"internal.error",
                                 fastflow_sdk::Status::kError,
                                 "synthetic failure",
                                 {}});
  assert(fastflow_sdk::ExitCode(error) == 2);
  assert(nlohmann::json::parse(fastflow_sdk::ToJson(error).dump()).is_object());
  assert(fastflow_sdk::Sha256Hex(std::vector<std::uint8_t>{'a', 'b', 'c'}) ==
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
  assert(fastflow_sdk::HasValidSelfTestFixtureTypes(fixture));
  assert(fastflow_sdk::HasValidSelfTestFixtureTypes(
      nlohmann::json::parse(fixture.dump())));
  nlohmann::json fixture_with_stale_file = fixture;
  fixture_with_stale_file["buffers"][0]["input"] = "input.bin";
  assert(!fastflow_sdk::HasValidSelfTestFixtureTypes(fixture_with_stale_file));
  fixture["buffers"][0]["byte_length"] = -4;
  assert(!fastflow_sdk::HasValidSelfTestFixtureTypes(fixture));
  fixture["buffers"][0]["byte_length"] = "4";
  assert(!fastflow_sdk::HasValidSelfTestFixtureTypes(fixture));
  return 0;
}
