// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm_npu/report.h"

#include <ostream>
#include <utility>

#include "rocm_npu/build_config.h"

namespace rocm_npu {

const char* StatusName(Status status) {
  switch (status) {
    case Status::kPass:
      return "pass";
    case Status::kFail:
      return "fail";
    case Status::kError:
      return "error";
  }
  return "error";
}

int ExitCode(const Report& report) {
  return report.status == Status::kPass
             ? 0
             : (report.status == Status::kFail ? 1 : 2);
}

nlohmann::json ToJson(const Report& report) {
  nlohmann::json checks = nlohmann::json::array();
  for (const Check& check : report.checks) {
    nlohmann::json value = {
        {"id", check.id},
        {"status", StatusName(check.status)},
        {"message", check.message},
    };
    if (!check.details.empty()) {
      value["details"] = check.details;
    }
    checks.push_back(std::move(value));
  }
  nlohmann::json result = {
      {"schema_version", report.schema_version},
      {"command", report.command},
      {"status", StatusName(report.status)},
      {"product",
       {
           {"version", report.product.version},
           {"source_revision", report.product.source_revision},
           {"build_platform", report.product.build_platform},
       }},
      {"checks", std::move(checks)},
  };
  if (!report.data.empty()) {
    result["data"] = report.data;
  }
  return result;
}

void RenderJson(const Report& report, std::ostream& output) {
  output << ToJson(report).dump() << '\n';
}

void RenderHuman(const Report& report, std::ostream& output) {
  output << "rocm-npu " << report.command << ": " << StatusName(report.status)
         << '\n';
  output << "  version: " << report.product.version << '\n';
  output << "  source revision: " << report.product.source_revision << '\n';
  for (const Check& check : report.checks) {
    output << "  [" << StatusName(check.status) << "] " << check.id;
    if (!check.message.empty()) {
      output << ": " << check.message;
    }
    output << '\n';
  }
  if (!report.data.empty()) {
    output << report.data.dump(2) << '\n';
  }
}

void AddCheck(Report& report, Check check) {
  if (check.status == Status::kError) {
    report.status = Status::kError;
  } else if (check.status == Status::kFail &&
             report.status == Status::kPass) {
    report.status = Status::kFail;
  }
  report.checks.push_back(std::move(check));
}

Report MakeReport(std::string command) {
  Report report;
  report.command = std::move(command);
  report.product.version = ROCM_NPU_VERSION;
  report.product.source_revision = ROCM_NPU_SOURCE_REVISION;
  report.product.build_platform = ROCM_NPU_BUILD_PLATFORM;
  return report;
}

}  // namespace rocm_npu
