#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace rocm_npu {

enum class Status { kPass, kFail, kError };

struct Check {
  std::string id;
  Status status = Status::kPass;
  std::string message;
  nlohmann::json details = nlohmann::json::object();
};

struct ProductInfo {
  std::string version;
  std::string source_revision;
  std::string build_platform;
};

struct Report {
  int schema_version = 1;
  std::string command;
  Status status = Status::kPass;
  ProductInfo product;
  std::vector<Check> checks;
  nlohmann::json data = nlohmann::json::object();
};

const char* StatusName(Status status);
int ExitCode(const Report& report);
nlohmann::json ToJson(const Report& report);
void RenderJson(const Report& report, std::ostream& output);
void RenderHuman(const Report& report, std::ostream& output);
void AddCheck(Report& report, Check check);
Report MakeReport(std::string command);

}  // namespace rocm_npu
