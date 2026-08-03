#include <cassert>

#include <nlohmann/json.hpp>

#include "rocm_npu/platform.h"

int main() {
  rocm_npu::ProbeResult empty =
      rocm_npu::EvaluateWindowsDeviceRecords(nlohmann::json::array());
  assert(empty.detected_devices == 0);
  assert(empty.usable_devices == 0);

  nlohmann::json records = nlohmann::json::array({
      {
          {"device_name", "NPU Compute Accelerator Device"},
          {"device_id", "PCI\\VEN_1022&DEV_17F0"},
          {"healthy", false},
          {"driver_path", ""},
          {"firmware_candidates", nlohmann::json::array()},
      },
      {
          {"device_name", "\u52a0\u901f\u5668"},
          {"device_id", "PCI\\VEN_1022&DEV_17F0&SUBSYS_00000000"},
          {"healthy", true},
          {"driver_path", "C:\\\u9a71\u52a8\\ipustack.sys"},
          {"firmware_candidates",
           nlohmann::json::array({{{"name", "ipu_pasid.sbin"}}})},
      },
  });
  rocm_npu::ProbeResult multiple =
      rocm_npu::EvaluateWindowsDeviceRecords(records);
  assert(multiple.detected_devices == 2);
  assert(multiple.usable_devices == 1);
  assert(multiple.target_family == "npu2");
  assert(multiple.facts.at("devices").at(1).at("device_name") ==
         "\u52a0\u901f\u5668");
  assert(multiple.checks.size() == 2);
  assert(multiple.checks.at(0).status == rocm_npu::Status::kPass);
  assert(multiple.checks.at(1).status == rocm_npu::Status::kPass);
  return 0;
}
