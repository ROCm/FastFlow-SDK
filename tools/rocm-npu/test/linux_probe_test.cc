#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

#include <sys/stat.h>
#include <sys/sysmacros.h>

#include "rocm_npu/platform.h"

namespace {

void Write(const std::filesystem::path& path, const std::string& value) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream(path) << value;
}

void MarkDriver(const std::filesystem::path& root,
                const std::filesystem::path& node,
                const std::string& name) {
  const std::filesystem::path driver = root / "drivers" / name;
  std::filesystem::create_directories(driver);
  std::filesystem::create_directories(node / "device");
  std::filesystem::create_directory_symlink(driver, node / "device" / "driver");
}

}  // namespace

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "rocm-npu-linux-probe-test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  rocm_npu::ProbeOptions options;
  options.sys_module_root = root / "module";
  options.sys_accel_root = root / "class" / "accel";
  options.dev_accel_root = root / "dev" / "accel";
  options.open_devices = false;
  std::filesystem::create_directories(options.sys_accel_root);

  rocm_npu::ProbeResult empty = rocm_npu::CollectPlatformProbe(options);
  assert(empty.detected_devices == 0);
  assert(empty.usable_devices == 0);

  Write(options.sys_accel_root / "accel9" / "dev", "5:6\n");
  MarkDriver(root, options.sys_accel_root / "accel9", "other");
  rocm_npu::ProbeResult non_amd = rocm_npu::CollectPlatformProbe(options);
  assert(non_amd.detected_devices == 0);

  Write(options.sys_accel_root / "accel0" / "dev", "malformed\n");
  MarkDriver(root, options.sys_accel_root / "accel0", "amdxdna");
  rocm_npu::ProbeResult malformed = rocm_npu::CollectPlatformProbe(options);
  assert(malformed.detected_devices == 1);
  assert(malformed.usable_devices == 0);
  assert(malformed.facts.at("devices").at(0).at("access_error") ==
         "sysfs dev value is missing or malformed");

  Write(options.sys_accel_root / "accel0" / "dev", "1:2\n");
  Write(options.sys_accel_root / "accel0" / "device" / "vbnv",
        "NPU Strix\n");
  Write(options.dev_accel_root / "accel0", "not a device");
  Write(options.sys_accel_root / "accel1" / "dev", "3:4\n");
  MarkDriver(root, options.sys_accel_root / "accel1", "amdxdna");
  Write(options.sys_accel_root / "accel1" / "device" / "vbnv",
        "RyzenAI-npu2\n");
  Write(options.dev_accel_root / "accel1", "also not a device");
  rocm_npu::ProbeResult multiple = rocm_npu::CollectPlatformProbe(options);
  assert(multiple.detected_devices == 2);
  assert(multiple.usable_devices == 0);
  assert(multiple.facts.at("devices").at(0).at("access_error") ==
         "device path is not a character device");
  assert(multiple.target_family == "npu2");

  rocm_npu::ProbeOptions ioctl_options;
  ioctl_options.sys_module_root = root / "ioctl" / "module";
  ioctl_options.sys_accel_root = root / "ioctl" / "class" / "accel";
  ioctl_options.dev_accel_root = root / "ioctl" / "dev" / "accel";
  ioctl_options.open_devices = true;
  const std::filesystem::path ioctl_sys_node =
      ioctl_options.sys_accel_root / "accel0";
  MarkDriver(root / "ioctl", ioctl_sys_node, "amdxdna");
  struct stat null_metadata = {};
  assert(stat("/dev/null", &null_metadata) == 0);
  Write(ioctl_sys_node / "dev",
        std::to_string(major(null_metadata.st_rdev)) + ":" +
            std::to_string(minor(null_metadata.st_rdev)) + "\n");
  std::filesystem::create_directories(ioctl_options.dev_accel_root);
  std::filesystem::create_symlink(
      "/dev/null", ioctl_options.dev_accel_root / "accel0");
  rocm_npu::ProbeResult incomplete_ioctls =
      rocm_npu::CollectPlatformProbe(ioctl_options);
  assert(incomplete_ioctls.usable_devices == 1);
  const auto ioctl_check =
      std::find_if(incomplete_ioctls.checks.begin(),
                   incomplete_ioctls.checks.end(),
                   [](const rocm_npu::Check& check) {
                     return check.id == "device.ioctl";
                   });
  assert(ioctl_check != incomplete_ioctls.checks.end());
  assert(ioctl_check->status == rocm_npu::Status::kPass);
  assert(!ioctl_check->details.at("diagnostics_complete").get<bool>());

  std::filesystem::remove_all(root);
  return 0;
}
