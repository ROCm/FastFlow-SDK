// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm_npu/platform.h"

#include "platform/hrx_library.h"
#include "platform/plugin.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/wait.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <system_error>
#include <vector>

extern char** environ;

namespace rocm_npu {
namespace {

volatile sig_atomic_t g_plugin_child = -1;

void ForwardPluginSignal(int signal_number) {
  if (g_plugin_child > 0) {
    if (kill(-g_plugin_child, signal_number) != 0) {
      (void)kill(g_plugin_child, signal_number);
    }
  }
}

struct DrmVersion {
  int version_major;
  int version_minor;
  int version_patchlevel;
  std::size_t name_len;
  char* name;
  std::size_t date_len;
  char* date;
  std::size_t desc_len;
  char* desc;
};

struct AmdxdnaGetInfo {
  std::uint32_t param;
  std::uint32_t buffer_size;
  std::uint64_t buffer;
};

struct FirmwareVersion {
  std::uint32_t major;
  std::uint32_t minor;
  std::uint32_t patch;
  std::uint32_t build;
};

struct AieVersion {
  std::uint32_t major;
  std::uint32_t minor;
};

struct AieTileMetadata {
  std::uint16_t row_count;
  std::uint16_t row_start;
  std::uint16_t dma_channel_count;
  std::uint16_t lock_count;
  std::uint16_t event_reg_count;
  std::uint16_t padding[3];
};

struct AieMetadata {
  std::uint32_t column_size;
  std::uint16_t columns;
  std::uint16_t rows;
  AieVersion version;
  AieTileMetadata core;
  AieTileMetadata memory;
  AieTileMetadata shim;
};

constexpr unsigned long kDrmIoctlVersion =
    _IOWR('d', 0x00, DrmVersion);
constexpr unsigned long kAmdxdnaGetInfo =
    _IOWR('d', 0x40 + 7, AmdxdnaGetInfo);
constexpr std::uint32_t kQueryAieMetadata = 1;
constexpr std::uint32_t kQueryFirmwareVersion = 8;

std::optional<std::string> ReadText(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    return std::nullopt;
  }
  std::ostringstream content;
  content << input.rdbuf();
  std::string value = content.str();
  while (!value.empty() &&
         (value.back() == '\n' || value.back() == '\r' ||
          value.back() == ' ' || value.back() == '\t')) {
    value.pop_back();
  }
  return value.empty() ? std::nullopt
                       : std::optional<std::string>(std::move(value));
}

std::vector<std::filesystem::path> SortedDirectories(
    const std::filesystem::path& root, const std::string& prefix) {
  std::vector<std::filesystem::path> result;
  std::error_code error;
  for (std::filesystem::directory_iterator iterator(root, error), end;
       !error && iterator != end; iterator.increment(error)) {
    if (PathToUtf8(iterator->path().filename()).rfind(prefix, 0) == 0) {
      result.push_back(iterator->path());
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

template <typename Value>
std::optional<std::string> QueryInfo(int descriptor, std::uint32_t parameter,
                                     Value& value) {
  AmdxdnaGetInfo query = {
      parameter,
      static_cast<std::uint32_t>(sizeof(value)),
      reinterpret_cast<std::uint64_t>(&value),
  };
  if (ioctl(descriptor, kAmdxdnaGetInfo, &query) == 0) {
    return std::nullopt;
  }
  return std::strerror(errno);
}

nlohmann::json QueryIoctls(int descriptor) {
  nlohmann::json result = nlohmann::json::object();
  DrmVersion version = {};
  if (ioctl(descriptor, kDrmIoctlVersion, &version) == 0) {
    result["drm_version"] = {
        {"major", version.version_major},
        {"minor", version.version_minor},
        {"patch", version.version_patchlevel},
    };
  } else {
    result["drm_version_error"] = std::strerror(errno);
  }

  FirmwareVersion firmware = {};
  if (auto error =
          QueryInfo(descriptor, kQueryFirmwareVersion, firmware)) {
    result["firmware_version_error"] = *error;
  } else {
    result["firmware_version"] = {
        {"major", firmware.major},
        {"minor", firmware.minor},
        {"patch", firmware.patch},
        {"build", firmware.build},
    };
  }

  AieMetadata metadata = {};
  if (auto error = QueryInfo(descriptor, kQueryAieMetadata, metadata)) {
    result["aie_metadata_error"] = *error;
  } else {
    result["aie_metadata"] = {
        {"columns", metadata.columns},
        {"rows", metadata.rows},
        {"column_size", metadata.column_size},
        {"version",
         {{"major", metadata.version.major}, {"minor", metadata.version.minor}}},
    };
  }
  return result;
}

std::string InferTarget(const nlohmann::json& device) {
  const std::string vbnv = device.value("vbnv", "");
  std::string lower = vbnv;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  std::string pci_device = device.value("device", "");
  std::transform(pci_device.begin(), pci_device.end(), pci_device.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  if (lower.find("npu2") != std::string::npos ||
      lower.find("strix") != std::string::npos ||
      pci_device.find("17f0") != std::string::npos) {
    return "npu2";
  }
  if (lower.find("npu1") != std::string::npos ||
      lower.find("phoenix") != std::string::npos ||
      pci_device.find("1502") != std::string::npos) {
    return "npu";
  }
  return {};
}

std::optional<std::filesystem::path::string_type> NativeEnvironmentVariable(
    const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return std::nullopt;
  }
  return std::string(value);
}

}  // namespace

std::filesystem::path ExecutablePath() {
  std::vector<char> buffer(4096);
  const ssize_t length =
      readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length < 0) {
    return {};
  }
  buffer[static_cast<std::size_t>(length)] = '\0';
  return std::filesystem::path(buffer.data());
}

std::optional<std::filesystem::path::string_type> PluginSearchPathOverride() {
  return NativeEnvironmentVariable("ROCM_NPU_PLUGIN_PATH");
}

std::optional<std::filesystem::path::string_type> SystemExecutableSearchPath() {
  return NativeEnvironmentVariable("PATH");
}

std::filesystem::path::value_type PluginPathListSeparator() { return ':'; }

std::filesystem::path::value_type PluginPathListQuote() { return '"'; }

std::string PluginDirectoryKey(std::string value) { return value; }

std::optional<std::string> PluginExecutableName(
    const std::filesystem::path& path) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error) ||
      ::access(path.c_str(), X_OK) != 0) {
    return std::nullopt;
  }
  const std::string filename = PathToUtf8(path.filename());
  constexpr const char* prefix = "rocm-npu-";
  if (filename.rfind(prefix, 0) != 0) {
    return std::nullopt;
  }
  std::string name = filename.substr(std::char_traits<char>::length(prefix));
  if (name.empty()) {
    return std::nullopt;
  }
  return name;
}

bool PluginEntryLess(const std::filesystem::path& left,
                     const std::filesystem::path& right) {
  return left.filename() < right.filename();
}

int ExecutePluginProcess(
    const std::filesystem::path& executable,
    const std::vector<std::string>& arguments,
    const std::map<std::string, std::string>& environment_overrides,
    std::string& error) {
  std::vector<std::string> argument_storage;
  argument_storage.reserve(arguments.size() + 1);
  argument_storage.push_back(PathToUtf8(executable));
  argument_storage.insert(argument_storage.end(), arguments.begin(),
                          arguments.end());
  std::vector<char*> native_arguments;
  native_arguments.reserve(argument_storage.size() + 1);
  for (std::string& argument : argument_storage) {
    native_arguments.push_back(argument.data());
  }
  native_arguments.push_back(nullptr);

  std::vector<std::string> environment_storage;
  for (char** entry = ::environ; entry != nullptr && *entry != nullptr;
       ++entry) {
    const std::string value(*entry);
    const std::size_t separator = value.find('=');
    const std::string name = value.substr(0, separator);
    if (environment_overrides.count(name) == 0) {
      environment_storage.push_back(value);
    }
  }
  for (const auto& [name, value] : environment_overrides) {
    if (!value.empty()) {
      environment_storage.push_back(name + "=" + value);
    }
  }
  std::vector<char*> native_environment;
  native_environment.reserve(environment_storage.size() + 1);
  for (std::string& value : environment_storage) {
    native_environment.push_back(value.data());
  }
  native_environment.push_back(nullptr);

  int error_pipe[2] = {-1, -1};
  if (pipe2(error_pipe, O_CLOEXEC) != 0) {
    error = std::strerror(errno);
    return 2;
  }
  sigset_t launch_signals = {};
  sigset_t previous_mask = {};
  sigemptyset(&launch_signals);
  sigaddset(&launch_signals, SIGINT);
  sigaddset(&launch_signals, SIGTERM);
  if (sigprocmask(SIG_BLOCK, &launch_signals, &previous_mask) != 0) {
    error = std::strerror(errno);
    close(error_pipe[0]);
    close(error_pipe[1]);
    return 2;
  }
  const pid_t child = fork();
  if (child < 0) {
    error = std::strerror(errno);
    close(error_pipe[0]);
    close(error_pipe[1]);
    sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
    return 2;
  }
  if (child == 0) {
    close(error_pipe[0]);
    if (setpgid(0, 0) != 0) {
      const int process_group_error = errno;
      // glibc marks write() warn_unused_result under _FORTIFY_SOURCE and GCC
      // does not accept a (void) cast as acknowledgement. The child has no
      // recovery available before _exit, so bind the result and drop it.
      [[maybe_unused]] const ssize_t group_error_written = write(
          error_pipe[1], &process_group_error, sizeof(process_group_error));
      _exit(127);
    }
    sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
    execve(executable.c_str(), native_arguments.data(),
           native_environment.data());
    const int execute_error = errno;
    [[maybe_unused]] const ssize_t execute_error_written =
        write(error_pipe[1], &execute_error, sizeof(execute_error));
    _exit(127);
  }

  close(error_pipe[1]);
  (void)setpgid(child, child);
  g_plugin_child = child;
  struct sigaction forward = {};
  forward.sa_handler = ForwardPluginSignal;
  sigemptyset(&forward.sa_mask);
  struct sigaction previous_interrupt = {};
  struct sigaction previous_terminate = {};
  sigaction(SIGINT, &forward, &previous_interrupt);
  sigaction(SIGTERM, &forward, &previous_terminate);
  sigprocmask(SIG_SETMASK, &previous_mask, nullptr);

  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) {
      error = std::strerror(errno);
      status = -1;
      break;
    }
  }
  sigprocmask(SIG_BLOCK, &launch_signals, nullptr);
  g_plugin_child = -1;
  sigaction(SIGINT, &previous_interrupt, nullptr);
  sigaction(SIGTERM, &previous_terminate, nullptr);
  sigprocmask(SIG_SETMASK, &previous_mask, nullptr);

  int execute_error = 0;
  const ssize_t error_size =
      read(error_pipe[0], &execute_error, sizeof(execute_error));
  close(error_pipe[0]);
  if (error_size == sizeof(execute_error)) {
    error = std::strerror(execute_error);
    return 2;
  }
  if (status < 0) {
    return 2;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 2;
}

const char* HrxLibraryFilename() { return "libhrx.so.0"; }

std::optional<std::filesystem::path> HrxEnvironmentPath() {
  auto value = NativeEnvironmentVariable("HRX_LIBHRX");
  if (!value || value->empty()) {
    return std::nullopt;
  }
  return std::filesystem::path(*value);
}

std::vector<std::filesystem::path> PackagedHrxLibraryCandidates(
    const std::filesystem::path& executable_directory) {
  return {
      executable_directory / ".." / "lib" / HrxLibraryFilename(),
      executable_directory / HrxLibraryFilename(),
  };
}

std::vector<std::filesystem::path> HrxLibraryFallbackCandidates() {
  return {HrxLibraryFilename(), "libhrx.so"};
}

void* OpenHrxLibrary(const std::filesystem::path& path, std::string& error) {
  dlerror();
  // HRX and its backend plugins share process-global runtime symbols.
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    const char* detail = dlerror();
    error = detail == nullptr ? "unknown dynamic loader error" : detail;
  }
  return handle;
}

void CloseHrxLibrary(void* handle) { dlclose(handle); }

void* LookupHrxSymbol(void* handle, const char* name) {
  dlerror();
  return dlsym(handle, name);
}

std::string OperatingSystemName() {
  utsname information = {};
  if (uname(&information) != 0) {
    return "Linux";
  }
  return std::string(information.sysname) + " " + information.release;
}

std::string ProcessArchitecture() {
#if defined(__x86_64__)
  return "x86_64";
#elif defined(__aarch64__)
  return "aarch64";
#else
  return "unknown";
#endif
}

bool IsAmdXdnaNode(const std::filesystem::path& node) {
  std::error_code error;
  const std::filesystem::path driver =
      std::filesystem::canonical(node / "device" / "driver", error);
  return !error && driver.filename() == "amdxdna";
}

ProbeResult CollectPlatformProbe(const ProbeOptions& options) {
  ProbeResult result;
  const auto module = options.sys_module_root / "amdxdna";
  result.facts = {
      {"os", OperatingSystemName()},
      {"architecture", ProcessArchitecture()},
      {"executable", PathToUtf8(ExecutablePath())},
      {"driver",
       {
           {"module_present", std::filesystem::is_directory(module)},
           {"version", ReadText(module / "version").value_or("")},
           {"source_version", ReadText(module / "srcversion").value_or("")},
           {"init_state", ReadText(module / "initstate").value_or("")},
       }},
      {"devices", nlohmann::json::array()},
  };

  auto sys_nodes = SortedDirectories(options.sys_accel_root, "accel");
  sys_nodes.erase(
      std::remove_if(sys_nodes.begin(), sys_nodes.end(),
                     [](const std::filesystem::path& node) {
                       return !IsAmdXdnaNode(node);
                     }),
      sys_nodes.end());
  result.detected_devices = static_cast<int>(sys_nodes.size());
  result.checks.push_back(
      {"device.present",
       sys_nodes.empty() ? Status::kFail : Status::kPass,
       sys_nodes.empty() ? "no amdxdna accelerator sysfs node found"
                         : std::to_string(sys_nodes.size()) +
                               " accelerator node(s) found",
       {{"count", sys_nodes.size()}}});

  std::set<std::string> target_families;
  int complete_ioctl_devices = 0;
  for (const auto& sys_node : sys_nodes) {
    nlohmann::json device = {
        {"name", PathToUtf8(sys_node.filename())},
        {"sysfs_path", PathToUtf8(sys_node)},
    };
    const auto device_root = sys_node / "device";
    for (const char* field : {"vendor", "device", "revision", "vbnv",
                              "fw_version"}) {
      device[field] = ReadText(device_root / field).value_or("");
    }

    const std::filesystem::path dev_node =
        options.dev_accel_root / sys_node.filename();
    device["device_path"] = PathToUtf8(dev_node);
    const auto expected = ReadText(sys_node / "dev");
    device["expected_device_number"] = expected.value_or("");

    struct stat metadata = {};
    std::string access_error;
    bool usable = false;
    if (!expected || expected->find(':') == std::string::npos) {
      access_error = "sysfs dev value is missing or malformed";
    } else if (stat(dev_node.c_str(), &metadata) != 0) {
      access_error = std::strerror(errno);
    } else if (!S_ISCHR(metadata.st_mode)) {
      access_error = "device path is not a character device";
    } else {
      const auto separator = expected->find(':');
      try {
        const auto expected_major =
            static_cast<unsigned int>(std::stoul(expected->substr(0, separator)));
        const auto expected_minor = static_cast<unsigned int>(
            std::stoul(expected->substr(separator + 1)));
        if (major(metadata.st_rdev) != expected_major ||
            minor(metadata.st_rdev) != expected_minor) {
          access_error = "character device major/minor does not match sysfs";
        } else if (options.open_devices) {
          const int descriptor =
              open(dev_node.c_str(), O_RDWR | O_CLOEXEC);
          if (descriptor < 0) {
            access_error = std::strerror(errno);
            device["open_errno"] = errno;
          } else {
            device["ioctl"] = QueryIoctls(descriptor);
            const nlohmann::json& ioctl_facts = device["ioctl"];
            if (!ioctl_facts.contains("drm_version_error") &&
                !ioctl_facts.contains("firmware_version_error") &&
                !ioctl_facts.contains("aie_metadata_error")) {
              ++complete_ioctl_devices;
            }
            close(descriptor);
            usable = true;
          }
        } else {
          usable = true;
        }
      } catch (const std::exception&) {
        access_error = "sysfs dev value is malformed";
      }
    }
    device["usable"] = usable;
    if (!access_error.empty()) {
      device["access_error"] = access_error;
    }
    if (usable) {
      ++result.usable_devices;
    }
    const std::string target_family = InferTarget(device);
    device["target_family"] = target_family;
    if (!target_family.empty()) {
      target_families.insert(target_family);
    }
    result.facts["devices"].push_back(std::move(device));
  }
  if (target_families.size() == 1) {
    result.target_family = *target_families.begin();
  }

  result.checks.push_back(
      {"device.access",
       result.usable_devices > 0 ? Status::kPass : Status::kFail,
       result.usable_devices > 0
           ? std::to_string(result.usable_devices) +
                 " usable accelerator device(s)"
           : "no accelerator device can be opened read/write",
       {{"usable_count", result.usable_devices}}});
  if (options.open_devices && !sys_nodes.empty()) {
    const bool complete =
        complete_ioctl_devices > 0 &&
        complete_ioctl_devices == result.usable_devices;
    const bool usable = result.usable_devices > 0;
    result.checks.push_back(
        {"device.ioctl",
         usable ? Status::kPass : Status::kFail,
         complete
             ? "all usable accelerator devices answered diagnostic ioctls"
             : (usable
                    ? "device is usable; optional diagnostic ioctls were unavailable"
                    : "no accessible accelerator device accepted ioctls"),
         {{"complete_count", complete_ioctl_devices},
          {"usable_count", result.usable_devices},
          {"diagnostics_complete", complete}}});
  }
  result.facts["target_family"] = result.target_family;
  return result;
}

}  // namespace rocm_npu
