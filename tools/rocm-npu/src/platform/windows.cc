// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm_npu/platform.h"

#include "platform/hrx_library.h"
#include "platform/plugin.h"

#include <windows.h>

#include <bcrypt.h>
#include <cfgmgr32.h>
#include <devguid.h>
#include <initguid.h>
#include <devpkey.h>
#include <setupapi.h>
#include <winsvc.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace rocm_npu {
namespace {

constexpr GUID kComputeAcceleratorClass = {
    0xf01a9d53,
    0x3ff6,
    0x48d2,
    {0x9f, 0x97, 0xc8, 0xa7, 0x00, 0x4b, 0xe1, 0x0c},
};

std::string WideToUtf8(const std::wstring& value) {
  if (value.empty()) {
    return {};
  }
  const int length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                         value.data(),
                                         static_cast<int>(value.size()), nullptr,
                                         0, nullptr, nullptr);
  if (length <= 0) {
    return {};
  }
  std::string result(static_cast<std::size_t>(length), '\0');
  WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                      static_cast<int>(value.size()), result.data(), length,
                      nullptr, nullptr);
  return result;
}

std::wstring Utf8ToWide(const std::string& value) {
  if (value.empty()) {
    return {};
  }
  const int length =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                          static_cast<int>(value.size()), nullptr, 0);
  if (length <= 0) {
    return {};
  }
  std::wstring result(static_cast<std::size_t>(length), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                      static_cast<int>(value.size()), result.data(), length);
  return result;
}

std::wstring QuoteWindowsArgument(const std::wstring& argument) {
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    return argument;
  }
  std::wstring result = L"\"";
  std::size_t backslashes = 0;
  for (const wchar_t character : argument) {
    if (character == L'\\') {
      ++backslashes;
      continue;
    }
    if (character == L'"') {
      result.append(backslashes * 2 + 1, L'\\');
      result.push_back(L'"');
    } else {
      result.append(backslashes, L'\\');
      result.push_back(character);
    }
    backslashes = 0;
  }
  result.append(backslashes * 2, L'\\');
  result.push_back(L'"');
  return result;
}

std::optional<std::wstring> DeviceStringProperty(
    HDEVINFO devices, SP_DEVINFO_DATA& device, const DEVPROPKEY& key) {
  DEVPROPTYPE type = 0;
  DWORD size = 0;
  SetupDiGetDevicePropertyW(devices, &device, &key, &type, nullptr, 0, &size, 0);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || size == 0) {
    return std::nullopt;
  }
  std::vector<BYTE> buffer(size);
  if (!SetupDiGetDevicePropertyW(devices, &device, &key, &type, buffer.data(),
                                 size, nullptr, 0) ||
      (type != DEVPROP_TYPE_STRING && type != DEVPROP_TYPE_STRING_LIST)) {
    return std::nullopt;
  }
  const wchar_t* text = reinterpret_cast<const wchar_t*>(buffer.data());
  return std::wstring(text);
}

std::optional<std::string> DeviceDateProperty(
    HDEVINFO devices, SP_DEVINFO_DATA& device, const DEVPROPKEY& key) {
  DEVPROPTYPE type = 0;
  FILETIME value = {};
  DWORD size = 0;
  if (!SetupDiGetDevicePropertyW(
          devices, &device, &key, &type, reinterpret_cast<BYTE*>(&value),
          sizeof(value), &size, 0) ||
      type != DEVPROP_TYPE_FILETIME) {
    return std::nullopt;
  }
  SYSTEMTIME system_time = {};
  if (!FileTimeToSystemTime(&value, &system_time)) {
    return std::nullopt;
  }
  std::ostringstream output;
  output << std::setfill('0') << std::setw(4) << system_time.wYear << '-'
         << std::setw(2) << system_time.wMonth << '-' << std::setw(2)
         << system_time.wDay;
  return output.str();
}

std::wstring DeviceInstanceId(HDEVINFO devices, SP_DEVINFO_DATA& device) {
  DWORD size = 0;
  SetupDiGetDeviceInstanceIdW(devices, &device, nullptr, 0, &size);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || size == 0) {
    return {};
  }
  std::vector<wchar_t> buffer(size);
  if (!SetupDiGetDeviceInstanceIdW(devices, &device, buffer.data(), size,
                                   nullptr)) {
    return {};
  }
  return buffer.data();
}

std::optional<std::wstring> ServiceBinaryPath(const std::wstring& service) {
  SC_HANDLE manager = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT);
  if (manager == nullptr) {
    return std::nullopt;
  }
  SC_HANDLE handle =
      OpenServiceW(manager, service.c_str(), SERVICE_QUERY_CONFIG);
  if (handle == nullptr) {
    CloseServiceHandle(manager);
    return std::nullopt;
  }
  DWORD size = 0;
  QueryServiceConfigW(handle, nullptr, 0, &size);
  std::vector<BYTE> buffer(size);
  std::optional<std::wstring> result;
  if (size != 0 &&
      QueryServiceConfigW(
          handle, reinterpret_cast<QUERY_SERVICE_CONFIGW*>(buffer.data()), size,
          &size)) {
    const auto* configuration =
        reinterpret_cast<const QUERY_SERVICE_CONFIGW*>(buffer.data());
    if (configuration->lpBinaryPathName != nullptr) {
      std::wstring path = configuration->lpBinaryPathName;
      if (path.size() >= 2 && path.front() == L'"' &&
          path.back() == L'"') {
        path = path.substr(1, path.size() - 2);
      }
      constexpr wchar_t kSystemRootPrefix[] = L"\\SystemRoot\\";
      if (path.rfind(kSystemRootPrefix, 0) == 0) {
        std::vector<wchar_t> root(32768);
        const DWORD root_size =
            GetEnvironmentVariableW(L"SystemRoot", root.data(),
                                    static_cast<DWORD>(root.size()));
        if (root_size > 0 && root_size < root.size()) {
          path = (std::filesystem::path(root.data()) /
                  path.substr(std::size(kSystemRootPrefix) - 1))
                     .wstring();
        }
      }
      std::vector<wchar_t> expanded(32768);
      DWORD expanded_size =
          ExpandEnvironmentStringsW(path.c_str(), expanded.data(),
                                    static_cast<DWORD>(expanded.size()));
      result = expanded_size > 0 && expanded_size <= expanded.size()
                   ? std::wstring(expanded.data())
                   : path;
    }
  }
  CloseServiceHandle(handle);
  CloseServiceHandle(manager);
  return result;
}

std::optional<std::string> Sha256(const std::filesystem::path& path) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_size = 0;
  DWORD result_size = 0;
  std::vector<BYTE> object;
  std::array<BYTE, 32> digest = {};
  std::ifstream input(path, std::ios::binary);
  if (!input ||
      BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr,
                                  0) != 0 ||
      BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                        reinterpret_cast<PUCHAR>(&object_size),
                        sizeof(object_size), &result_size, 0) != 0) {
    if (algorithm != nullptr) {
      BCryptCloseAlgorithmProvider(algorithm, 0);
    }
    return std::nullopt;
  }
  object.resize(object_size);
  if (BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0,
                       0) != 0) {
    BCryptCloseAlgorithmProvider(algorithm, 0);
    return std::nullopt;
  }
  std::vector<char> buffer(1024 * 1024);
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0 &&
        BCryptHashData(hash, reinterpret_cast<PUCHAR>(buffer.data()),
                       static_cast<ULONG>(count), 0) != 0) {
      BCryptDestroyHash(hash);
      BCryptCloseAlgorithmProvider(algorithm, 0);
      return std::nullopt;
    }
  }
  const NTSTATUS status =
      BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0);
  BCryptDestroyHash(hash);
  BCryptCloseAlgorithmProvider(algorithm, 0);
  if (status != 0) {
    return std::nullopt;
  }
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (BYTE byte : digest) {
    result << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return result.str();
}

nlohmann::json FirmwareCandidates(const std::filesystem::path& driver_path) {
  nlohmann::json result = nlohmann::json::array();
  if (driver_path.empty()) {
    return result;
  }
  std::error_code error;
  for (std::filesystem::directory_iterator iterator(driver_path.parent_path(),
                                                     error),
       end;
       !error && iterator != end; iterator.increment(error)) {
    if (!iterator->is_regular_file(error)) {
      continue;
    }
    std::string name = iterator->path().filename().u8string();
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char character) {
                     return static_cast<char>(std::tolower(character));
                   });
    if (lower.rfind("ipu", 0) != 0 ||
        lower.size() < 11 ||
        lower.substr(lower.size() - 11) != "_pasid.sbin") {
      continue;
    }
    nlohmann::json candidate = {{"name", name}};
    if (auto digest = Sha256(iterator->path())) {
      candidate["sha256"] = *digest;
    } else {
      candidate["error"] = "could not hash firmware candidate";
    }
    std::ifstream input(iterator->path(), std::ios::binary);
    std::string header(4096, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    header.resize(static_cast<std::size_t>(input.gcount()));
    const std::size_t release = header.find("Release ");
    if (release != std::string::npos) {
      const std::size_t begin = release + 8;
      std::size_t end_version = begin;
      while (end_version < header.size() &&
             (std::isdigit(static_cast<unsigned char>(header[end_version])) ||
              header[end_version] == '.')) {
        ++end_version;
      }
      candidate["release"] = header.substr(begin, end_version - begin);
    }
    result.push_back(std::move(candidate));
  }
  return result;
}

std::optional<std::filesystem::path::string_type> NativeEnvironmentVariable(
    const char* name) {
  std::wstring wide_name;
  for (const char* cursor = name; *cursor != '\0'; ++cursor) {
    wide_name.push_back(static_cast<unsigned char>(*cursor));
  }
  const DWORD size = GetEnvironmentVariableW(wide_name.c_str(), nullptr, 0);
  if (size == 0) {
    return std::nullopt;
  }
  std::wstring value(size, L'\0');
  const DWORD length =
      GetEnvironmentVariableW(wide_name.c_str(), value.data(), size);
  if (length == 0 || length >= size) {
    return std::nullopt;
  }
  value.resize(length);
  return value;
}

}  // namespace

std::filesystem::path ExecutablePath() {
  std::vector<wchar_t> buffer(32768);
  const DWORD length =
      GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
  if (length == 0 || length == buffer.size()) {
    return {};
  }
  return std::filesystem::path(std::wstring(buffer.data(), length));
}

std::optional<std::filesystem::path::string_type> PluginSearchPathOverride() {
  return NativeEnvironmentVariable("ROCM_NPU_PLUGIN_PATH");
}

std::optional<std::filesystem::path::string_type> SystemExecutableSearchPath() {
  return NativeEnvironmentVariable("PATH");
}

std::filesystem::path::value_type PluginPathListSeparator() { return L';'; }

std::filesystem::path::value_type PluginPathListQuote() { return L'"'; }

std::string PluginDirectoryKey(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return value;
}

std::optional<std::string> PluginExecutableName(
    const std::filesystem::path& path) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    return std::nullopt;
  }
  const std::string filename = PathToUtf8(path.filename());
  constexpr const char* prefix = "rocm-npu-";
  if (filename.rfind(prefix, 0) != 0) {
    return std::nullopt;
  }
  const std::string extension =
      PluginDirectoryKey(PathToUtf8(path.extension()));
  if (!extension.empty() && extension != ".exe") {
    return std::nullopt;
  }
  std::string name = filename.substr(
      std::char_traits<char>::length(prefix),
      filename.size() - std::char_traits<char>::length(prefix) -
          extension.size());
  if (name.empty()) {
    return std::nullopt;
  }
  return name;
}

bool PluginEntryLess(const std::filesystem::path& left,
                     const std::filesystem::path& right) {
  const auto sort_key = [](const std::filesystem::path& path) {
    const std::string extension =
        PluginDirectoryKey(PathToUtf8(path.extension()));
    const int rank = extension == ".exe" ? 0 : 1;
    return std::make_pair(PluginDirectoryKey(PathToUtf8(path.stem())),
                          rank);
  };
  const auto left_key = sort_key(left);
  const auto right_key = sort_key(right);
  return left_key != right_key
             ? left_key < right_key
             : PathToUtf8(left.filename()) < PathToUtf8(right.filename());
}

int ExecutePluginProcess(
    const std::filesystem::path& executable,
    const std::vector<std::string>& arguments,
    const std::map<std::string, std::string>& environment_overrides,
    std::string& error) {
  struct SavedEnvironment {
    std::wstring name;
    std::optional<std::wstring> value;
  };
  std::vector<SavedEnvironment> saved_environment;
  for (const auto& [name, value] : environment_overrides) {
    (void)value;
    const std::wstring wide_name = Utf8ToWide(name);
    saved_environment.push_back(
        {wide_name, NativeEnvironmentVariable(name.c_str())});
  }
  const auto restore_environment = [&]() {
    for (const SavedEnvironment& saved : saved_environment) {
      SetEnvironmentVariableW(
          saved.name.c_str(),
          saved.value ? saved.value->c_str() : nullptr);
    }
  };
  for (const auto& [name, value] : environment_overrides) {
    const std::wstring wide_name = Utf8ToWide(name);
    const std::wstring wide_value = Utf8ToWide(value);
    if (!SetEnvironmentVariableW(
            wide_name.c_str(),
            value.empty() ? nullptr : wide_value.c_str())) {
      error =
          std::system_category().message(static_cast<int>(GetLastError()));
      restore_environment();
      return 2;
    }
  }

  std::wstring command_line = QuoteWindowsArgument(executable.native());
  for (const std::string& argument : arguments) {
    command_line.push_back(L' ');
    command_line += QuoteWindowsArgument(Utf8ToWide(argument));
  }
  std::vector<wchar_t> mutable_command(command_line.begin(),
                                       command_line.end());
  mutable_command.push_back(L'\0');
  const DWORD standard_handle_ids[] = {
      STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE};
  HANDLE inherited_handles[3] = {};
  for (std::size_t index = 0; index < 3; ++index) {
    const HANDLE source = GetStdHandle(standard_handle_ids[index]);
    if (source == nullptr || source == INVALID_HANDLE_VALUE ||
        !DuplicateHandle(GetCurrentProcess(), source, GetCurrentProcess(),
                         &inherited_handles[index], 0, TRUE,
                         DUPLICATE_SAME_ACCESS)) {
      error =
          std::system_category().message(static_cast<int>(GetLastError()));
      for (HANDLE handle : inherited_handles) {
        if (handle != nullptr) {
          CloseHandle(handle);
        }
      }
      restore_environment();
      return 2;
    }
  }
  SIZE_T attribute_size = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_size);
  if (attribute_size == 0) {
    error =
        std::system_category().message(static_cast<int>(GetLastError()));
    for (HANDLE handle : inherited_handles) {
      CloseHandle(handle);
    }
    restore_environment();
    return 2;
  }
  std::vector<unsigned char> attribute_storage(attribute_size);
  auto* attributes = reinterpret_cast<PPROC_THREAD_ATTRIBUTE_LIST>(
      attribute_storage.data());
  const bool attributes_initialized =
      InitializeProcThreadAttributeList(attributes, 1, 0, &attribute_size);
  if (!attributes_initialized ||
      !UpdateProcThreadAttribute(
          attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
          inherited_handles, sizeof(inherited_handles), nullptr, nullptr)) {
    error =
        std::system_category().message(static_cast<int>(GetLastError()));
    if (attributes_initialized) {
      DeleteProcThreadAttributeList(attributes);
    }
    for (HANDLE handle : inherited_handles) {
      CloseHandle(handle);
    }
    restore_environment();
    return 2;
  }

  STARTUPINFOEXW startup = {};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = inherited_handles[0];
  startup.StartupInfo.hStdOutput = inherited_handles[1];
  startup.StartupInfo.hStdError = inherited_handles[2];
  startup.lpAttributeList = attributes;
  PROCESS_INFORMATION process = {};
  const BOOL created = CreateProcessW(
      executable.c_str(), mutable_command.data(), nullptr, nullptr, TRUE,
      EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, &startup.StartupInfo,
      &process);
  const DWORD create_error = created ? ERROR_SUCCESS : GetLastError();
  DeleteProcThreadAttributeList(attributes);
  for (HANDLE handle : inherited_handles) {
    CloseHandle(handle);
  }
  restore_environment();
  if (!created) {
    error = std::system_category().message(static_cast<int>(create_error));
    return 2;
  }
  CloseHandle(process.hThread);
  const DWORD wait_result = WaitForSingleObject(process.hProcess, INFINITE);
  DWORD exit_code = 2;
  if (wait_result != WAIT_OBJECT_0 ||
      !GetExitCodeProcess(process.hProcess, &exit_code)) {
    error =
        std::system_category().message(static_cast<int>(GetLastError()));
    exit_code = 2;
  }
  CloseHandle(process.hProcess);
  return static_cast<int>(exit_code);
}

const char* HrxLibraryFilename() { return "hrx.dll"; }

std::optional<std::filesystem::path> HrxEnvironmentPath() {
  auto value = NativeEnvironmentVariable("HRX_LIBHRX");
  if (!value || value->empty()) {
    return std::nullopt;
  }
  return std::filesystem::path(*value);
}

std::vector<std::filesystem::path> PackagedHrxLibraryCandidates(
    const std::filesystem::path& executable_directory) {
  return {executable_directory / HrxLibraryFilename()};
}

std::vector<std::filesystem::path> HrxLibraryFallbackCandidates() {
  return {HrxLibraryFilename()};
}

void* OpenHrxLibrary(const std::filesystem::path& path, std::string& error) {
  DWORD flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
  std::filesystem::path load_path = path;
  if (load_path.has_parent_path()) {
    if (load_path.is_relative()) {
      std::error_code filesystem_error;
      load_path = std::filesystem::absolute(load_path, filesystem_error);
      if (filesystem_error) {
        error = filesystem_error.message();
        return nullptr;
      }
    }
    flags |= LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
  }
  HMODULE handle = LoadLibraryExW(load_path.c_str(), nullptr, flags);
  if (handle == nullptr) {
    error =
        std::system_category().message(static_cast<int>(GetLastError()));
  }
  return static_cast<void*>(handle);
}

void CloseHrxLibrary(void* handle) { FreeLibrary(static_cast<HMODULE>(handle)); }

void* LookupHrxSymbol(void* handle, const char* name) {
  return reinterpret_cast<void*>(
      GetProcAddress(static_cast<HMODULE>(handle), name));
}

std::string OperatingSystemName() {
  using RtlGetVersionFunction = LONG(WINAPI*)(OSVERSIONINFOW*);
  HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
  auto function = reinterpret_cast<RtlGetVersionFunction>(
      GetProcAddress(ntdll, "RtlGetVersion"));
  OSVERSIONINFOW version = {};
  version.dwOSVersionInfoSize = sizeof(version);
  if (function == nullptr || function(&version) != 0) {
    return "Windows";
  }
  return "Windows " + std::to_string(version.dwMajorVersion) + "." +
         std::to_string(version.dwMinorVersion) + "." +
         std::to_string(version.dwBuildNumber);
}

std::string ProcessArchitecture() {
#if defined(_M_X64)
  return "x86_64";
#elif defined(_M_ARM64)
  return "arm64";
#else
  return "unknown";
#endif
}

ProbeResult EvaluateWindowsDeviceRecords(const nlohmann::json& records) {
  ProbeResult result;
  result.facts["devices"] = nlohmann::json::array();
  std::set<std::string> target_families;
  if (records.is_array()) {
    for (const auto& source_record : records) {
      nlohmann::json record = source_record;
      std::string id = record.value("device_id", "");
      std::transform(id.begin(), id.end(), id.begin(),
                     [](unsigned char character) {
                       return static_cast<char>(std::toupper(character));
                     });
      std::string target_family;
      if (id.find("&DEV_17F0") != std::string::npos) {
        target_family = "npu2";
      } else if (id.find("&DEV_1502") != std::string::npos) {
        target_family = "npu";
      } else {
        continue;
      }
      record["target_family"] = target_family;
      result.facts["devices"].push_back(std::move(record));
      ++result.detected_devices;
      if (source_record.value("healthy", false)) {
        ++result.usable_devices;
      }
      target_families.insert(target_family);
    }
  }
  if (target_families.size() == 1) {
    result.target_family = *target_families.begin();
  }
  result.checks.push_back(
      {"device.present",
       result.detected_devices > 0 ? Status::kPass : Status::kFail,
       result.detected_devices > 0
           ? std::to_string(result.detected_devices) +
                 " AMD NPU device(s) found"
           : "no present AMD NPU compute accelerator found",
       {{"count", result.detected_devices}}});
  result.checks.push_back(
      {"device.access",
       result.usable_devices > 0 ? Status::kPass : Status::kFail,
       result.usable_devices > 0
           ? std::to_string(result.usable_devices) +
                 " healthy AMD NPU device(s)"
           : "no AMD NPU device has a healthy PnP state",
       {{"healthy_count", result.usable_devices}}});
  result.facts["target_family"] = result.target_family;
  return result;
}

ProbeResult CollectPlatformProbe(const ProbeOptions&) {
  ProbeResult result;
  result.facts = {
      {"os", OperatingSystemName()},
      {"architecture", ProcessArchitecture()},
      {"executable", ExecutablePath().u8string()},
      {"firmware_caveat",
       "driver package contents do not prove which firmware image the KMD loaded"},
  };
  nlohmann::json records = nlohmann::json::array();

  HDEVINFO devices = SetupDiGetClassDevsW(
      &kComputeAcceleratorClass, nullptr, nullptr, DIGCF_PRESENT);
  if (devices == INVALID_HANDLE_VALUE) {
    result.checks.push_back(
        {"device.enumeration", Status::kError,
         "SetupDiGetClassDevsW failed",
         {{"win32_error", GetLastError()}}});
    return result;
  }

  for (DWORD index = 0;; ++index) {
    SP_DEVINFO_DATA device = {};
    device.cbSize = sizeof(device);
    if (!SetupDiEnumDeviceInfo(devices, index, &device)) {
      if (GetLastError() != ERROR_NO_MORE_ITEMS) {
        const DWORD enumeration_error = GetLastError();
        result.facts["enumeration_error"] = enumeration_error;
        result.checks.push_back(
            {"device.enumeration", Status::kError,
             "SetupDiEnumDeviceInfo failed",
             {{"win32_error", enumeration_error}}});
      }
      break;
    }
    const std::wstring instance = DeviceInstanceId(devices, device);
    const auto service =
        DeviceStringProperty(devices, device, DEVPKEY_Device_Service);
    std::wstring upper_instance = instance;
    std::transform(upper_instance.begin(), upper_instance.end(),
                   upper_instance.begin(), towupper);
    if (upper_instance.rfind(L"PCI\\VEN_1022", 0) != 0 ||
        !service || *service != L"IpuMcdmDriver") {
      continue;
    }

    ULONG status = 0;
    ULONG problem = 0;
    const CONFIGRET configuration_status =
        CM_Get_DevNode_Status(&status, &problem, device.DevInst, 0);
    const bool healthy =
        configuration_status == CR_SUCCESS && (status & DN_STARTED) != 0 &&
        problem == 0;
    auto friendly =
        DeviceStringProperty(devices, device, DEVPKEY_Device_FriendlyName);
    if (!friendly) {
      friendly = DeviceStringProperty(devices, device, DEVPKEY_Device_DeviceDesc);
    }
    const auto version =
        DeviceStringProperty(devices, device, DEVPKEY_Device_DriverVersion);
    const auto provider =
        DeviceStringProperty(devices, device, DEVPKEY_Device_DriverProvider);
    const auto inf =
        DeviceStringProperty(devices, device, DEVPKEY_Device_DriverInfPath);
    const auto date =
        DeviceDateProperty(devices, device, DEVPKEY_Device_DriverDate);
    const auto driver_path = ServiceBinaryPath(*service);

    nlohmann::json record = {
        {"device_name", friendly ? WideToUtf8(*friendly) : ""},
        {"device_id", WideToUtf8(instance)},
        {"service", WideToUtf8(*service)},
        {"driver_version", version ? WideToUtf8(*version) : ""},
        {"driver_provider", provider ? WideToUtf8(*provider) : ""},
        {"driver_date", date.value_or("")},
        {"inf_name", inf ? WideToUtf8(*inf) : ""},
        {"driver_path", driver_path ? WideToUtf8(*driver_path) : ""},
        {"device_status", status},
        {"problem_code", problem},
        {"healthy", healthy},
    };
    if (driver_path) {
      record["firmware_candidates"] = FirmwareCandidates(*driver_path);
    }
    records.push_back(std::move(record));
  }
  SetupDiDestroyDeviceInfoList(devices);

  ProbeResult evaluated = EvaluateWindowsDeviceRecords(records);
  evaluated.facts.update(result.facts);
  evaluated.checks.insert(evaluated.checks.end(), result.checks.begin(),
                          result.checks.end());
  return evaluated;
}

}  // namespace rocm_npu
