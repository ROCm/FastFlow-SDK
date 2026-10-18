// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cwctype>
#include <string>
#include <vector>

namespace {

std::wstring QuoteArgument(const std::wstring& argument) {
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    return argument;
  }
  std::wstring quoted = L"\"";
  std::size_t backslashes = 0;
  for (const wchar_t character : argument) {
    if (character == L'\\') {
      ++backslashes;
      continue;
    }
    if (character == L'"') {
      quoted.append(backslashes * 2 + 1, L'\\');
      quoted.push_back(L'"');
      backslashes = 0;
      continue;
    }
    quoted.append(backslashes, L'\\');
    backslashes = 0;
    quoted.push_back(character);
  }
  quoted.append(backslashes * 2, L'\\');
  quoted.push_back(L'"');
  return quoted;
}

std::wstring ExecutablePath() {
  std::vector<wchar_t> buffer(512);
  while (true) {
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(),
                                            static_cast<DWORD>(buffer.size()));
    if (length == 0) {
      return {};
    }
    if (length < buffer.size() - 1) {
      return std::wstring(buffer.data(), length);
    }
    buffer.resize(buffer.size() * 2);
  }
}

std::wstring Lowercase(std::wstring value) {
  for (wchar_t& character : value) {
    character = static_cast<wchar_t>(std::towlower(character));
  }
  return value;
}

struct Forwarding {
  const wchar_t* executable;
  std::vector<const wchar_t*> arguments;
};

bool ResolveForwarding(const std::wstring& alias, Forwarding& forwarding) {
  if (alias == L"clang.exe") {
    forwarding = {L"clang++.exe", {L"--driver-mode=gcc"}};
  } else if (alias == L"clang-cl.exe") {
    forwarding = {L"clang++.exe", {L"--driver-mode=cl"}};
  } else if (alias == L"clang-cpp.exe") {
    forwarding = {L"clang++.exe", {L"--driver-mode=cpp"}};
  } else if (alias == L"ld.lld.exe") {
    forwarding = {L"lld.exe", {L"-flavor", L"gnu"}};
  } else if (alias == L"ld64.lld.exe") {
    forwarding = {L"lld.exe", {L"-flavor", L"darwin"}};
  } else if (alias == L"lld-link.exe") {
    forwarding = {L"lld.exe", {L"-flavor", L"link"}};
  } else if (alias == L"wasm-ld.exe") {
    forwarding = {L"lld.exe", {L"-flavor", L"wasm"}};
  } else if (alias == L"llvm-readelf.exe") {
    forwarding = {L"llvm-readobj.exe", {L"--elf-output-style=GNU"}};
  } else {
    return false;
  }
  return true;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  const std::wstring launcher_path = ExecutablePath();
  const std::size_t separator = launcher_path.find_last_of(L"\\/");
  if (launcher_path.empty() || separator == std::wstring::npos) {
    return 127;
  }
  const std::wstring directory = launcher_path.substr(0, separator + 1);
  const std::wstring alias = Lowercase(launcher_path.substr(separator + 1));
  Forwarding forwarding;
  if (!ResolveForwarding(alias, forwarding)) {
    return 127;
  }

  const std::wstring executable = directory + forwarding.executable;
  std::wstring command_line = QuoteArgument(executable);
  // Clang's integrated frontend dispatch requires -cc1* to be argv[1].
  const bool direct_frontend =
      std::wstring(forwarding.executable) == L"clang++.exe" && argc > 1 &&
      std::wstring(argv[1]).rfind(L"-cc1", 0) == 0;
  if (!direct_frontend) {
    for (const wchar_t* argument : forwarding.arguments) {
      command_line.push_back(L' ');
      command_line += QuoteArgument(argument);
    }
  }
  for (int index = 1; index < argc; ++index) {
    command_line.push_back(L' ');
    command_line += QuoteArgument(argv[index]);
  }

  STARTUPINFOW startup = {};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process = {};
  const BOOL created =
      CreateProcessW(executable.c_str(), command_line.data(), nullptr, nullptr,
                     TRUE, 0, nullptr, nullptr, &startup, &process);
  if (!created) {
    return 127;
  }
  CloseHandle(process.hThread);
  const DWORD wait_result = WaitForSingleObject(process.hProcess, INFINITE);
  DWORD exit_code = 127;
  if (wait_result == WAIT_OBJECT_0) {
    GetExitCodeProcess(process.hProcess, &exit_code);
  }
  CloseHandle(process.hProcess);
  return static_cast<int>(exit_code);
}
