# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)
project(rocm_npu_cli LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)

if(MSVC)
  add_compile_options(/utf-8)
endif()

set(ROCM_NPU_VERSION "0.0.0+unknown" CACHE STRING
    "Product version supplied by the release or packaging build")
set(ROCM_NPU_SOURCE_REVISION "unknown" CACHE STRING
    "Source revision supplied by the release or packaging build")
set(ROCM_NPU_BUILD_PLATFORM
    "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" CACHE STRING
    "Build platform recorded in command reports")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/build_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/rocm_npu/build_config.h"
  @ONLY
)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/manifest.json.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/rocm_npu/manifest.json"
  @ONLY
)

add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/nlohmann_json"
  "${CMAKE_CURRENT_BINARY_DIR}/nlohmann_json"
)

set(ROCM_NPU_SOURCES
  src/command.cc
  src/report.cc
  src/product.cc
  src/sha256.cc
  src/hrx_loader.cc
  src/commands/version.cc
  src/commands/plugin.cc
  src/commands/probe.cc
  src/commands/doctor.cc
  src/commands/self_test.cc
)

if(WIN32)
  list(APPEND ROCM_NPU_SOURCES src/platform/windows.cc)
  set(ROCM_NPU_MAIN_SOURCE src/main_windows.cc)
else()
  list(APPEND ROCM_NPU_SOURCES src/platform/linux.cc)
  set(ROCM_NPU_MAIN_SOURCE src/main_posix.cc)
endif()

add_library(rocm_npu_core STATIC ${ROCM_NPU_SOURCES})
target_compile_features(rocm_npu_core PUBLIC cxx_std_17)
target_compile_definitions(rocm_npu_core PUBLIC JSON_USE_IMPLICIT_CONVERSIONS=0)
target_include_directories(
  rocm_npu_core
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../projects/hrx/libhrx/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(rocm_npu_core PUBLIC rocm_npu_nlohmann_json)

if(WIN32)
  target_compile_definitions(rocm_npu_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_link_libraries(rocm_npu_core PUBLIC SetupAPI Cfgmgr32 Advapi32 Bcrypt)
else()
  target_link_libraries(rocm_npu_core PUBLIC dl)
endif()

if(MSVC)
  target_compile_options(rocm_npu_core PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(rocm_npu_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(rocm-npu src/main.cc "${ROCM_NPU_MAIN_SOURCE}")
target_link_libraries(rocm-npu PRIVATE rocm_npu_core)

install(TARGETS rocm-npu RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/testdata/self-test/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/rocm-npu/self-test"
  PATTERN "README.md" EXCLUDE
  PATTERN "regenerate-*.sh" EXCLUDE
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/rocm_npu/manifest.json"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/rocm-npu"
)
install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/nlohmann_json/LICENSE.MIT"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/rocm-npu/licenses"
  RENAME nlohmann-json-LICENSE.MIT
)
install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../docs/third-party-notices.md"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/rocm-npu/licenses"
  RENAME THIRD_PARTY_NOTICES.md
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
