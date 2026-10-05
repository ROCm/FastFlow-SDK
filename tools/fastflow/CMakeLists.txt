# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)
project(fastflow_cli LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)

if(MSVC)
  add_compile_options(/utf-8)
endif()

set(FASTFLOW_SDK_VERSION "0.0.0+unknown" CACHE STRING
    "Product version supplied by the release or packaging build")
set(FASTFLOW_SDK_SOURCE_REVISION "unknown" CACHE STRING
    "Source revision supplied by the release or packaging build")
set(FASTFLOW_SDK_BUILD_PLATFORM
    "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" CACHE STRING
    "Build platform recorded in command reports")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/build_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/fastflow_sdk/build_config.h"
  @ONLY
)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/manifest.json.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/fastflow_sdk/manifest.json"
  @ONLY
)

add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/nlohmann_json"
  "${CMAKE_CURRENT_BINARY_DIR}/nlohmann_json"
)

set(FASTFLOW_SDK_SOURCES
  src/command.cc
  src/report.cc
  src/product.cc
  src/sha256.cc
  src/power_mode.cc
  src/hrx_loader.cc
  src/commands/version.cc
  src/commands/plugin.cc
  src/commands/probe.cc
  src/commands/powermode.cc
  src/commands/doctor.cc
  src/commands/self_test.cc
)

if(WIN32)
  list(APPEND FASTFLOW_SDK_SOURCES src/platform/windows.cc)
  set(FASTFLOW_SDK_MAIN_SOURCE src/main_windows.cc)
else()
  list(APPEND FASTFLOW_SDK_SOURCES src/platform/linux.cc)
  set(FASTFLOW_SDK_MAIN_SOURCE src/main_posix.cc)
endif()

add_library(fastflow_sdk_core STATIC ${FASTFLOW_SDK_SOURCES})
target_compile_features(fastflow_sdk_core PUBLIC cxx_std_17)
target_compile_definitions(fastflow_sdk_core PUBLIC JSON_USE_IMPLICIT_CONVERSIONS=0)
target_include_directories(
  fastflow_sdk_core
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../projects/hrx/libhrx/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(fastflow_sdk_core PUBLIC fastflow_sdk_nlohmann_json)

if(WIN32)
  target_compile_definitions(fastflow_sdk_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_link_libraries(fastflow_sdk_core PUBLIC SetupAPI Cfgmgr32 Advapi32 Bcrypt)
else()
  target_link_libraries(fastflow_sdk_core PUBLIC dl)
endif()

if(MSVC)
  target_compile_options(fastflow_sdk_core PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(fastflow_sdk_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(fastflow src/main.cc "${FASTFLOW_SDK_MAIN_SOURCE}")
target_link_libraries(fastflow PRIVATE fastflow_sdk_core)

install(TARGETS fastflow RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/testdata/self-test/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/fastflow-sdk/self-test"
  PATTERN "README.md" EXCLUDE
  PATTERN "regenerate-*.sh" EXCLUDE
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/fastflow_sdk/manifest.json"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/fastflow-sdk"
)
install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/nlohmann_json/LICENSE.MIT"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/fastflow-sdk/licenses"
  RENAME nlohmann-json-LICENSE.MIT
)
install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../docs/third-party-notices.md"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/fastflow-sdk/licenses"
  RENAME THIRD_PARTY_NOTICES.md
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
