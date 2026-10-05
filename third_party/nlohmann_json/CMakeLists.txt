# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(fastflow_sdk_nlohmann_json INTERFACE)
target_include_directories(
  fastflow_sdk_nlohmann_json
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
