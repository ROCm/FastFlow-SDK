# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocm_npu_nlohmann_json INTERFACE)
target_include_directories(
  rocm_npu_nlohmann_json
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
