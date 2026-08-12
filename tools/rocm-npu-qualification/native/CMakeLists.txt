# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Qualification-owned native asset generation tools.
cmake_minimum_required(VERSION 3.25)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rocm_npu_fastflowlm_qualification_tools LANGUAGES CXX)
  if(NOT DEFINED FASTFLOWLM_SOURCE_DIR)
    message(FATAL_ERROR "FASTFLOWLM_SOURCE_DIR must point to FastFlowLM/src")
  endif()
  find_package(hrx CONFIG REQUIRED)
endif()
include(CTest)

if(NOT DEFINED KERNEL_CAPABILITIES_PATH)
  set(
    KERNEL_CAPABILITIES_PATH
    "${CMAKE_CURRENT_LIST_DIR}/../python/rocm_npu_qualification/fastflowlm-kernel-capabilities-v1.json")
endif()
if(NOT EXISTS "${KERNEL_CAPABILITIES_PATH}")
  message(FATAL_ERROR "KERNEL_CAPABILITIES_PATH must name the qualification contract")
endif()

add_executable(
  fastflowlm-xadx-implementation-probe xadx_implementation_probe.cpp)
add_executable(
  fastflowlm-q4nx-vector-dump q4nx_vector_dump.cpp kernel_test_data.cpp)
set(qualification_native_targets
  fastflowlm-xadx-implementation-probe fastflowlm-q4nx-vector-dump)

if(BUILD_TESTING)
  add_executable(
    fastflowlm-kernel-native-tests
    kernel_native_tests.cpp kernel_capabilities.cpp kernel_test_data.cpp)
  add_executable(
    fastflowlm-kernel-shape-probe kernel_shape_probe.cpp)
  list(APPEND qualification_native_targets
    fastflowlm-kernel-native-tests fastflowlm-kernel-shape-probe)
  add_test(
    NAME fastflowlm-kernel-native-tests
    COMMAND fastflowlm-kernel-native-tests "${KERNEL_CAPABILITIES_PATH}")
  add_test(
    NAME fastflowlm-dequant-baseline-shape
    COMMAND fastflowlm-kernel-shape-probe dequant 1024 1024)
  add_test(
    NAME fastflowlm-dequant-rejects-64x512
    COMMAND fastflowlm-kernel-shape-probe dequant 64 512)
  set_tests_properties(
    fastflowlm-dequant-rejects-64x512 PROPERTIES WILL_FAIL TRUE)
  add_test(
    NAME fastflowlm-mm-baseline-shape
    COMMAND fastflowlm-kernel-shape-probe mm 1024 1024 128)
endif()

foreach(target IN LISTS qualification_native_targets)
  set_target_properties(
    ${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  target_compile_features(${target} PRIVATE cxx_std_20)
  # FLM defaults FLM_USE_HRX OFF (XRT backend); these targets compile the FLM HRX
  # runtime path and must opt in, else the headers pull the XRT backend
  # (xrt/xrt_bo.h) and fail to build.
  target_compile_definitions(${target} PRIVATE FLM_USE_HRX)
  target_include_directories(
    ${target} PRIVATE
    "${FASTFLOWLM_SOURCE_DIR}/include" "${CMAKE_CURRENT_LIST_DIR}")
  # The HRX build links the per-backend engine libs under lib/hrx
  # (dequant, gemm, q4_npu_eXpress, ...).
  target_link_directories(
    ${target} PRIVATE "${FASTFLOWLM_SOURCE_DIR}/lib/hrx")
  if(MSVC)
    target_compile_options(${target} PRIVATE /EHsc)
    target_compile_definitions(
      ${target} PRIVATE __WINDOWS__ NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()
endforeach()

target_link_libraries(
  fastflowlm-xadx-implementation-probe PRIVATE
  dequant gemm gemma_text_npu mha q4_npu_eXpress hrx::hrx)
target_link_libraries(
  fastflowlm-q4nx-vector-dump PRIVATE q4_npu_eXpress hrx::hrx)
if(BUILD_TESTING)
  target_link_libraries(
    fastflowlm-kernel-native-tests PRIVATE q4_npu_eXpress hrx::hrx)
  target_link_libraries(
    fastflowlm-kernel-shape-probe PRIVATE dequant gemm hrx::hrx)
endif()
