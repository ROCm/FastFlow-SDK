cmake_minimum_required(VERSION 3.25)
project(rocm_npu_fastflowlm_kernel_smoke LANGUAGES CXX)
include(CTest)

if(NOT DEFINED FASTFLOWLM_SOURCE_DIR)
  message(FATAL_ERROR "FASTFLOWLM_SOURCE_DIR must point to FastFlowLM/src")
endif()

find_package(hrx CONFIG REQUIRED)

add_executable(
  fastflowlm-hrx-kernel-smoke
  hrx_kernel_smoke.cpp kernel_capabilities.cpp kernel_case.cpp
  kernel_test_data.cpp)

set(fastflowlm_targets fastflowlm-hrx-kernel-smoke)
if(BUILD_TESTING)
  add_executable(
    fastflowlm-kernel-native-tests
    kernel_native_tests.cpp kernel_capabilities.cpp kernel_test_data.cpp)
  add_executable(
    fastflowlm-kernel-shape-probe kernel_shape_probe.cpp)
  list(APPEND fastflowlm_targets
    fastflowlm-kernel-native-tests fastflowlm-kernel-shape-probe)
  add_test(
    NAME fastflowlm-kernel-native-tests
    COMMAND fastflowlm-kernel-native-tests
      "${CMAKE_CURRENT_SOURCE_DIR}/kernel_capabilities.json")
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

foreach(target IN LISTS fastflowlm_targets)
  target_compile_features(${target} PRIVATE cxx_std_20)
  target_include_directories(
    ${target} PRIVATE "${FASTFLOWLM_SOURCE_DIR}/include")
  target_link_directories(
    ${target} PRIVATE "${FASTFLOWLM_SOURCE_DIR}/lib")
  if(MSVC)
    target_compile_options(${target} PRIVATE /EHsc)
    target_compile_definitions(
      ${target} PRIVATE __WINDOWS__ NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()
endforeach()

target_link_libraries(
  fastflowlm-hrx-kernel-smoke PRIVATE
  dequant
  gemm
  q4_npu_eXpress
  hrx::hrx)
if(BUILD_TESTING)
  target_link_libraries(
    fastflowlm-kernel-native-tests PRIVATE q4_npu_eXpress hrx::hrx)
  target_link_libraries(
    fastflowlm-kernel-shape-probe PRIVATE
    dequant gemm hrx::hrx)
endif()

if(NOT MSVC)
  set_target_properties(
    fastflowlm-hrx-kernel-smoke PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib/flm:$ORIGIN/../lib")
endif()

install(TARGETS fastflowlm-hrx-kernel-smoke RUNTIME DESTINATION bin)
install(
  FILES kernel_capabilities.json
  DESTINATION share/fastflowlm-kernel-smoke)
