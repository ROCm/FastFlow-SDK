# Qualification-owned native asset generation tools.
cmake_minimum_required(VERSION 3.25)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rocm_npu_fastflowlm_qualification_tools LANGUAGES CXX)
  if(NOT DEFINED FASTFLOWLM_SOURCE_DIR)
    message(FATAL_ERROR "FASTFLOWLM_SOURCE_DIR must point to FastFlowLM/src")
  endif()
  find_package(hrx CONFIG REQUIRED)
endif()

add_executable(
  fastflowlm-xadx-implementation-probe xadx_implementation_probe.cpp)
add_executable(
  fastflowlm-q4nx-vector-dump q4nx_vector_dump.cpp kernel_test_data.cpp)
set_target_properties(
  fastflowlm-xadx-implementation-probe fastflowlm-q4nx-vector-dump PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_features(
  fastflowlm-xadx-implementation-probe PRIVATE cxx_std_20)
target_compile_features(
  fastflowlm-q4nx-vector-dump PRIVATE cxx_std_20)
target_include_directories(
  fastflowlm-xadx-implementation-probe
  PRIVATE "${FASTFLOWLM_SOURCE_DIR}/include")
target_include_directories(
  fastflowlm-q4nx-vector-dump
  PRIVATE "${FASTFLOWLM_SOURCE_DIR}/include")
target_link_directories(
  fastflowlm-xadx-implementation-probe
  PRIVATE "${FASTFLOWLM_SOURCE_DIR}/lib")
target_link_directories(
  fastflowlm-q4nx-vector-dump
  PRIVATE "${FASTFLOWLM_SOURCE_DIR}/lib")
target_link_libraries(
  fastflowlm-xadx-implementation-probe PRIVATE
  dequant gemm gemma_text_npu mha q4_npu_eXpress hrx::hrx)
target_link_libraries(
  fastflowlm-q4nx-vector-dump PRIVATE q4_npu_eXpress hrx::hrx)
if(MSVC)
  foreach(target
      fastflowlm-xadx-implementation-probe fastflowlm-q4nx-vector-dump)
    target_compile_options(${target} PRIVATE /EHsc)
    target_compile_definitions(
      ${target} PRIVATE __WINDOWS__ NOMINMAX WIN32_LEAN_AND_MEAN)
  endforeach()
endif()
