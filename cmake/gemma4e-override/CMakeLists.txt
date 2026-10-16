# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Builds the Gemma 4 engine from the sources shipped in an installed FastFlow
# SDK, against that install's HRX. The product flm stays on the prebuilt engine.
cmake_minimum_required(VERSION 3.24)
project(gemma4e_override LANGUAGES CXX)

if(NOT DEFINED FASTFLOW_SDK_ROOT)
    # share/fastflow-sdk/gemma4e -> archive or wheel prefix root
    get_filename_component(FASTFLOW_SDK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
endif()
list(PREPEND CMAKE_PREFIX_PATH "${FASTFLOW_SDK_ROOT}")

find_package(hrx CONFIG REQUIRED)

set(FLM_USE_HRX ON)
set(FLM_ENGINE_VERBOSE 0)
set(FLM_ENGINE_DEBUG_LEVEL 0)
set(FLM_ENGINE_NATIVE_ARCH OFF)
# Archive installs keep the import libraries next to the HRX prefix. A wheel
# install splits them: FASTFLOW_SDK_ROOT is fastflow-sdk-runtime, and
# FLM_ENGINE_LIB_DIR is fastflow_sdk_fastflowlm.engine_lib_dir().
if(NOT FLM_ENGINE_LIB_DIR)
    set(FLM_ENGINE_LIB_DIR "${FASTFLOW_SDK_ROOT}/lib/flm")
endif()
# Directories for the override header, the generated sequence headers, and the
# AIE runtime headers. A semicolon-separated CMake list, so paths may contain
# spaces. FLM_OVERRIDES_HEADER is the header the engine includes. Empty leaves
# the prebuilt dispatch in place.
set(FLM_IRON_INCLUDE_DIRS "" CACHE STRING
    "Include directories for an FLM_OVERRIDE header and its generated sources.")
set(FLM_OVERRIDES_HEADER "overrides.hpp" CACHE STRING
    "Header included via FLM_OVERRIDES. Pass an empty string to build the engine unchanged.")
set(FLM_OVERRIDE_FLAGS "" CACHE STRING
    "Extra compile flags for the Gemma 4 engine. See include/flm_override.hpp.")

add_subdirectory(src/detail)

# Applied here, after the engine target exists, so a path with spaces is one
# include directory. FastFlowLM's FLM_OVERRIDE_FLAGS goes through the Windows
# command-line splitter, which strips quotes.
if(FLM_IRON_INCLUDE_DIRS)
    target_include_directories(gemma4e_npu PRIVATE ${FLM_IRON_INCLUDE_DIRS})
endif()
if(FLM_OVERRIDES_HEADER)
    if(MSVC)
        target_compile_options(gemma4e_npu PRIVATE "-DFLM_OVERRIDES=<${FLM_OVERRIDES_HEADER}>")
    else()
        target_compile_options(gemma4e_npu PRIVATE "-DFLM_OVERRIDES=\"${FLM_OVERRIDES_HEADER}\"")
    endif()
    # overrides.hpp uses __VA_OPT__. cl rejects that token unless its
    # conforming preprocessor is enabled. clang-cl accepts it already.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(gemma4e_npu PRIVATE /Zc:preprocessor)
    endif()
endif()
