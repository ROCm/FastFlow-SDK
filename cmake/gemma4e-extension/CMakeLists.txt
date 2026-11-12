# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Builds a Gemma 4 model extension from the sources shipped in an installed
# FastFlow SDK. The engine sources are compiled into the plugin. The plugin
# does not replace the engine or XCLBINs installed beside flm.
cmake_minimum_required(VERSION 3.24)
project(gemma4e_open_extension LANGUAGES C CXX)

# tokenizers-cpp is a static library linked into this shared object. Linux
# rejects that link unless those objects were built position-independent.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# share/fastflow-sdk/gemma4e -> archive or FastFlowLM wheel prefix root.
get_filename_component(FASTFLOWLM_PACKAGE_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT DEFINED FASTFLOW_SDK_ROOT)
    set(FASTFLOW_SDK_ROOT "${FASTFLOWLM_PACKAGE_ROOT}")
endif()
list(PREPEND CMAKE_PREFIX_PATH
    "${FASTFLOWLM_PACKAGE_ROOT}"
    "${FASTFLOW_SDK_ROOT}")

find_package(hrx CONFIG REQUIRED)
find_package(FastFlowLMModelExtension CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# Archive installs keep the import libraries next to the HRX prefix. A wheel
# install splits them: FASTFLOW_SDK_ROOT is fastflow-sdk-runtime, and
# FLM_ENGINE_LIB_DIR is fastflow_sdk_fastflowlm.engine_lib_dir().
if(NOT FLM_ENGINE_LIB_DIR)
    set(FLM_ENGINE_LIB_DIR "${FASTFLOW_SDK_ROOT}/lib/flm")
endif()
set(FLM_ENGINE_BIN_DIR "${FLM_ENGINE_LIB_DIR}" CACHE PATH
    "Directory containing the FastFlowLM engine runtime libraries.")
if(WIN32 AND NOT EXISTS "${FLM_ENGINE_BIN_DIR}/q4_npu_eXpress.dll")
    set(FLM_ENGINE_BIN_DIR "${FASTFLOWLM_PACKAGE_ROOT}/bin")
endif()
# Directories for the custom dispatch header, generated sequence headers, and
# AIE runtime headers. A semicolon-separated CMake list, so paths may contain
# spaces.
set(FLM_IRON_INCLUDE_DIRS "" CACHE STRING
    "Include directories for the IRON dispatch header and generated sources.")
set(FLM_EXTENSION_ENGINE_HEADER "" CACHE STRING
    "Optional header that customizes the extension-owned Gemma 4 engine. Empty keeps the stock engine.")
set(FLM_OVERRIDE_FLAGS "" CACHE STRING
    "Extra compile flags for the Gemma 4 engine. See include/flm_override.hpp.")
set(FLM_EXTENSION_BUNDLE_DIR "${CMAKE_BINARY_DIR}/bundle" CACHE PATH
    "Output directory containing extension.json, libraries, and assets.")

# The installed tree is share/fastflow-sdk/gemma4e. Its src/ and third_party/
# are enough to configure; a FastFlowLM checkout is not required.
set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/src" CACHE PATH
    "Shipped Gemma 4 open sources (engine, frontend, and media helpers)")
if(NOT EXISTS "${FLM_SOURCE_DIR}/frontend/gemma4_open_frontend.cpp")
    message(FATAL_ERROR
        "FLM_SOURCE_DIR must contain frontend/gemma4_open_frontend.cpp")
endif()
set(FLM_TOKENIZERS_CPP_DIR
    "${CMAKE_CURRENT_LIST_DIR}/third_party/tokenizers-cpp" CACHE PATH
    "tokenizers-cpp sources shipped beside this project")
set(_tokenizers_dir "${FLM_TOKENIZERS_CPP_DIR}")
if(NOT EXISTS "${_tokenizers_dir}/CMakeLists.txt")
    message(FATAL_ERROR
        "tokenizers-cpp is missing: ${_tokenizers_dir}")
endif()
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # The plugin uses the static CRT. tokenizers-cpp is a static library, so
    # its C++ objects have to be built with that same runtime.
    # tokenizers-cpp clears TOKENIZERS_CPP_RUST_FLAGS before it builds Cargo,
    # then appends -Ctarget-feature=+crt-static when this option is MT.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(TOKENIZERS_CPP_MSVC_RUNTIME_LIBRARY "MT"
        CACHE STRING "CRT used by the Gemma 4 open tokenizer" FORCE)
endif()
add_subdirectory("${_tokenizers_dir}"
    "${CMAKE_CURRENT_BINARY_DIR}/tokenizers-cpp" EXCLUDE_FROM_ALL)
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    file(READ "${FLM_SOURCE_DIR}/cmake/FlmImageCodecs.cmake" _flm_image_codecs)
    string(REPLACE "-DWITH_CRT_DLL=ON" "-DWITH_CRT_DLL=OFF"
        _flm_image_codecs "${_flm_image_codecs}")
    string(REPLACE
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL"
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
        _flm_image_codecs "${_flm_image_codecs}")
    string(REPLACE "\${CMAKE_CURRENT_LIST_DIR}/.."
        "${FLM_SOURCE_DIR}" _flm_image_codecs "${_flm_image_codecs}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/FlmImageCodecs.cmake"
        "${_flm_image_codecs}")
    file(READ "${FLM_SOURCE_DIR}/cmake/FlmAudioCodecs.cmake" _flm_audio_codecs)
    string(REPLACE
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON\n    -DCMAKE_POLICY_DEFAULT_CMP0091=NEW\n    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
        _flm_audio_codecs "${_flm_audio_codecs}")
    string(REPLACE "\${CMAKE_CURRENT_LIST_DIR}/.."
        "${FLM_SOURCE_DIR}" _flm_audio_codecs "${_flm_audio_codecs}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/FlmAudioCodecs.cmake"
        "${_flm_audio_codecs}")
    include("${CMAKE_CURRENT_BINARY_DIR}/FlmImageCodecs.cmake")
    include("${CMAKE_CURRENT_BINARY_DIR}/FlmAudioCodecs.cmake")
else()
    include("${FLM_SOURCE_DIR}/cmake/FlmImageCodecs.cmake")
    include("${FLM_SOURCE_DIR}/cmake/FlmAudioCodecs.cmake")
endif()

set(_engine_dir "${CMAKE_CURRENT_LIST_DIR}/src/detail")
file(GLOB _engine_sources
    "${_engine_dir}/gemma4e_npu/*.cpp"
    "${_engine_dir}/dequant/*.cpp"
    "${_engine_dir}/gemm/*.cpp"
    "${_engine_dir}/lm_head/*.cpp")
list(APPEND _engine_sources "${_engine_dir}/vision_common/norm.cpp")
set(_frontend_sources
    "${FLM_SOURCE_DIR}/frontend/gemma4_open_frontend.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_continue.cpp"
    "${FLM_SOURCE_DIR}/common/image/image_reader.cpp"
    "${FLM_SOURCE_DIR}/common/image_process_utils/imageproc.cpp"
    "${FLM_SOURCE_DIR}/common/image_process_utils/imageprocAVX512.cpp"
    "${FLM_SOURCE_DIR}/common/audio/audio_reader.cpp"
    "${FLM_SOURCE_DIR}/common/audio/audio_decoder.cpp"
    "${FLM_SOURCE_DIR}/common/audio/audio_resampler.cpp"
    "${FLM_SOURCE_DIR}/common/audio_process_utils/audioproc.cpp"
    "${FLM_SOURCE_DIR}/common/audio_process_utils/audioprocAVX512.cpp")

add_library(gemma4e_open_extension SHARED
    model_extension.cpp
    ${_frontend_sources}
    ${_engine_sources})
target_compile_features(gemma4e_open_extension PRIVATE cxx_std_20)
target_include_directories(gemma4e_open_extension PRIVATE
    src/include
    src/detail)
target_compile_definitions(gemma4e_open_extension PRIVATE
    FLM_MODEL_EXTENSION_IMPLEMENTATION
    FLM_USE_HRX=1
    VERBOSE=0
    DEBUG_LEVEL=0)
# Applied after the target exists, so a path with spaces stays one include
# directory.
if(FLM_IRON_INCLUDE_DIRS)
    target_include_directories(gemma4e_open_extension PRIVATE
        ${FLM_IRON_INCLUDE_DIRS})
endif()
if(FLM_EXTENSION_ENGINE_HEADER)
    if(MSVC)
        target_compile_options(gemma4e_open_extension PRIVATE
            "-DFLM_OVERRIDES=<${FLM_EXTENSION_ENGINE_HEADER}>")
    else()
        target_compile_options(gemma4e_open_extension PRIVATE
            "-DFLM_OVERRIDES=\"${FLM_EXTENSION_ENGINE_HEADER}\"")
    endif()
    # The IRON dispatch header uses __VA_OPT__. cl rejects that token unless its
    # conforming preprocessor is enabled. clang-cl accepts it already.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(gemma4e_open_extension PRIVATE /Zc:preprocessor)
    endif()
endif()
if(FLM_OVERRIDE_FLAGS)
    separate_arguments(_extension_override_flags NATIVE_COMMAND
        "${FLM_OVERRIDE_FLAGS}")
    target_compile_options(gemma4e_open_extension PRIVATE
        ${_extension_override_flags})
endif()

find_library(Q4_NPU_EXPRESS_LIBRARY NAMES q4_npu_eXpress
    HINTS "${FLM_ENGINE_LIB_DIR}" REQUIRED NO_DEFAULT_PATH)
if(WIN32)
    find_library(MHA_LIBRARY NAMES mha
        HINTS "${FLM_ENGINE_LIB_DIR}" REQUIRED NO_DEFAULT_PATH)
endif()
target_link_libraries(gemma4e_open_extension PRIVATE
    FastFlowLM::ModelExtension
    tokenizers_cpp
    flm_image_codecs
    flm_audio_codecs
    hrx::hrx
    OpenMP::OpenMP_CXX
    "${Q4_NPU_EXPRESS_LIBRARY}")
if(WIN32)
    target_link_libraries(gemma4e_open_extension PRIVATE "${MHA_LIBRARY}")
endif()
if(NOT WIN32)
    # The compiled-in engine must not export gemma4e_npu symbols. flm already
    # loaded the prebuilt engine, and Linux would bind this plugin's calls to
    # that global copy. q4nx is linked above. mha stays prebuilt and is
    # resolved from the process.
    target_compile_options(gemma4e_open_extension PRIVATE
        -O3 -Wall -fmax-errors=1
        -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma
        -ffast-math)
    set_target_properties(gemma4e_open_extension PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
    target_link_options(gemma4e_open_extension PRIVATE
        -Wl,-Bsymbolic-functions
        -Wl,--allow-shlib-undefined)
endif()
if(MSVC)
    target_compile_definitions(gemma4e_open_extension PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_DEPRECATE
        __WINDOWS__)
    target_compile_options(gemma4e_open_extension PRIVATE /arch:AVX512)
    # cl keeps the static CRT. clang-cl links the /MD HRX and vcpkg libraries.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(_extension_msvc_runtime "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set(_extension_msvc_runtime "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
    set_target_properties(gemma4e_open_extension PROPERTIES
        MSVC_RUNTIME_LIBRARY "${_extension_msvc_runtime}")
    if(NOT VCPKG_TOOLCHAIN)
        target_include_directories(gemma4e_open_extension PRIVATE
            "C:/dev/boost_1_88_0")
    endif()
endif()

file(MAKE_DIRECTORY "${FLM_EXTENSION_BUNDLE_DIR}/catalog")
set(_extension_lib_dir "${FLM_EXTENSION_BUNDLE_DIR}/lib")
set_target_properties(gemma4e_open_extension PROPERTIES
    BUILD_RPATH "$ORIGIN"
    LIBRARY_OUTPUT_DIRECTORY "${_extension_lib_dir}"
    RUNTIME_OUTPUT_DIRECTORY "${_extension_lib_dir}")
foreach(_config RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
    set_target_properties(gemma4e_open_extension PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${_config} "${_extension_lib_dir}"
        RUNTIME_OUTPUT_DIRECTORY_${_config} "${_extension_lib_dir}")
endforeach()
if(WIN32)
    file(COPY
        "${FLM_ENGINE_BIN_DIR}/q4_npu_eXpress.dll"
        "${FLM_ENGINE_BIN_DIR}/mha.dll"
        DESTINATION "${_extension_lib_dir}")
else()
    file(COPY "${Q4_NPU_EXPRESS_LIBRARY}"
        DESTINATION "${_extension_lib_dir}")
endif()

# The engine loads <asset_root>/xclbins/Gemma4-E2B-IT-NPU2. asset_root is the
# bundle, so the xclbins have to be copied in before the manifest is usable.
# The default is the tree installed beside flm. An IRON build passes its
# staged Gemma4-E2B-IT-NPU2 directory, which also contains iron/*.bin.
set(FLM_GEMMA4_XCLBIN_DIR
    "${FASTFLOWLM_PACKAGE_ROOT}/share/flm/xclbins/Gemma4-E2B-IT-NPU2"
    CACHE PATH "Gemma4-E2B-IT-NPU2 directory copied into the extension bundle")
if(NOT IS_DIRECTORY "${FLM_GEMMA4_XCLBIN_DIR}")
    message(FATAL_ERROR
        "FLM_GEMMA4_XCLBIN_DIR must be a Gemma4-E2B-IT-NPU2 xclbin directory. "
        "The installed default is missing: ${FLM_GEMMA4_XCLBIN_DIR}")
endif()
file(GLOB _gemma4_xclbins "${FLM_GEMMA4_XCLBIN_DIR}/*.xclbin")
if(NOT _gemma4_xclbins)
    message(FATAL_ERROR
        "FLM_GEMMA4_XCLBIN_DIR contains no .xclbin files: ${FLM_GEMMA4_XCLBIN_DIR}")
endif()
set(_bundle_xclbins "${FLM_EXTENSION_BUNDLE_DIR}/xclbins/Gemma4-E2B-IT-NPU2")
file(REMOVE_RECURSE "${_bundle_xclbins}")
file(COPY "${FLM_GEMMA4_XCLBIN_DIR}/" DESTINATION "${_bundle_xclbins}")

configure_file(model-list.json
    "${FLM_EXTENSION_BUNDLE_DIR}/catalog/model-list.json" COPYONLY)
file(GENERATE OUTPUT "${FLM_EXTENSION_BUNDLE_DIR}/extension.json"
    CONTENT
"{
  \"schema_version\": 1,
  \"name\": \"gemma4-open\",
  \"version\": \"1.0.0\",
  \"library\": \"lib/$<TARGET_FILE_NAME:gemma4e_open_extension>\",
  \"asset_root\": \".\",
  \"catalog\": \"catalog/model-list.json\"
}
")
