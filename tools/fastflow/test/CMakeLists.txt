# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
if(WIN32)
  set(_fastflow_sdk_test_platform_source platform_windows.cc)
  set(_fastflow_sdk_fake_plugin_main_source fake_plugin_main_windows.cc)
  set(_fastflow_sdk_dispatch_platform_source plugin_dispatch_windows.cc)
else()
  set(_fastflow_sdk_test_platform_source platform_posix.cc)
  set(_fastflow_sdk_fake_plugin_main_source fake_plugin_main_posix.cc)
  set(_fastflow_sdk_dispatch_platform_source plugin_dispatch_posix.cc)
endif()

add_library(fastflow_sdk_test_platform STATIC "${_fastflow_sdk_test_platform_source}")
target_compile_features(fastflow_sdk_test_platform PUBLIC cxx_std_17)

add_executable(fastflow_sdk_core_test core_test.cc)
target_link_libraries(fastflow_sdk_core_test PRIVATE fastflow_sdk_core)
add_test(NAME fastflow_sdk_core_test COMMAND fastflow_sdk_core_test)

add_executable(fastflow_sdk_product_test product_test.cc)
target_link_libraries(fastflow_sdk_product_test PRIVATE fastflow_sdk_core)
add_test(NAME fastflow_sdk_product_test COMMAND fastflow_sdk_product_test)

add_library(fastflow_sdk_fake_hrx_missing SHARED fake_hrx_missing.cc)
include(GenerateExportHeader)
generate_export_header(
  fastflow_sdk_fake_hrx_missing
  BASE_NAME fake_hrx_missing
  EXPORT_FILE_NAME
    "${CMAKE_CURRENT_BINARY_DIR}/generated/fake_hrx_missing_export.h"
)
target_include_directories(
  fastflow_sdk_fake_hrx_missing
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
set_target_properties(
  fastflow_sdk_fake_hrx_missing
  PROPERTIES PREFIX "" OUTPUT_NAME "fake-hrx-missing"
)
add_library(fastflow_sdk_fake_hrx_complete SHARED fake_hrx_complete.cc)
target_include_directories(
  fastflow_sdk_fake_hrx_complete
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hrx/libhrx/include"
)
set_target_properties(
  fastflow_sdk_fake_hrx_complete
  PROPERTIES PREFIX "" OUTPUT_NAME "fake-hrx-complete"
)
add_library(fastflow_sdk_fake_hrx_incompatible SHARED fake_hrx_complete.cc)
target_include_directories(
  fastflow_sdk_fake_hrx_incompatible
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hrx/libhrx/include"
)
target_compile_definitions(
  fastflow_sdk_fake_hrx_incompatible PRIVATE FAKE_HRX_VERSION_MINOR=99
)
set_target_properties(
  fastflow_sdk_fake_hrx_incompatible
  PROPERTIES PREFIX "" OUTPUT_NAME "fake-hrx-incompatible"
)

add_executable(fastflow_sdk_hrx_loader_test hrx_loader_test.cc)
target_link_libraries(
  fastflow_sdk_hrx_loader_test PRIVATE fastflow_sdk_core fastflow_sdk_test_platform
)
add_test(
  NAME fastflow_sdk_hrx_loader_test
  COMMAND fastflow_sdk_hrx_loader_test
          $<TARGET_FILE:fastflow_sdk_fake_hrx_missing>
          $<TARGET_FILE:fastflow_sdk_fake_hrx_incompatible>
          $<TARGET_FILE:fastflow_sdk_fake_hrx_complete>
)

add_executable(fastflow_sdk_plugin_test plugin_test.cc)
target_link_libraries(
  fastflow_sdk_plugin_test PRIVATE fastflow_sdk_core fastflow_sdk_test_platform
)
add_test(NAME fastflow_sdk_plugin_test COMMAND fastflow_sdk_plugin_test)

add_executable(
  fastflow_sdk_fake_plugin fake_plugin.cc "${_fastflow_sdk_fake_plugin_main_source}"
)
target_compile_features(fastflow_sdk_fake_plugin PRIVATE cxx_std_17)
target_link_libraries(fastflow_sdk_fake_plugin PRIVATE fastflow_sdk_test_platform)
set_target_properties(
  fastflow_sdk_fake_plugin PROPERTIES OUTPUT_NAME "fastflow-echo"
)
add_executable(
  fastflow_sdk_plugin_dispatch_test
  plugin_dispatch_test.cc
  "${_fastflow_sdk_dispatch_platform_source}"
)
target_link_libraries(
  fastflow_sdk_plugin_dispatch_test PRIVATE fastflow_sdk_core fastflow_sdk_test_platform
)
add_test(
  NAME fastflow_sdk_plugin_dispatch_test
  COMMAND fastflow_sdk_plugin_dispatch_test
          $<TARGET_FILE:fastflow_sdk_fake_plugin>
          $<TARGET_FILE:fastflow_sdk_fake_hrx_complete>
)
add_test(
  NAME fastflow_sdk_plugin_stdio
  COMMAND fastflow --json echo --require-json
)
set_tests_properties(
  fastflow_sdk_plugin_stdio
  PROPERTIES
    ENVIRONMENT
      "FASTFLOW_SDK_PLUGIN_PATH=$<TARGET_FILE_DIR:fastflow_sdk_fake_plugin>"
    PASS_REGULAR_EXPRESSION
      "argument=--require-json;fake-plugin-stderr"
)
add_test(
  NAME fastflow_sdk_missing_plugin_json
  COMMAND
    "${CMAKE_COMMAND}"
    "-DCLI=$<TARGET_FILE:fastflow>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check_missing_plugin_json.cmake"
)

add_test(NAME fastflow_sdk_version_json COMMAND fastflow --json version)
add_test(NAME fastflow_sdk_probe_json COMMAND fastflow probe --json)
add_test(NAME fastflow_sdk_plugin_list_json COMMAND fastflow plugin list --json)
add_test(
  NAME fastflow_sdk_invalid_invocation_json
  COMMAND
    "${CMAKE_COMMAND}"
    "-DCLI=$<TARGET_FILE:fastflow>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check_invalid_invocation_json.cmake"
)
if(NOT WIN32)
  set_tests_properties(fastflow_sdk_plugin_list_json PROPERTIES ENVIRONMENT "PATH=")
endif()

if(WIN32)
  set(_fastflow_sdk_unicode_runtime_dir
      "${CMAKE_CURRENT_BINARY_DIR}/unicode-é")
  set(_fastflow_sdk_unicode_runtime
      "${_fastflow_sdk_unicode_runtime_dir}/fake-hrx-complete.dll")
  add_custom_command(
    TARGET fastflow_sdk_fake_hrx_complete POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${_fastflow_sdk_unicode_runtime_dir}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:fastflow_sdk_fake_hrx_complete>"
            "${_fastflow_sdk_unicode_runtime}"
  )
  add_test(
    NAME fastflow_sdk_unicode_argv
    COMMAND fastflow --json version --runtime
            "${_fastflow_sdk_unicode_runtime}"
  )
  add_executable(fastflow_sdk_windows_probe_test windows_probe_test.cc)
  target_link_libraries(fastflow_sdk_windows_probe_test PRIVATE fastflow_sdk_core)
  add_test(NAME fastflow_sdk_windows_probe_test COMMAND fastflow_sdk_windows_probe_test)
else()
  add_executable(fastflow_sdk_linux_probe_test linux_probe_test.cc)
  target_link_libraries(fastflow_sdk_linux_probe_test PRIVATE fastflow_sdk_core)
  add_test(NAME fastflow_sdk_linux_probe_test COMMAND fastflow_sdk_linux_probe_test)
endif()

# These tests intentionally use the standard assert() macro. Keep assertions
# active even when the product is configured as Release so CTest cannot pass
# without evaluating its checks.
set(_fastflow_sdk_assert_test_targets
  fastflow_sdk_core_test
  fastflow_sdk_product_test
  fastflow_sdk_hrx_loader_test
  fastflow_sdk_plugin_test
  fastflow_sdk_plugin_dispatch_test
  fastflow_sdk_windows_probe_test
  fastflow_sdk_linux_probe_test
)
foreach(_target IN LISTS _fastflow_sdk_assert_test_targets)
  if(TARGET "${_target}")
    if(MSVC)
      target_compile_options("${_target}" PRIVATE /UNDEBUG)
    else()
      target_compile_options("${_target}" PRIVATE -UNDEBUG)
    endif()
  endif()
endforeach()
