if(WIN32)
  set(_rocm_npu_test_platform_source platform_windows.cc)
  set(_rocm_npu_fake_plugin_main_source fake_plugin_main_windows.cc)
  set(_rocm_npu_dispatch_platform_source plugin_dispatch_windows.cc)
else()
  set(_rocm_npu_test_platform_source platform_posix.cc)
  set(_rocm_npu_fake_plugin_main_source fake_plugin_main_posix.cc)
  set(_rocm_npu_dispatch_platform_source plugin_dispatch_posix.cc)
endif()

add_library(rocm_npu_test_platform STATIC "${_rocm_npu_test_platform_source}")
target_compile_features(rocm_npu_test_platform PUBLIC cxx_std_17)

add_executable(rocm_npu_core_test core_test.cc)
target_link_libraries(rocm_npu_core_test PRIVATE rocm_npu_core)
add_test(NAME rocm_npu_core_test COMMAND rocm_npu_core_test)

add_executable(rocm_npu_product_test product_test.cc)
target_link_libraries(rocm_npu_product_test PRIVATE rocm_npu_core)
add_test(NAME rocm_npu_product_test COMMAND rocm_npu_product_test)

add_library(rocm_npu_fake_hrx_missing SHARED fake_hrx_missing.cc)
include(GenerateExportHeader)
generate_export_header(
  rocm_npu_fake_hrx_missing
  BASE_NAME fake_hrx_missing
  EXPORT_FILE_NAME
    "${CMAKE_CURRENT_BINARY_DIR}/generated/fake_hrx_missing_export.h"
)
target_include_directories(
  rocm_npu_fake_hrx_missing
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
set_target_properties(
  rocm_npu_fake_hrx_missing
  PROPERTIES PREFIX "" OUTPUT_NAME "fake-hrx-missing"
)
add_library(rocm_npu_fake_hrx_complete SHARED fake_hrx_complete.cc)
target_include_directories(
  rocm_npu_fake_hrx_complete
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hrx/libhrx/include"
)
set_target_properties(
  rocm_npu_fake_hrx_complete
  PROPERTIES PREFIX "" OUTPUT_NAME "fake-hrx-complete"
)
add_library(rocm_npu_fake_hrx_incompatible SHARED fake_hrx_complete.cc)
target_include_directories(
  rocm_npu_fake_hrx_incompatible
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hrx/libhrx/include"
)
target_compile_definitions(
  rocm_npu_fake_hrx_incompatible PRIVATE FAKE_HRX_VERSION_MINOR=99
)
set_target_properties(
  rocm_npu_fake_hrx_incompatible
  PROPERTIES PREFIX "" OUTPUT_NAME "fake-hrx-incompatible"
)

add_executable(rocm_npu_hrx_loader_test hrx_loader_test.cc)
target_link_libraries(
  rocm_npu_hrx_loader_test PRIVATE rocm_npu_core rocm_npu_test_platform
)
add_test(
  NAME rocm_npu_hrx_loader_test
  COMMAND rocm_npu_hrx_loader_test
          $<TARGET_FILE:rocm_npu_fake_hrx_missing>
          $<TARGET_FILE:rocm_npu_fake_hrx_incompatible>
          $<TARGET_FILE:rocm_npu_fake_hrx_complete>
)

add_executable(rocm_npu_plugin_test plugin_test.cc)
target_link_libraries(
  rocm_npu_plugin_test PRIVATE rocm_npu_core rocm_npu_test_platform
)
add_test(NAME rocm_npu_plugin_test COMMAND rocm_npu_plugin_test)

add_executable(
  rocm_npu_fake_plugin fake_plugin.cc "${_rocm_npu_fake_plugin_main_source}"
)
target_compile_features(rocm_npu_fake_plugin PRIVATE cxx_std_17)
target_link_libraries(rocm_npu_fake_plugin PRIVATE rocm_npu_test_platform)
set_target_properties(
  rocm_npu_fake_plugin PROPERTIES OUTPUT_NAME "rocm-npu-echo"
)
add_executable(
  rocm_npu_plugin_dispatch_test
  plugin_dispatch_test.cc
  "${_rocm_npu_dispatch_platform_source}"
)
target_link_libraries(
  rocm_npu_plugin_dispatch_test PRIVATE rocm_npu_core rocm_npu_test_platform
)
add_test(
  NAME rocm_npu_plugin_dispatch_test
  COMMAND rocm_npu_plugin_dispatch_test
          $<TARGET_FILE:rocm_npu_fake_plugin>
          $<TARGET_FILE:rocm_npu_fake_hrx_complete>
)
add_test(
  NAME rocm_npu_plugin_stdio
  COMMAND rocm-npu --json echo --require-json
)
set_tests_properties(
  rocm_npu_plugin_stdio
  PROPERTIES
    ENVIRONMENT
      "ROCM_NPU_PLUGIN_PATH=$<TARGET_FILE_DIR:rocm_npu_fake_plugin>"
    PASS_REGULAR_EXPRESSION
      "argument=--require-json;fake-plugin-stderr"
)
add_test(
  NAME rocm_npu_missing_plugin_json
  COMMAND
    "${CMAKE_COMMAND}"
    "-DCLI=$<TARGET_FILE:rocm-npu>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check_missing_plugin_json.cmake"
)

add_test(NAME rocm_npu_version_json COMMAND rocm-npu --json version)
add_test(NAME rocm_npu_probe_json COMMAND rocm-npu probe --json)
add_test(NAME rocm_npu_plugin_list_json COMMAND rocm-npu plugin list --json)
add_test(
  NAME rocm_npu_invalid_invocation_json
  COMMAND
    "${CMAKE_COMMAND}"
    "-DCLI=$<TARGET_FILE:rocm-npu>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check_invalid_invocation_json.cmake"
)
if(NOT WIN32)
  set_tests_properties(rocm_npu_plugin_list_json PROPERTIES ENVIRONMENT "PATH=")
endif()

if(WIN32)
  set(_rocm_npu_unicode_runtime_dir
      "${CMAKE_CURRENT_BINARY_DIR}/unicode-é")
  set(_rocm_npu_unicode_runtime
      "${_rocm_npu_unicode_runtime_dir}/fake-hrx-complete.dll")
  add_custom_command(
    TARGET rocm_npu_fake_hrx_complete POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${_rocm_npu_unicode_runtime_dir}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:rocm_npu_fake_hrx_complete>"
            "${_rocm_npu_unicode_runtime}"
  )
  add_test(
    NAME rocm_npu_unicode_argv
    COMMAND rocm-npu --json version --runtime
            "${_rocm_npu_unicode_runtime}"
  )
  add_executable(rocm_npu_windows_probe_test windows_probe_test.cc)
  target_link_libraries(rocm_npu_windows_probe_test PRIVATE rocm_npu_core)
  add_test(NAME rocm_npu_windows_probe_test COMMAND rocm_npu_windows_probe_test)
else()
  add_executable(rocm_npu_linux_probe_test linux_probe_test.cc)
  target_link_libraries(rocm_npu_linux_probe_test PRIVATE rocm_npu_core)
  add_test(NAME rocm_npu_linux_probe_test COMMAND rocm_npu_linux_probe_test)
endif()

# These tests intentionally use the standard assert() macro. Keep assertions
# active even when the product is configured as Release so CTest cannot pass
# without evaluating its checks.
set(_rocm_npu_assert_test_targets
  rocm_npu_core_test
  rocm_npu_product_test
  rocm_npu_hrx_loader_test
  rocm_npu_plugin_test
  rocm_npu_plugin_dispatch_test
  rocm_npu_windows_probe_test
  rocm_npu_linux_probe_test
)
foreach(_target IN LISTS _rocm_npu_assert_test_targets)
  if(TARGET "${_target}")
    if(MSVC)
      target_compile_options("${_target}" PRIVATE /UNDEBUG)
    else()
      target_compile_options("${_target}" PRIVATE -UNDEBUG)
    endif()
  endif()
endforeach()
