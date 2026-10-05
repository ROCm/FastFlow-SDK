# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
execute_process(
  COMMAND "${CLI}" Unknown --json
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)
if(NOT result EQUAL 2 OR NOT error STREQUAL "")
  message(FATAL_ERROR
          "invalid JSON invocation exited ${result}; stderr: ${error}")
endif()
string(JSON command ERROR_VARIABLE json_error GET "${output}" command)
string(JSON check_id ERROR_VARIABLE check_error GET "${output}" checks 0 id)
if(json_error OR check_error OR NOT command STREQUAL "invocation" OR
   NOT check_id STREQUAL "invocation.arguments")
  message(FATAL_ERROR "invalid invocation did not emit one JSON report: ${output}")
endif()
