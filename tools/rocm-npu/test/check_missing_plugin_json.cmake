# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Confirms that missing-plugin failures produce one valid JSON report.

execute_process(
  COMMAND "${CLI}" --json missing
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)
if(NOT result EQUAL 2)
  message(FATAL_ERROR "missing plugin exited ${result}; stderr: ${error}")
endif()
string(JSON check_id ERROR_VARIABLE json_error GET "${output}" checks 0 id)
if(json_error OR NOT check_id STREQUAL "plugin.dispatch")
  message(FATAL_ERROR "missing plugin did not emit dispatch JSON: ${output}")
endif()
