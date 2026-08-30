// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#define HRX_BUILDING_SHARED
#include <hrx_runtime.h>

#ifndef FAKE_HRX_VERSION_MINOR
#define FAKE_HRX_VERSION_MINOR HRX_VERSION_MINOR
#endif

extern "C" {

HRX_API void hrx_runtime_version(int* major, int* minor, int* patch) {
  *major = HRX_VERSION_MAJOR;
  *minor = FAKE_HRX_VERSION_MINOR;
  *patch = HRX_VERSION_PATCH;
}
HRX_API hrx_status_code_t hrx_status_code(hrx_status_t) {
  return HRX_STATUS_OK;
}
HRX_API hrx_status_t hrx_status_to_string(hrx_status_t, char**,
                                          size_t*) {
  return nullptr;
}
HRX_API void hrx_status_free_message(char*) {}
HRX_API void hrx_status_ignore(hrx_status_t) {}
HRX_API hrx_status_t hrx_gpu_initialize(uint32_t) { return nullptr; }
HRX_API hrx_status_t hrx_gpu_shutdown() { return nullptr; }
HRX_API hrx_status_t hrx_gpu_device_count(int*) { return nullptr; }
HRX_API hrx_status_t hrx_gpu_device_get(int, hrx_device_t*) {
  return nullptr;
}
HRX_API void hrx_device_release(hrx_device_t) {}
HRX_API hrx_status_t hrx_stream_create(hrx_device_t, uint32_t,
                                       hrx_stream_t*) {
  return nullptr;
}
HRX_API void hrx_stream_release(hrx_stream_t) {}
HRX_API hrx_status_t hrx_stream_synchronize(hrx_stream_t) {
  return nullptr;
}
HRX_API hrx_status_t hrx_buffer_allocate(hrx_stream_t, size_t,
                                         hrx_memory_type_t,
                                         hrx_buffer_usage_t, hrx_buffer_t*) {
  return nullptr;
}
HRX_API hrx_status_t hrx_buffer_map_with_mode(
    hrx_buffer_t, hrx_mapping_mode_t, hrx_map_flags_t, size_t, size_t,
    void**) {
  return nullptr;
}
HRX_API hrx_status_t hrx_buffer_unmap(hrx_buffer_t) { return nullptr; }
HRX_API hrx_status_t hrx_buffer_flush_range(hrx_buffer_t, size_t,
                                            size_t) {
  return nullptr;
}
HRX_API hrx_status_t hrx_buffer_invalidate_range(hrx_buffer_t, size_t,
                                                 size_t) {
  return nullptr;
}
HRX_API void hrx_buffer_release(hrx_buffer_t) {}
HRX_API hrx_status_t hrx_executable_load_data(
    hrx_device_t, const void*, size_t, const char*, const char*,
    hrx_executable_t*) {
  return nullptr;
}
HRX_API hrx_status_t hrx_executable_export_info(
    hrx_executable_t, uint32_t, hrx_executable_export_info_t*) {
  return nullptr;
}
HRX_API hrx_status_t hrx_executable_lookup_export_by_name(
    hrx_executable_t, const char*, uint32_t*) {
  return nullptr;
}
HRX_API void hrx_executable_release(hrx_executable_t) {}
HRX_API hrx_status_t hrx_stream_dispatch(
    hrx_stream_t, hrx_executable_t, uint32_t,
    const hrx_dispatch_config_t*, const void*, size_t,
    const hrx_buffer_ref_t*, size_t, uint32_t) {
  return nullptr;
}

}  // extern "C"
