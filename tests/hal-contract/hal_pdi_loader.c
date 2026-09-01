// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// HAL artifact-contract smoke: prove the HRX amdxdna HAL consumes the
// hrx-xclbinutil artifact. Reads a real .xclbin (PDI embedded in AIE_PARTITION
// by hrx-xclbinutil) and calls the HAL's own loader entrypoint
// iree_hal_amdxdna_xclbin_extract_pdi, then (optionally) diffs the extracted
// PDI against the bootgen-produced main.pdi.
//
// No NPU needed: this validates the artifact contract (parse/extract), not
// device submission. Usage: hal_pdi_loader <file.xclbin> [expected_main.pdi]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdxdna/xclbin_util.h"

// Minimal malloc-backed iree_allocator_t (the build doesn't define
// IREE_ALLOCATOR_SYSTEM_CTL, so ws_d_allocator() is unavailable here).
static iree_status_t ws_d_malloc_ctl(void* self,
                                     iree_allocator_command_t command,
                                     const void* params, void** inout_ptr) {
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC: {
      const iree_allocator_alloc_params_t* p =
          (const iree_allocator_alloc_params_t*)params;
      void* m = (command == IREE_ALLOCATOR_COMMAND_CALLOC)
                    ? calloc(1, p->byte_length)
                    : malloc(p->byte_length);
      if (!m) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
      *inout_ptr = m;
      return iree_ok_status();
    }
    case IREE_ALLOCATOR_COMMAND_REALLOC: {
      const iree_allocator_alloc_params_t* p =
          (const iree_allocator_alloc_params_t*)params;
      void* m = realloc(*inout_ptr, p->byte_length);
      if (!m) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
      *inout_ptr = m;
      return iree_ok_status();
    }
    case IREE_ALLOCATOR_COMMAND_FREE:
      free(*inout_ptr);
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
  }
}
static iree_allocator_t ws_d_allocator(void) {
  iree_allocator_t a = {NULL, ws_d_malloc_ctl};
  return a;
}

static int read_file(const char* path, uint8_t** out, size_t* out_size) {
  FILE* f = fopen(path, "rb");
  if (!f) return -1;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t* buf = (uint8_t*)malloc((size_t)n);
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    free(buf);
    return -1;
  }
  fclose(f);
  *out = buf;
  *out_size = (size_t)n;
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <file.xclbin> [expected_main.pdi]\n", argv[0]);
    return 2;
  }

  uint8_t* xclbin = NULL;
  size_t xclbin_size = 0;
  if (read_file(argv[1], &xclbin, &xclbin_size) != 0) {
    fprintf(stderr, "cannot read %s\n", argv[1]);
    return 2;
  }
  printf("xclbin: %s (%zu bytes)\n", argv[1], xclbin_size);

  iree_const_byte_span_t span = iree_make_const_byte_span(xclbin, xclbin_size);
  iree_byte_span_t pdi = iree_byte_span_empty();
  iree_status_t status = iree_hal_amdxdna_xclbin_extract_pdi(
      span, /*pdi_index=*/0, ws_d_allocator(), &pdi);
  if (!iree_status_is_ok(status)) {
    char buf[512];
    iree_host_size_t len = 0;
    iree_status_format(status, sizeof(buf), buf, &len);
    fprintf(stderr, "extract_pdi FAILED: %.*s\n", (int)len, buf);
    iree_status_ignore(status);
    return 1;
  }
  printf("extract_pdi OK: PDI #0 = %zu bytes; first bytes:",
         (size_t)pdi.data_length);
  for (int i = 0; i < 8 && i < (int)pdi.data_length; ++i)
    printf(" %02x", pdi.data[i]);
  printf("\n");

  int rc = 0;
  if (argc >= 3) {
    uint8_t* exp = NULL;
    size_t exp_size = 0;
    if (read_file(argv[2], &exp, &exp_size) == 0) {
      if (exp_size == pdi.data_length && memcmp(exp, pdi.data, exp_size) == 0) {
        printf("PDI matches bootgen %s BYTE-FOR-BYTE (%zu bytes)\n", argv[2],
               exp_size);
      } else {
        printf("PDI DIFFERS from %s (extracted=%zu expected=%zu)\n", argv[2],
               (size_t)pdi.data_length, exp_size);
        rc = 1;
      }
      free(exp);
    }
  }

  iree_allocator_free(ws_d_allocator(), pdi.data);
  free(xclbin);
  return rc;
}
