#include "fake_hrx_missing_export.h"

extern "C" FAKE_HRX_MISSING_EXPORT int hrx_status_code(void*) { return 0; }
