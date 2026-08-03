#ifdef _WIN32
#define TEST_EXPORT __declspec(dllexport)
#else
#define TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" TEST_EXPORT int hrx_status_code(void*) { return 0; }
