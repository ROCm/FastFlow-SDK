# Native ROCm NPU runtime CLI

`rocm-npu` provides runtime discovery, health checks, NPU power-mode
configuration, and a bounded device self-test on Linux and Windows. The
executable is C++17 and does not require Python. HRX is loaded only when a
command needs it.

## Commands and exit status

- `rocm-npu version` reports build identity, package-manifest identity, and HRX
  discovery without initializing HRX or requiring hardware.
- `rocm-npu probe` reports operating-system, driver, firmware, device, and HRX
  facts without applying product-support policy. It reports the current NPU
  power mode when the platform supports it.
- `rocm-npu powermode` reports the current NPU power mode for detected NPUs.
- `rocm-npu powermode <default|powersaver|balanced|performance|turbo>` sets the
  NPU power mode and then re-queries the device to confirm the requested mode.
- `rocm-npu doctor` evaluates package, HRX, and device health. Release package
  manifests verify every declared file's size and SHA-256 digest. It initializes
  and shuts down HRX but does not dispatch an executable.
- `rocm-npu self-test` runs doctor checks and then dispatches the packaged NPU2
  integer XADX fixture. NPU1 devices remain supported by probe and doctor, but
  self-test reports that no validated NPU1 fixture is provided. Optional
  diagnostic ioctl failures do
  not prevent this authoritative HRX dispatch test. Output bytes are compared
  exactly.
- `rocm-npu plugin list` discovers `rocm-npu-*` executables from
  `ROCM_NPU_PLUGIN_PATH`, the CLI directory, and `PATH`. It does not execute
  plugin code.

Exit status `0` means the command completed and required checks passed, `1`
means a health check or test failed, and `2` means invalid input or an internal
execution error.

Global options may precede or follow the command:

```text
--json
--runtime <path>
--verbose
--help
```

The [versioned JSON contract](contracts/runtime-cli-json.md) is intended for
automation. Human output is informational and is not a stable parsing surface.

## Non-mutating behavior

Probe, doctor, and `powermode` without a mode never create device nodes, change
device permissions, modify drivers or firmware, alter process/system limits, or
change NPU power mode. They only inspect existing package, runtime, driver,
firmware, and device state.

`rocm-npu powermode <mode>` reports success only when a subsequent query
confirms every reported NPU reached the requested mode. Some host drivers
require elevated privileges to set power mode; permission failures are reported
as command failures. Non-mutating power-mode reporting remains usable when the
platform permits device access.

## HRX discovery

Without an explicit override, the loader tries:

1. `HRX_LIBHRX`
2. the packaged path relative to the executable
3. the normal operating-system library search path

`--runtime` is authoritative: if that candidate cannot be loaded or is
incompatible, the command reports the failure instead of silently selecting a
different runtime. It may name the library or its containing directory. The packaged
layouts are `bin/rocm-npu` plus `lib/libhrx.so.0` on Linux and
`bin/rocm-npu.exe` plus `bin/hrx.dll` on Windows. Missing required public C
symbols are reported individually. The loader also requires the runtime's HRX
major and minor ABI version to match the headers used to build the CLI.

## Packaging

The runtime wheel contains a small Python console launcher that locates and
executes the packaged native binary and returns its exit status.
`python -m rocm_npu_runtime` uses the same path. Native ZIP installations run
`bin/rocm-npu[.exe]` directly and require no Python installation.

The self-test fixture is immutable product data under
`share/rocm-npu/self-test`. The XADX is checked in because the runtime CLI must
self-test without a compiler or Python installation; deterministic input and
expected output bytes are generated from the manifest. Updating the XADX or
generated buffer contract requires a deliberate compiler/runtime compatibility
review and updated producer provenance. Before dispatch, the CLI validates every
fixture hash, buffer dtype/shape/size, dispatch geometry, and the loaded export
metadata against the manifest.

Legacy XADX files that do not carry a reliable export binding count record that
field as `null`; the CLI still validates buffer count, order, dtype, shape, and
byte length from the fixture descriptors.
External commands are resolved through the versioned executable plugin
contract in `docs/contracts/plugin-api-v1.md`. The optional
`rocm-npu-qualification` wheel provides `rocm-npu qualify`; see
`docs/qualification.md`.
