# ROCm NPU external plugin API v1

`rocm-npu <name> ...` resolves and directly executes
`rocm-npu-<name>` (`rocm-npu-<name>.exe` on Windows). Built-in commands always
win. Names are limited to lowercase ASCII letters, digits, and dashes.

Resolution order is:

1. `ROCM_NPU_PLUGIN_PATH`
2. the directory containing the native `rocm-npu` executable
3. `PATH`

Empty path entries are ignored, so the current working directory is never
searched implicitly. An explicit current-directory entry is a deliberate trust
decision. Duplicate names use the first resolved executable. `plugin list`
reports later candidates as shadowed.

Plugins are child executables, not shared libraries. The core never invokes a
shell and does not execute Windows `.cmd` or `.bat` files. Standard input,
standard output, standard error, and normal process exit codes are preserved.
POSIX termination signals are forwarded to the child process group; Windows
console control propagation is best effort through the shared console.
An explicit `--runtime` is authoritative and must load successfully before the
plugin starts; it is never silently replaced by an installed default. Core-side
discovery, validation, and launch failures are emitted as one native report
when JSON output is requested.

The child receives:

- `ROCM_NPU_PLUGIN_API=1`
- `ROCM_NPU_CORE_VERSION`
- `ROCM_NPU_CORE_EXECUTABLE` as an absolute path
- `ROCM_NPU_RUNTIME_PATH` when a packaged or explicit HRX runtime resolves
- `ROCM_NPU_OUTPUT_FORMAT=human|json`

Plugins must reject absent or unsupported protocol versions. `--json` is a core
global option and is removed from plugin arguments; its canonical value is
communicated through `ROCM_NPU_OUTPUT_FORMAT`. Therefore `--json` before and
after the external command is equivalent.

Plugin search paths are executable-code trust boundaries. Administrators must
not place attacker-writable directories before trusted installation
directories. The core rejects a resolved plugin that aliases its own
executable.
