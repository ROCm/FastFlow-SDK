# Native runtime CLI JSON contract

`rocm-npu --json <command>` writes exactly one UTF-8 JSON document followed by
a newline to standard output. Diagnostics that are not part of the report are
written to standard error. A failed health check or self-test still produces a
complete report. If `--json` is present but invocation parsing fails, the CLI
emits an `invocation` report containing `invocation.arguments` and exits 2.

Consumers must use `schema_version` before interpreting a report. Version 1
has a stable envelope with these top-level members:

- `schema_version`: integer, currently `1`
- `command`: command name
- `status`: `pass`, `fail`, or `error`
- `product`: object with `version`, `source_revision`, and `build_platform`
- `checks`: array of check result objects
- `data`: optional command-specific object

Each check result has `id`, `status`, and `message` fields. Check IDs are
lowercase dotted identifiers such as `runtime.library.load`. Checks may include
a `details` object for structured diagnostics.

Adding optional fields inside `data` or check `details` is compatible. New
commands and new check IDs are compatible when existing command behavior is
unchanged. Removing or renaming required envelope fields, changing their types,
or changing status semantics requires a new `schema_version`.

Changing or removing an existing check ID is breaking for automation and must be
handled as an explicit compatibility change, but does not by itself require a
new envelope schema version.

The order of JSON object members is not part of the contract.

## Common report

```json
{
  "schema_version": 1,
  "command": "doctor",
  "status": "pass",
  "product": {
    "version": "0.1.0.dev0",
    "source_revision": "0123456789abcdef",
    "build_platform": "Linux-x86_64"
  },
  "checks": []
}
```

Statuses are `pass`, `fail`, or `error`. `fail` means the command completed but
a health assertion or output comparison failed. `error` means the invocation
could not be completed because its input or execution was invalid.

Stable doctor check IDs include:

- `package.manifest`
- `device.enumeration`
- `runtime.library.present`
- `runtime.library.load`
- `runtime.symbols`
- `runtime.abi`
- `device.present`
- `device.access`
- `device.ioctl`
- `runtime.initialize`
- `runtime.device_count`
- `runtime.shutdown`

Self-test adds `self_test.prerequisites`, `self_test.fixture`, `self_test.load`,
`self_test.dispatch`, `self_test.cleanup`, and `self_test.output`. Plugin
discovery uses `plugin.discovery`. Unhandled execution failures use
`internal.error`.
