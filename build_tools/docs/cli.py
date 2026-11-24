# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The ``fastflow`` CLI reference and Troubleshooting pages.

Both come from the CLI's sources: the help text and built-in commands in
``tools/fastflow/src/command.cc``, the check IDs every command emits, and the
remediation for each ID in ``tools/fastflow/checks.toml``.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

CLI_SOURCE = Path("tools") / "fastflow" / "src"
COMMAND_SOURCE = CLI_SOURCE / "command.cc"
CHECKS = Path("tools") / "fastflow" / "checks.toml"

_HELP_TEXT = re.compile(r'std::string HelpText\(\)\s*\{\s*return R"\((.*?)\)";', re.S)
_BUILTINS = re.compile(r"kBuiltins\s*=\s*\{(.*?)\};", re.S)
_STRING = re.compile(r'"([^"]*)"')
# The ID that opens a check initializer, as in AddCheck(report, {"device.access",
# ...}) or checks.push_back({"device.access", ...}).
_CHECK_ID = re.compile(
    r'\b(?:AddCheck|push_back)\(\s*(?:\w+\s*,\s*)?\{\s*"([a-z_]+(?:\.[a-z_]+)+)"'
)


@dataclass(frozen=True)
class Check:
    id: str
    commands: tuple[str, ...]
    summary: str
    fix: str

    @property
    def anchor(self) -> str:
        return self.id.replace(".", "-").replace("_", "-")


@dataclass(frozen=True)
class Command:
    name: str
    synopsis: str
    description: str


def help_text(command_source: str) -> str:
    match = _HELP_TEXT.search(command_source)
    if match is None:
        raise ValueError("HelpText() raw string not found in command.cc")
    return match.group(1)


def builtin_commands(command_source: str) -> list[str]:
    match = _BUILTINS.search(command_source)
    if match is None:
        raise ValueError("kBuiltins not found in command.cc")
    return _STRING.findall(match.group(1))


def commands(command_source: str) -> list[Command]:
    """Each built-in command, in help order, with its synopsis and description."""
    lines = help_text(command_source).splitlines()
    start = lines.index("Commands:") + 1
    entries: list[list[str]] = []
    for line in lines[start:]:
        if not line.strip():
            break
        if line.startswith("    "):
            entries[-1].append(line.strip())
        else:
            entries.append([line.strip()])
    by_name: dict[str, Command] = {}
    for head, *rest in entries:
        synopsis, _, description = head.partition("  ")
        name = synopsis.split()[0]
        description = " ".join([description.strip(), *rest]).strip()
        by_name[name] = Command(name, synopsis.strip(), description)
    builtins = builtin_commands(command_source)
    missing = [name for name in builtins if name not in by_name]
    if missing:
        raise ValueError(f"built-in commands missing from HelpText(): {missing}")
    return [command for command in by_name.values() if command.name in builtins]


def emitted_check_ids(source_root: Path) -> set[str]:
    """Every check ID constructed anywhere in the CLI sources."""
    ids: set[str] = set()
    for path in sorted(source_root.rglob("*")):
        if path.suffix in {".cc", ".h"}:
            ids.update(_CHECK_ID.findall(path.read_text(encoding="utf-8")))
    return ids


def load_checks(path: Path) -> list[Check]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    checks = []
    for check_id, entry in data["checks"].items():
        unknown = set(entry) - {"commands", "summary", "fix"}
        if unknown:
            raise ValueError(f"{path}: {check_id} has unknown keys {sorted(unknown)}")
        checks.append(
            Check(
                id=check_id,
                commands=tuple(entry["commands"]),
                summary=entry["summary"].strip(),
                fix=entry["fix"].strip(),
            )
        )
    return checks


def _command_label(name: str) -> str:
    return "any command" if name == "*" else f"`fastflow {name}`"


def _check_links(checks: list[Check], command: str) -> str:
    ids = [c for c in checks if command in c.commands]
    if not ids:
        return "none"
    return ", ".join(f"[`{c.id}`](troubleshooting.md#{c.anchor})" for c in ids)


def cli_page(command_source: str, checks: list[Check]) -> str:
    rows = "\n".join(
        f"| `fastflow {c.synopsis}` | {c.description} | {_check_links(checks, c.name)} |"
        for c in commands(command_source)
    )
    always = ", ".join(
        f"[`{c.id}`](troubleshooting.md#{c.anchor})"
        for c in checks
        if "*" in c.commands
    )
    return f"""# `fastflow` CLI reference

`fastflow` is the native runtime CLI in every FastFlow SDK release: the
`fastflow-sdk` wheels put it on `PATH`, and the standalone archives ship it as
`bin/fastflow`. It needs no Python. This page is generated from the CLI's
source; [Runtime CLI behavior](../runtime-cli.md) explains what each command
does and does not change on the system.

## Help

```text
{help_text(command_source).rstrip()}
```

## Commands

| Command | Description | Checks in its report |
| --- | --- | --- |
{rows}

Every command can also report {always}.

## Exit status

| Status | Meaning |
| --- | --- |
| `0` | The command completed and every required check passed. |
| `1` | A health check or the self-test failed. |
| `2` | Invalid input or an internal execution error. |

## Machine-readable output

For a built-in command, `fastflow --json <command>` writes exactly one JSON
report to standard output, with `schema_version`, `command`, `status`,
`product`, `checks`, and, when the command has any, `data`. Invalid arguments
also produce a report. `--help` prints the usage text even with `--json`. An
external plugin writes its own output and sets the exit status; `fastflow`
reports only when it cannot start one (`plugin.dispatch`).

Parse the report instead of the human output; the
[JSON contract](../contracts/runtime-cli-json.md) defines it. For each check
whose `status` is not `pass`, look up its `id` in
[Troubleshooting](troubleshooting.md).

```bash
fastflow doctor --json
```
"""


def troubleshooting_page(checks: list[Check]) -> str:
    sections = []
    for check in checks:
        reported = ", ".join(_command_label(c) for c in check.commands)
        sections.append(
            f"## `{check.id}` {{#{check.anchor}}}\n\n"
            f"{check.summary}\n\n"
            f"Reported by: {reported}.\n\n"
            f"{check.fix}\n"
        )
    index = "\n".join(f"| [`{c.id}`](#{c.anchor}) | {c.summary} |" for c in checks)
    body = "\n".join(sections)
    return f"""# Troubleshooting

Start with the CLI. It reports every problem it finds as a check with a stable
ID:

```bash
fastflow doctor --json      # package, driver, device, and runtime health
fastflow self-test --json   # doctor, then a real dispatch on the NPU
```

If `pip install fastflow-sdk` fails with `No matching distribution found`,
the CLI is not installed yet; see
[If pip finds no matching distribution](../install.md#no-matching-distribution).

Find each check whose `status` is `fail` or `error` below. After installing or
changing an NPU driver, reboot before you debug further: a driver switch
without a reboot can leave the device unhealthy even though the operating
system reports the new driver.

| Check | What it verifies |
| --- | --- |
{index}

{body}"""
