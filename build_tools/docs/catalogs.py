# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The model and IRON operator catalog pages.

Models come from FastFlowLM's ``src/model_list.json``, the registry ``flm``
installs. Operators are read statically from IRON's ``iron/operators`` sources,
so the docs build needs neither mlir-aie nor an NPU.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

# The NPU2 (XDNA2) platform tag in FastFlowLM's model list.
NPU2_PLATFORM = "aie2p"
IRON_API = "https://amd.github.io/IRON/api/operators/"


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _model_capabilities(entry: dict) -> list[str]:
    capabilities = list(entry.get("label") or [])
    if entry.get("vlm"):
        capabilities.append("vision")
    if entry.get("asr"):
        capabilities.append("audio")
    details = entry.get("details") or {}
    if details.get("think_toggleable"):
        capabilities.append("thinking toggle")
    return sorted(set(capabilities))


def models_page(model_list: dict | None, revision: str | None) -> str:
    header = """# Models

FastFlowLM runs these models on NPU2 (XDNA2). Use the tag with `flm`:

```bash
flm pull qwen3:0.6b    # download
flm run qwen3:0.6b     # chat in the terminal; /bye exits
flm serve qwen3:0.6b   # OpenAI-compatible server, see Serve models
```

The registry is versioned with each release, so `flm list` is authoritative
for the release you installed. Benchmarks and model cards are on the
[FastFlowLM model pages](https://fastflowlm.com/docs/models/).
"""
    if model_list is None:
        return (
            header
            + "\n!!! note\n    This build of the documentation had no FastFlowLM "
            "model list. Run `flm list`.\n"
        )
    rows = []
    for family, sizes in model_list["models"].items():
        for size, entry in sizes.items():
            if NPU2_PLATFORM not in (entry.get("supported_platforms") or []):
                continue
            details = entry.get("details") or {}
            name = entry.get("name", "")
            url = entry.get("url", "")
            source = f"[{_cell(name)}]({url})" if url else _cell(name)
            capabilities = ", ".join(_model_capabilities(entry)) or ""
            context = entry.get("default_context_length", "")
            footprint = entry.get("footprint", "")
            rows.append(
                f"| `{family}:{size}` | {_cell(details.get('parameter_size', ''))} "
                f"| {_cell(details.get('quantization_level', ''))} | {context} "
                f"| {footprint} | {_cell(capabilities)} "
                f"| {_cell(entry.get('flm_min_version', ''))} | {source} |"
            )
    pin = f" at FastFlowLM `{revision[:12]}`" if revision else ""
    return (
        header + f"\nGenerated from `src/model_list.json`{pin}. Footprint is the "
        "approximate memory the model needs, in GB.\n\n"
        "| Tag | Parameters | Quantization | Default context | Footprint (GB) "
        "| Capabilities | Minimum flm | Weights |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n" + "\n".join(rows) + "\n"
    )


@dataclass
class _Class:
    name: str
    module: str
    path: Path
    node: ast.ClassDef
    bases: list[str]
    params: list[tuple[str, bool]] = field(default_factory=list)
    operands: list[tuple[str, str]] = field(default_factory=list)


def _call_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return ""


def _class_index(iron_root: Path) -> dict[str, list[_Class]]:
    index: dict[str, list[_Class]] = {}
    package = iron_root / "iron"
    for path in sorted(package.rglob("*.py")):
        if "tests" in path.relative_to(package).parts:
            continue
        module = ".".join(path.relative_to(iron_root).with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [
                b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                for b in node.bases
            ]
            entry = _Class(node.name, module, path, node, bases)
            for statement in node.body:
                if isinstance(statement, ast.AnnAssign) and isinstance(
                    statement.target, ast.Name
                ):
                    target, value = statement.target.id, statement.value
                elif (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                ):
                    target, value = statement.targets[0].id, statement.value
                else:
                    continue
                kind = _call_name(value)
                if kind == "param":
                    assert isinstance(value, ast.Call)
                    required = not any(k.arg == "default" for k in value.keywords)
                    entry.params.append((target, required))
                elif kind in {"In", "Out", "InOut"}:
                    entry.operands.append((target, kind))
            index.setdefault(node.name, []).append(entry)
    return index


def _resolve(index: dict[str, list[_Class]], name: str, near: str) -> _Class | None:
    candidates = index.get(name) or []
    for candidate in candidates:
        if candidate.module == near:
            return candidate
    for candidate in candidates:
        if candidate.module.startswith("iron.common"):
            return candidate
    return candidates[0] if candidates else None


def _members(
    index, cls: _Class
) -> tuple[list[tuple[str, bool]], list[tuple[str, str]]]:
    """Params and operands of ``cls`` and its bases, subclass declarations last."""
    params: dict[str, bool] = {}
    operands: dict[str, str] = {}
    seen: set[int] = set()

    def visit(current: _Class) -> None:
        if id(current) in seen:
            return
        seen.add(id(current))
        for base in current.bases:
            resolved = _resolve(index, base, current.module)
            if resolved is not None:
                visit(resolved)
        params.update(dict(current.params))
        operands.update(dict(current.operands))

    visit(cls)
    return list(params.items()), list(operands.items())


def _exports(init: Path, iron_root: Path) -> list[tuple[str, str]]:
    """``(name, module)`` for every name in the package's ``__all__``."""
    tree = ast.parse(init.read_text(encoding="utf-8"))
    imported: dict[str, str] = {}
    exported: list[str] = []
    package = ".".join(init.parent.relative_to(iron_root).parts)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                imported[alias.asname or alias.name] = f"{package}.{node.module}"
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            exported = [ast.literal_eval(e) for e in node.value.elts]  # type: ignore[attr-defined]
    return [(name, imported[name]) for name in exported if name in imported]


def _summary(node: ast.ClassDef) -> str:
    doc = ast.get_docstring(node) or ""
    return " ".join(doc.split("\n\n")[0].split())


_OPERATORS_INTRO = """# IRON operators

IRON is the operator and graph library on top of MLIR-AIE's `aie.iron`, and
`fastflow-sdk` installs it with the compiler and the HRX runtime. An operator,
such as `GEMM`, `RMSNorm` or `MHA`, is a class you construct by keyword. Called
on device tensors, it compiles once and runs on the NPU; compiled images are
cached by content under `NPU_CACHE_HOME` (default `~/.npu/cache`). A graph, a
subclass of `iron.Graph`, records the steps of several operators and compiles
them into one image, which is how `iron.lm` runs language models.
"""

_RUN_AN_OPERATOR = """## Run an operator

This is the first example of IRON's programming guide{where}:

```python
{example}```

- Construction declares the problem and touches no device. Parameters such as
  `size` fix the shape. Tunables such as `num_aie_columns` and `tile_size`
  choose how the work spreads over the array; leave them out and
  `op.resolved()` shows what the operator picks for your device.
- A call takes one device tensor per declared buffer, in the order
  `op.buffers` lists them. `aie.utils.DEFAULT_TENSOR_CLASS` is the device
  tensor type of the host runtime: importing `aie` with `fastflow-sdk`
  installed selects HRX unless `NPU_RUNTIME` says otherwise. Wrap a NumPy
  array to upload it, or give a shape and dtype for a buffer the NPU writes.
- `op.reference()` computes the expected result with NumPy, and
  `op.resolved().tolerance()` says how close the NPU must come.

## Compile without an NPU

Bind the device before building anything, then compile. Only calling an
operator needs an NPU:

```python
import aie.utils as aie_utils
from aie.iron.device import from_name
from iron.operators import AXPY

aie_utils.set_current_device(from_name("npu2", n_cols=8))
artifacts = AXPY(size=2048, scalar_factor=1.5).compile()
```
"""


def _guide_sections(iron_root: Path, base: str) -> str:
    """The programming guide's section list, linked at ``base``."""
    readme = iron_root / "programming_guide" / "README.md"
    if not readme.is_file():
        return ""
    text = readme.read_text(encoding="utf-8").partition("\n## Sections\n")[2]
    links = re.findall(r"^\* \[([^\]]+)\]\(\./(section-\d+)/\)", text, re.MULTILINE)
    return "".join(f"- [{title}]({base}/{section})\n" for title, section in links)


def operators_page(iron_root: Path | None, revision: str | None) -> str:
    ref = revision or "devel"
    guide = f"https://github.com/amd/IRON/tree/{ref}/programming_guide"
    agents = f"https://github.com/amd/IRON/blob/{ref}/AGENTS.md"
    example = (
        iron_root / "programming_guide" / "section-1" / "axpy_example.py"
        if iron_root
        else None
    )
    page = _OPERATORS_INTRO
    if example is not None and example.is_file():
        page += "\n" + _RUN_AN_OPERATOR.format(
            where=f", [section 1]({guide}/section-1)",
            example=example.read_text(encoding="utf-8"),
        )
    further = (_guide_sections(iron_root, guide) if iron_root else "") or (
        f"- [IRON programming guide]({guide})\n"
    )
    page += (
        "\n## Go further\n\nIRON's programming guide, at the revision this SDK "
        "ships, continues from one operator to graphs and models:\n\n"
        + further
        + f"\nThe [IRON API reference]({IRON_API}) documents every field of every "
        "operator, including the tunables. Agents writing or changing operators "
        f"should follow IRON's [AGENTS.md]({agents}).\n"
    )
    if iron_root is None:
        return (
            page + "\n!!! note\n    This build of the documentation had no IRON "
            "source tree. See the IRON API reference.\n"
        )
    index = _class_index(iron_root)
    operators_root = iron_root / "iron" / "operators"
    sections = []
    for package, title in (
        (operators_root, "Operators"),
        (operators_root / "flm", "FastFlowLM ports (`iron.operators.flm`)"),
    ):
        rows = []
        for name, module in _exports(package / "__init__.py", iron_root):
            cls = _resolve(index, name, module)
            if cls is None:
                continue
            params, operands = _members(index, cls)
            required = ", ".join(f"`{p}`" for p, needed in params if needed)
            inputs = ", ".join(f"`{o}`" for o, kind in operands if kind != "Out")
            outputs = ", ".join(f"`{o}`" for o, kind in operands if kind != "In")
            source = cls.path.relative_to(iron_root).as_posix()
            line = cls.node.lineno
            qualified = (
                f"{package.relative_to(iron_root).as_posix().replace('/', '.')}.{name}"
            )
            rows.append(
                f"| [`{name}`]({IRON_API}#{qualified}) | {_cell(_summary(cls.node))} "
                f"| {required} | {inputs} | {outputs} "
                f"| [source](https://github.com/amd/IRON/blob/{ref}/{source}#L{line}) |"
            )
        sections.append(
            f"## {title}\n\n"
            "| Operator | Description | Required parameters | Inputs | Outputs "
            "| Source |\n| --- | --- | --- | --- | --- | --- |\n"
            + "\n".join(rows)
            + "\n"
        )
    pin = f" at IRON `{revision[:12]}`" if revision else ""
    return (
        page + f"\nThe tables are generated from `iron/operators`{pin}, the revision "
        "this SDK ships.\n\n" + "\n".join(sections)
    )
