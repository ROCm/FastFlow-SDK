# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The documentation site stays in step with the product it documents."""

from __future__ import annotations

import ast
import fnmatch
import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path, PurePosixPath

import pytest

from build_tools.docs import catalogs, cli, pages, releases, versions

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
COMMAND_SOURCE = (ROOT / cli.COMMAND_SOURCE).read_text(encoding="utf-8")
CHECKS = cli.load_checks(ROOT / cli.CHECKS)
SITE = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))["project"]
_SNIPPET = re.compile(r'^\s*--8<--\s+"([^":]+)(?::([\w-]+))?"\s*$', re.M)
_LINK = re.compile(
    r"\]\(<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\)|^\s*\[[^\]]+\]:\s*<?(\S+?)>?\s*$", re.M
)
_FENCE = re.compile(r"^\s*(```|~~~).*?^\s*\1", re.M | re.S)

pytestmark = pytest.mark.presubmit


def _nav_pages(nav) -> list[str]:
    if isinstance(nav, str):
        return [nav]
    if isinstance(nav, dict):
        return [page for value in nav.values() for page in _nav_pages(value)]
    return [page for item in nav for page in _nav_pages(item)]


def _site_pages() -> set[str]:
    written = {p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.md")}
    return written | set(pages.generate(pages.Inputs(root=ROOT)))


def _serve_runner():
    path = ROOT / "tests" / "fastflowlm" / "run_serve_examples.py"
    spec = importlib.util.spec_from_file_location("run_serve_examples", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses look the module up while the class is being created.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def test_every_emitted_check_id_has_troubleshooting_and_no_stale_ones():
    emitted = cli.emitted_check_ids(ROOT / cli.CLI_SOURCE)
    documented = {check.id for check in CHECKS}
    assert emitted, "no check IDs found; the extraction pattern no longer matches"
    assert sorted(emitted - documented) == [], "add these to tools/fastflow/checks.toml"
    assert sorted(documented - emitted) == [], "no longer emitted by tools/fastflow/src"


def test_json_contract_names_every_check_id():
    contract = (DOCS / "contracts" / "runtime-cli-json.md").read_text(encoding="utf-8")
    assert [c.id for c in CHECKS if f"`{c.id}`" not in contract] == []


def test_checks_name_only_real_commands():
    known = set(cli.builtin_commands(COMMAND_SOURCE)) | {"*"}
    for check in CHECKS:
        assert check.commands, check.id
        assert set(check.commands) <= known, check.id
        assert check.summary and check.fix, check.id


def test_cli_reference_comes_from_the_help_text():
    commands = {c.name: c for c in cli.commands(COMMAND_SOURCE)}
    assert set(commands) == set(cli.builtin_commands(COMMAND_SOURCE))
    assert commands["powermode"].synopsis == "powermode [mode]"
    assert commands["plugin"].synopsis == "plugin list"
    assert commands["doctor"].description.startswith("Evaluate")
    page = cli.cli_page(COMMAND_SOURCE, CHECKS)
    assert cli.help_text(COMMAND_SOURCE).strip() in page
    for check in CHECKS:
        assert f"troubleshooting.md#{check.anchor}" in page


def test_troubleshooting_anchors_are_unique():
    anchors = [check.anchor for check in CHECKS]
    assert len(anchors) == len(set(anchors))
    page = cli.troubleshooting_page(CHECKS)
    for check in CHECKS:
        assert f"## `{check.id}` {{#{check.anchor}}}" in page


def test_every_snippet_include_resolves():
    for page in DOCS.rglob("*.md"):
        for path, name in _SNIPPET.findall(page.read_text(encoding="utf-8")):
            target = ROOT / path
            assert target.is_file(), f"{page.relative_to(ROOT)} includes {path}"
            if name:
                text = target.read_text(encoding="utf-8")
                assert f"--8<-- [start:{name}]" in text, f"{path}:{name}"
                assert f"--8<-- [end:{name}]" in text, f"{path}:{name}"


def test_every_published_example_is_shown_in_the_docs():
    included = {
        path
        for page in DOCS.rglob("*.md")
        for path, _ in _SNIPPET.findall(page.read_text(encoding="utf-8"))
    }
    examples = pages.examples(ROOT)
    assert "examples/hello_npu.py" in examples
    assert {"examples/serve/chat.py", "examples/serve/tool_call.py"} <= set(examples)
    assert sorted(set(examples) - included) == []


def test_ci_runs_every_serve_example_as_its_docstring_says():
    runner = _serve_runner()
    commands = {
        script.name: server.args
        for server in runner.SERVERS
        for script in server.scripts
    }
    examples = sorted(p.name for p in runner.EXAMPLES.glob("*.py"))
    assert not set(commands) & set(runner.NOT_RUN)
    assert sorted({*commands, *runner.NOT_RUN}) == examples
    for name, args in commands.items():
        source = (runner.EXAMPLES / name).read_text(encoding="utf-8")
        docstring = ast.get_docstring(ast.parse(source)) or ""
        match = re.search(r"^\s*flm serve(.*)$", docstring, re.M)
        assert match, f"{name} does not say how to start the server"
        shown = tuple(match.group(1).split())
        assert args[: len(shown)] == shown, name


def test_every_page_is_in_the_nav_and_llms_txt():
    nav = _nav_pages(SITE["nav"])
    assert len(nav) == len(set(nav))
    site_pages = _site_pages()
    assert sorted(site_pages - set(nav)) == []
    assert sorted(set(nav) - site_pages) == []
    listed = [
        next(iter(entry)) if isinstance(entry, dict) else entry
        for entries in SITE["plugins"]["llmstxt"]["sections"].values()
        for entry in entries
    ]
    assert [
        page
        for page in sorted(site_pages)
        if not any(fnmatch.fnmatchcase(page, pattern) for pattern in listed)
    ] == []


def test_no_relative_link_leaves_the_docs():
    """The site serves docs/ only; link to anything else on GitHub."""
    escapes = []
    for page in DOCS.rglob("*.md"):
        text = _FENCE.sub("", page.read_text(encoding="utf-8"))
        for inline, reference in _LINK.findall(text):
            target = inline or reference
            if re.match(r"[a-z][a-z0-9+.-]*:|#|/", target):
                continue
            path = PurePosixPath(page.relative_to(DOCS).parent.as_posix(), target)
            depth = 0
            for part in path.parts:
                depth += -1 if part == ".." else 0 if part == "." else 1
                if depth < 0:
                    escapes.append(f"{page.relative_to(ROOT).as_posix()}: {target}")
                    break
    assert escapes == []


def test_pages_render_without_external_data():
    rendered = pages.generate(pages.Inputs(root=ROOT))
    assert "no release data" in rendered["releases.md"].text
    assert "no FastFlowLM model list" in rendered["reference/models.md"].text
    for page in rendered.values():
        assert (ROOT / page.source).is_file(), page.source


def _release(tag, body, *, draft=False, published="2026-11-17T00:00:00Z"):
    return {
        "tag_name": tag,
        "draft": draft,
        "published_at": published,
        "html_url": f"https://github.com/ROCm/FastFlow-SDK/releases/tag/{tag}",
        "body": body,
    }


def test_releases_filter_sort_and_extract_sections(tmp_path):
    body = (
        "Intro\r\n\r\n## What's new\r\n\r\n- Added a thing.\r\n\r\n"
        "## Qualification evidence\r\n\r\n| System | Driver |\r\n| --- | --- |\r\n"
        "| Strix Point | 32.0.203.329 |\r\n\r\n```console\r\n## not a heading\r\n```\r\n"
    )
    data = [
        [
            _release("fastflow-sdk-v0.1.0.dev2", body),
            _release("fastflow-sdk-v0.1.0.dev3", body, draft=True),
            _release("rocm-npu-v0.1.0.dev1", body),
        ],
        [
            _release(
                "fastflow-sdk-v0.1.0.dev1",
                "Old notes",
                published="2026-10-01T00:00:00Z",
            )
        ],
    ]
    path = tmp_path / "releases.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    published = releases.load_releases(path)
    assert [releases.version(r) for r in published] == ["0.1.0.dev2", "0.1.0.dev1"]

    evidence = releases.section(body, "Qualification evidence")
    assert evidence is not None and "32.0.203.329" in evidence
    assert "## not a heading" in evidence
    assert releases.section(body, "Missing") is None

    notes = releases.releases_page(published)
    assert "## 0.1.0.dev2" in notes and "- Added a thing." in notes
    assert "Old notes" in notes
    hardware = releases.hardware_page(published)
    assert "Qualified systems for 0.1.0.dev2" in hardware
    assert "| Strix Point | 32.0.203.329 |" in hardware


def test_hardware_page_never_shows_an_older_releases_drivers():
    older = _release("fastflow-sdk-v0.1.0.dev1", "## Qualification evidence\n\n| old |")
    documented = _release("fastflow-sdk-v0.1.0.dev2", "## What's new\n\n- A thing.")
    hardware = releases.hardware_page([documented, older])
    assert "Release 0.1.0.dev2 published no qualification summary" in hardware
    assert "| old |" not in hardware
    assert "no release data" in releases.hardware_page([])


def test_demote_skips_code_fences():
    assert releases.demote("## A\n```\n## B\n```\n", 1) == "### A\n```\n## B\n```\n"


def test_models_page_lists_npu2_models_only():
    model_list = {
        "models": {
            "qwen3": {
                "0.6b": {
                    "supported_platforms": ["aie2p"],
                    "name": "Qwen3-0.6B-NPU2",
                    "url": "https://huggingface.co/FastFlowLM/Qwen3-0.6B-NPU2",
                    "default_context_length": 32768,
                    "flm_min_version": "0.9.22",
                    "footprint": 0.66,
                    "label": ["reasoning"],
                    "details": {"parameter_size": "0.6B", "quantization_level": "Q4_1"},
                }
            },
            "qwen3-core": {"0.6b": {"supported_platforms": ["dev04"]}},
        }
    }
    page = catalogs.models_page(model_list, "07defcb8cb9a6da9")
    assert "| `qwen3:0.6b` | 0.6B | Q4_1 | 32768 | 0.66 | reasoning | 0.9.22 |" in page
    assert "qwen3-core" not in page
    assert "`07defcb8cb9a`" in page


def test_operators_page_reads_params_operands_and_inheritance(tmp_path):
    common = tmp_path / "iron" / "common"
    operators = tmp_path / "iron" / "operators"
    common.mkdir(parents=True)
    operators.mkdir(parents=True)
    (common / "elementwise.py").write_text(
        "class Binary(Operator):\n"
        "    size: int = param()\n"
        "    a = In(size)\n"
        "    b = In(size)\n"
        "    y = Out(like=a)\n",
        encoding="utf-8",
    )
    (operators / "add.py").write_text(
        "class Add(Binary):\n"
        '    """Element-wise addition.\n\n    More detail.\n    """\n'
        "    tile: int = tunable(default=64)\n"
        "    scale: float = param(default=1.0)\n",
        encoding="utf-8",
    )
    (operators / "__init__.py").write_text(
        'from .add import Add\n__all__ = ["Add"]\n', encoding="utf-8"
    )
    (operators / "flm").mkdir()
    (operators / "flm" / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    page = catalogs.operators_page(tmp_path, "b7e8ea91eaad163c")
    row = next(line for line in page.splitlines() if line.startswith("| [`Add`]"))
    assert "#iron.operators.Add" in row
    assert "| Element-wise addition. | `size` | `a`, `b` | `y` |" in row
    assert "scale" not in row and "tile" not in row
    assert "/blob/b7e8ea91eaad163c/iron/operators/add.py#L1" in row
    assert "## Run an operator" not in page
    assert (
        "[IRON programming guide](https://github.com/amd/IRON/tree/b7e8ea91eaad163c/programming_guide)"
        in page
    )


def test_operators_page_embeds_the_guides_first_example(tmp_path):
    (tmp_path / "iron" / "operators" / "flm").mkdir(parents=True)
    for init in ("__init__.py", "flm/__init__.py"):
        (tmp_path / "iron" / "operators" / init).write_text("__all__ = []\n")
    guide = tmp_path / "programming_guide"
    (guide / "section-1").mkdir(parents=True)
    (guide / "section-1" / "axpy_example.py").write_text(
        "y = AXPY(size=2048)(a, b, y)\n", encoding="utf-8"
    )
    (guide / "README.md").write_text(
        "# Guide\n\n## Sections\n\n"
        "* [Section 1 - Your First Operator](./section-1/)\n"
        "* [Section 2 - Graphs](./section-2/)\n\n## Next\n\n"
        "* [Not a section](./elsewhere/)\n",
        encoding="utf-8",
    )
    page = catalogs.operators_page(tmp_path, "b7e8ea91eaad163c")
    base = "https://github.com/amd/IRON/tree/b7e8ea91eaad163c/programming_guide"
    assert "```python\ny = AXPY(size=2048)(a, b, y)\n```" in page
    assert f"[section 1]({base}/section-1)" in page
    assert f"- [Section 2 - Graphs]({base}/section-2)\n" in page
    assert "Not a section" not in page
    assert "IRON/blob/b7e8ea91eaad163c/AGENTS.md" in page
    assert page.index("## Run an operator") < page.index("## Go further")


def test_shipped_operators_page_runs_a_real_example():
    iron_root = pages.inputs_from_environment(ROOT, {}).iron_root
    if iron_root is None:
        pytest.skip("projects/iron-ops is not checked out")
    page = catalogs.operators_page(iron_root, None)
    assert "## Run an operator" in page and "Section 1" in page
    example = iron_root / "programming_guide" / "section-1" / "axpy_example.py"
    for name in ("DEFAULT_TENSOR_CLASS", "set_current_device", "compile()"):
        assert name in page, name
    assert "aie_utils.DEFAULT_TENSOR_CLASS" in example.read_text(encoding="utf-8")


def test_versions_page_lists_every_pinned_component():
    pins, release = versions.load(ROOT)
    page = versions.versions_page(pins, release)
    for component, entry in pins.items():
        if component != "meta" and entry.get("commit"):
            assert entry["commit"] in page, component
    tagged = versions.versions_page(pins, release, "fastflow-sdk-v0.1.0.dev2")
    assert "/blob/fastflow-sdk-v0.1.0.dev2/manifests/versions.toml" in tagged


def test_releases_start_at_the_documented_release(tmp_path):
    data = [
        [
            _release(
                "fastflow-sdk-v0.1.0.dev3", "Newer", published="2026-11-18T00:00:00Z"
            ),
            _release("fastflow-sdk-v0.1.0.dev2", "Documented"),
            _release(
                "fastflow-sdk-v0.1.0.dev1", "Older", published="2026-10-01T00:00:00Z"
            ),
        ]
    ]
    path = tmp_path / "releases.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert releases.latest_tag(path) == "fastflow-sdk-v0.1.0.dev3"
    shown = releases.load_releases(path, "fastflow-sdk-v0.1.0.dev2")
    assert [releases.version(r) for r in shown] == ["0.1.0.dev2", "0.1.0.dev1"]
    with pytest.raises(ValueError, match="fastflow-sdk-v9"):
        releases.load_releases(path, "fastflow-sdk-v9")
    path.write_text("[[]]", encoding="utf-8")
    with pytest.raises(ValueError):
        releases.latest_tag(path)


def test_write_puts_pages_and_examples_where_the_site_reads_them(tmp_path):
    for name in (
        cli.COMMAND_SOURCE,
        cli.CHECKS,
        versions.VERSIONS,
        Path("release.toml"),
        Path("examples/hello_npu.py"),
        Path("examples/serve/chat.py"),
    ):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    written = pages.write(pages.Inputs(root=tmp_path))
    docs = tmp_path / "docs"
    assert {p.relative_to(docs).as_posix() for p in written} == set(
        pages.generate(pages.Inputs(root=tmp_path))
    ) | {"examples/hello_npu.py", "examples/serve/chat.py"}
    cli_page = (docs / "reference" / "cli.md").read_bytes()
    assert cli_page.startswith(b"<!-- Generated by build_tools/docs from tools/")
    assert b"\r\n" not in cli_page
    assert (docs / "examples" / "hello_npu.py").read_bytes() == (
        ROOT / "examples" / "hello_npu.py"
    ).read_bytes()


def test_generated_paths_are_gitignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for path in (
        "/site/",
        "/docs/reference/",
        "/docs/releases.md",
        "/docs/examples/*.py",
    ):
        assert path in ignored, path
