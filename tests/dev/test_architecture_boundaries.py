# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.presubmit
ROOT = Path(__file__).resolve().parents[2]


def _mentions_tests(node, test_path_names):
    if node is None:
        return False
    if isinstance(node, ast.Name) and node.id in test_path_names:
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value == "tests" or node.value.startswith(("tests/", "tests\\"))
    return any(
        _mentions_tests(child, test_path_names) for child in ast.iter_child_nodes(node)
    )


def _imported_modules(node):
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        imported = {module}
        if module:
            imported.update(f"{module}.{alias.name}" for alias in node.names)
        return imported
    return set()


def test_production_modules_do_not_load_test_implementations():
    violations = []
    for path in sorted((ROOT / "build_tools").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        test_path_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name) and _mentions_tests(node.value, set())
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                imported = []
            if any(name == "tests" or name.startswith("tests.") for name in imported):
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: imports tests"
                )

            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            owner = node.func.value
            mutates_sys_path = (
                isinstance(owner, ast.Attribute)
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "sys"
                and owner.attr == "path"
                and node.func.attr in {"append", "extend", "insert"}
            )
            if mutates_sys_path and any(
                _mentions_tests(argument, test_path_names)
                for argument in (*node.args, *[kw.value for kw in node.keywords])
            ):
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: bootstraps tests on sys.path"
                )

    assert violations == []


def test_non_dev_build_tools_modules_do_not_import_dev():
    violations = []
    build_tools = ROOT / "build_tools"
    for path in sorted(build_tools.rglob("*.py")):
        if (build_tools / "dev") in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports_dev = any(
                    alias.name == "build_tools.dev"
                    or alias.name.startswith("build_tools.dev.")
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports_dev = (
                    module == "build_tools.dev"
                    or module.startswith("build_tools.dev.")
                    or (
                        module == "build_tools"
                        and any(alias.name == "dev" for alias in node.names)
                    )
                    or (
                        node.level == 1
                        and not module
                        and any(alias.name == "dev" for alias in node.names)
                    )
                )
            else:
                imports_dev = False
            if imports_dev:
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: imports build_tools.dev"
                )

    assert violations == []


def test_product_domains_do_not_import_sibling_orchestration_layers():
    forbidden = {
        "fastflowlm_artifact.py": {"build_tools.fastflow_sdk_wheel_common"},
        "compiler_product.py": {"build_tools.fastflow_sdk_compiler_wheels"},
        "compiler_wheels.py": {"build_tools.fastflow_sdk_linux_wheels"},
        "manylinux_audit.py": {"build_tools.fastflow_sdk_wheel_common"},
    }
    violations = []
    build_tools = ROOT / "build_tools"
    for filename, forbidden_modules in forbidden.items():
        path = build_tools / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported = _imported_modules(node)
            violations_for_node = sorted(imported & forbidden_modules)
            for imported_module in violations_for_node:
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: imports {imported_module}"
                )

    package = build_tools / "fastflowlm_package.py"
    tree = ast.parse(package.read_text(encoding="utf-8"), filename=str(package))
    platform_modules = {
        "build_tools.fastflow_sdk_linux_wheels",
        "build_tools.fastflow_sdk_windows_wheels",
    }
    for node in tree.body:
        for imported_module in sorted(_imported_modules(node) & platform_modules):
            violations.append(
                f"{package.relative_to(ROOT)}:{node.lineno}: "
                f"eagerly imports {imported_module}"
            )

    staging = build_tools / "runtime_wheel_staging.py"
    tree = ast.parse(staging.read_text(encoding="utf-8"), filename=str(staging))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if alias.name.startswith("_"):
                violations.append(
                    f"{staging.relative_to(ROOT)}:{node.lineno}: "
                    f"imports private symbol {alias.name}"
                )

    assert violations == []


@pytest.mark.parametrize(
    ("directory", "module"),
    [
        ("tests/npu", "hrx_api"),
        ("tests/npu", "npu_cases"),
        ("tests/iron-ops", "suite"),
    ],
)
def test_thin_test_contract_modules_load_from_their_legacy_paths(directory, module):
    completed = subprocess.run(
        [sys.executable, "-S", "-c", f"import {module}"],
        cwd=ROOT / directory,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
