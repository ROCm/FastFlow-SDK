# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Portable qualification evidence bundle writer."""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .candidate import canonical_candidate_bytes
from .contracts import CANDIDATE_SCHEMA_VERSION
from .hashing import write_sha256sums


def junit_bytes(tests: list[dict[str, Any]]) -> bytes:
    """Render the canonical JUnit projection of qualification test records."""
    failures = sum(test.get("status") == "fail" for test in tests)
    skipped = sum(test.get("status") == "unsupported" for test in tests)
    suite = ET.Element(
        "testsuite",
        {
            "name": "rocm-npu-qualification",
            "tests": str(len(tests)),
            "failures": str(failures),
            "skipped": str(skipped),
        },
    )
    for test in tests:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": ".".join(
                    (
                        str(test.get("model", "qualification")),
                        str(test.get("kind", "qualification")),
                    )
                ),
                "name": str(test.get("name", "unnamed")),
                "time": f"{float(test.get('duration_seconds', 0.0)):.6f}",
            },
        )
        properties = ET.SubElement(case, "properties")
        for key in (
            "kind",
            "turn_count",
            "process_iteration",
            "soak_iterations",
            "rounds",
        ):
            if key in test:
                ET.SubElement(
                    properties,
                    "property",
                    {"name": key, "value": str(test[key])},
                )
        if "metrics" in test:
            ET.SubElement(
                properties,
                "property",
                {
                    "name": "metrics",
                    "value": json.dumps(test["metrics"], sort_keys=True),
                },
            )
        if test.get("status") == "fail":
            failure = ET.SubElement(case, "failure")
            failure.text = str(test.get("error", "qualification failed"))
        elif test.get("status") == "unsupported":
            skipped_element = ET.SubElement(case, "skipped")
            skipped_element.text = str(
                test.get("error", "qualification case is unsupported")
            )
        output = ET.SubElement(case, "system-out")
        output.text = str(test.get("log", ""))
    tree = ET.ElementTree(suite)
    ET.indent(tree, space="  ")
    stream = io.BytesIO()
    tree.write(stream, encoding="utf-8", xml_declaration=True)
    return stream.getvalue()


class EvidenceWriter:
    def __init__(self, root: Path, *, allow_existing: bool = False):
        self.root = root
        if self.root.exists() and any(self.root.iterdir()) and not allow_existing:
            raise FileExistsError(
                f"qualification evidence output is not empty: {self.root}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "logs" / "pull").mkdir(parents=True, exist_ok=True)
        (self.root / "logs" / "tests").mkdir(parents=True, exist_ok=True)

    def path(self, relative: str) -> Path:
        return self.root / relative

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def write_json(self, relative: str, value: Any) -> None:
        destination = self.path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def write_candidate_manifest(self, candidate: dict[str, Any] | None) -> None:
        if candidate is None:
            self.write_json(
                "candidate-manifest.json",
                {
                    "schema_version": CANDIDATE_SCHEMA_VERSION,
                    "kind": "ad-hoc",
                    "candidate_bound": False,
                },
            )
        else:
            destination = self.path("candidate-manifest.json")
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_bytes(canonical_candidate_bytes(candidate))
            temporary.replace(destination)

    def write_junit(self, tests: list[dict[str, Any]]) -> None:
        self.path("junit.xml").write_bytes(junit_bytes(tests))

    def finalize(self) -> None:
        write_sha256sums(self.root)
