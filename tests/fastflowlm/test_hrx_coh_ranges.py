# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Disjoint HRX sub-buffer slices keep independent dirty ranges."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = (
    ROOT
    / "projects"
    / "FastFlowLM"
    / "src"
    / "include"
    / "hrx_cpp"
    / "hrx_coh_ranges.hpp"
)

DRIVER = r"""
#include "hrx_coh_ranges.hpp"
#include <cstdio>
#include <vector>

static void print_ranges(const char* label, const std::vector<hrx::coh_range>& ranges) {
    std::printf("%s", label);
    for (const hrx::coh_range& range : ranges) {
        std::printf(" %zu+%zu", range.offset, range.length);
    }
    std::printf("\n");
}

int main() {
    std::vector<hrx::coh_range> host;
    hrx::coh_range_add(host, 0, 4096);
    auto first = hrx::coh_range_take(host, 0, 512);
    auto second = hrx::coh_range_take(host, 2048, 512);
    print_ranges("flush-a", first);
    print_ranges("flush-b", second);
    print_ranges("host-left", host);

    std::vector<hrx::coh_range> device;
    hrx::coh_range_add(device, 0, 512);
    hrx::coh_range_add(device, 2048, 512);
    auto read_a = hrx::coh_range_take(device, 0, 512);
    print_ranges("invalidate-a", read_a);
    print_ranges("device-left", device);

    std::vector<hrx::coh_range> pending;
    hrx::coh_range_add(pending, 0, 512);
    hrx::coh_range_add(pending, 2048, 512);
    hrx::coh_range_add(pending, 4096, 512);
    std::vector<hrx::coh_range> restored;
    int applied = 0;
    bool failed = hrx::coh_range_apply(
        restored, pending, [&](const hrx::coh_range& range) {
            ++applied;
            return range.offset != 2048;
        });
    std::printf("apply-failed %d\n", failed ? 1 : 0);
    std::printf("apply-count %d\n", applied);
    print_ranges("apply-restored", restored);

    std::printf(
        "slice-fit %d %d %d\n",
        hrx::coh_slice_fits(4096, 0, 512, 1024) ? 1 : 0,
        hrx::coh_slice_fits(4096, 0, 512, 4000) ? 1 : 0,
        hrx::coh_slice_fits(100, static_cast<size_t>(-1) - 10, 1, 20) ? 1 : 0);
    return 0;
}
"""

pytestmark = [
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


def _compile(
    source: Path, binary: Path, include: Path
) -> subprocess.CompletedProcess[str]:
    for name in ("c++", "g++", "clang++"):
        compiler = shutil.which(name)
        if compiler:
            return subprocess.run(
                [
                    compiler,
                    "-std=c++20",
                    f"-I{include}",
                    str(source),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
    clang_cl = shutil.which("clang-cl")
    if clang_cl:
        return subprocess.run(
            [
                clang_cl,
                "/std:c++20",
                "/EHsc",
                f"/I{include}",
                str(source),
                f"/Fe{binary}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    return subprocess.CompletedProcess(
        args=[], returncode=127, stdout="", stderr="no compiler"
    )


def test_disjoint_slices_flush_and_invalidate_independently(tmp_path):
    if not HEADER.is_file():
        pytest.skip("HRX dirty-range header is not in this checkout")
    source = tmp_path / "driver.cpp"
    source.write_text(DRIVER, encoding="utf-8")
    binary = tmp_path / "driver.exe"
    build = _compile(source, binary, HEADER.parent)
    if build.returncode == 127:
        pytest.skip("no C++ compiler for the HRX dirty-range header")
    assert build.returncode == 0, build.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "flush-a 0+512",
        "flush-b 2048+512",
        "host-left 512+1536 2560+1536",
        "invalidate-a 0+512",
        "device-left 2048+512",
        "apply-failed 1",
        "apply-count 3",
        "apply-restored 2048+512",
        "slice-fit 1 0 0",
    ]
