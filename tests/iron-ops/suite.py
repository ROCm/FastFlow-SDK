# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""What the iron-ops gates cover: the operator table, and nothing else.

ONE definition, read by both gates -- `dev.py test-iron-ops` (build it, CPU runners) and
`dev.py test-npu-iron-ops` (run it, NPU runner). Adding a row extends both. It lives here
rather than in manifests/versions.toml because it is a test matrix, not a pin: versions.toml
records WHICH revision of iron-ops we build, this records WHAT we build out of it.

Growing this: add a row. Shapes are deliberately small -- these are correctness gates on
shared CI runners, not a benchmark. The build kwargs should match one case in the
operator's own test.py, and `hrx_pytest_filter` selects the matching on-device HRX smoke.
"""

from typing import NamedTuple


class Operator(NamedTuple):
    """One operator under test.

    Attributes:
        name: directory under iron-ops' iron/operators/
        build_kwargs: constructor kwargs for the CPU-side compile gate
        device: NPU generation to build for (dev.py's NPU_TARGETS)
        hrx_pytest_filter: pytest -k expression for the on-device HRX smoke
    """

    name: str
    build_kwargs: dict[str, object]
    device: str = "npu2"
    hrx_pytest_filter: str | None = None


# Keep the HRX on-device gate to out-of-box single-stream cases for now. Multi-stream
# direct-FIFO designs currently show zeroed output chunks through HRX, so they need a
# runtime/topology fix rather than a CI expectation that every upstream parameter passes.
OPERATORS = (
    Operator(
        name="elementwise_add",
        build_kwargs={"size": 2048, "num_aie_columns": 1, "tile_size": 2048},
        hrx_pytest_filter=(
            "test_elementwise_add and input_length_2048 and "
            "num_aie_columns_1 and tile_size_2048"
        ),
    ),
    Operator(
        name="relu",
        build_kwargs={
            "size": 2048,
            "num_aie_columns": 1,
            "num_channels": 1,
            "tile_size": 2048,
        },
        hrx_pytest_filter=(
            "test_relu and input_length_2048 and num_aie_columns_1 and "
            "num_channels_1 and tile_size_2048"
        ),
    ),
    Operator(
        name="gemv",
        build_kwargs={
            "M": 128,
            "K": 128,
            "num_aie_columns": 1,
            "tile_size_input": 32,
            "tile_size_output": 128,
        },
        hrx_pytest_filter="test_gemv and M_128 and K_128 and num_aie_columns_1",
    ),
)

DEFAULT = OPERATORS[0].name


def select(name: str | None) -> tuple[Operator, ...]:
    """The operators to gate on: all of them, or the one named."""
    if name is None:
        return OPERATORS
    chosen = [op for op in OPERATORS if op.name == name]
    if not chosen:
        known = ", ".join(op.name for op in OPERATORS)
        raise KeyError(f"{name}: not in the iron-ops suite ({known}) -- add a row to OPERATORS")
    return tuple(chosen)
