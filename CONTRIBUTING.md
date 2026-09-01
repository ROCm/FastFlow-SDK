# Contributing to ROCm-NPU

Contributions to code and documentation are welcome. This is an early snapshot.

Open [issues](https://github.com/ROCm/rocm-npu/issues) for bugs and feature
requests. Include OS, NPU device, driver version, and the release tag or
commit. For security reports, see [SECURITY.md](SECURITY.md).

Organization-wide ROCm guidelines are in
[ROCm/CONTRIBUTING.md](https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md).

## Pull requests

Target `main`. Keep the change focused, add tests for new behavior, and do not
break existing tests. Presubmit runs `python -m pytest -m presubmit`.

```bash
python3 dev.py --help
python3 -m pytest -m presubmit
```

By opening a pull request, you agree that the contribution is licensed under
the [LICENSE](LICENSE).

## Formatting and lint

Presubmit runs `python3 dev.py format --check` on first-party Python and C/C++.
Pins live in `requirements/test.txt`:

```bash
python3 -m pip install --require-hashes -r requirements/test.lock
python3 dev.py format --check
python3 -m pytest -m presubmit
```

`python3 dev.py format` rewrites. Use `python3 dev.py ruff` or
`python3 dev.py clang-format` to run one tool. `python3 dev.py setup`
installs the wheels into `.venv`; `python3 dev.py doctor` reports them.

## Submodules

Runtime, compiler, and model changes belong upstream. Bump the pin here after
they land:

- HRX: [ROCm/hrx-system](https://github.com/ROCm/hrx-system)
- FastFlowLM: [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM)
- iron-tools: [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie)
- iron-ops: [amd/IRON](https://github.com/amd/IRON)

Temporary submodule fixes go in `patches/` and must stay upstreamable. See
[patches/README.md](patches/README.md).
