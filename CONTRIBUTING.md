# Contributing to FastFlow SDK

Contributions to code and documentation are welcome. This is an early snapshot.

Open [issues](https://github.com/ROCm/FastFlow-SDK/issues) for bugs and feature
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

## Documentation

The site at [rocm.github.io/FastFlow-SDK](https://rocm.github.io/FastFlow-SDK/)
is built from `docs/` by [Zensical](https://zensical.org/) (`zensical.toml`).
It documents a release: after publishing one, run the
`Publish FastFlow SDK docs and simple index` workflow, which builds the site
from the release tag (see
[Package index](docs/wheel-releases.md#package-index)). Presubmit builds the
pull request with `--strict`, so a broken link, anchor, or include fails. To
preview it:

```bash
python3 -m pip install --require-hashes -r requirements/docs.lock
python3 -m build_tools.docs serve
```

`python3 -m build_tools.docs build` is the strict build into `site/`. Both
first write the generated pages, `docs/reference/` and `docs/releases.md`, from
the files that own the facts, such as the CLI sources,
`manifests/versions.toml`, and the IRON and FastFlowLM submodules. Those
paths are gitignored: change the source, not the page. A new `fastflow` check
ID needs an entry in `tools/fastflow/checks.toml`, which becomes its
Troubleshooting section; `tests/dev/test_docs.py` fails until it has one.
Guides include runnable files from the repository root
(`--8<-- "examples/serve/chat.py"`) instead of copying code, and the site
publishes those files under `/examples/`. CI runs them on an NPU;
`tests/fastflowlm/run_serve_examples.py` lists any it skips, and why. The site
serves `docs/` only, so link to other repository files with a GitHub URL.

## Submodules

Runtime, compiler, and model changes belong upstream. Bump the pin here after
they land:

- HRX: [ROCm/hrx-system](https://github.com/ROCm/hrx-system)
- FastFlowLM: [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM)
- iron-tools: [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie)
- iron-ops: [amd/IRON](https://github.com/amd/IRON)

Temporary submodule fixes go in `patches/` and must stay upstreamable. See
[patches/README.md](patches/README.md).
