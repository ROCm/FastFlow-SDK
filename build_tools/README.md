# Build tools

`build_tools` contains repository-local automation shared by developers, CI
workflows, tests, packaging jobs, and release jobs. It is not limited to CI,
and it is not part of a shipped FastFlow SDK product.

The root `dev.py` command is the primary developer entry point.
`build_tools/dev` owns its parser, dispatch, setup, source preparation, build,
diagnostic, and test orchestration. Product-specific artifact and release logic
belongs to the corresponding domain under `build_tools`. Repository-wide
paths, process environments, and toolchain contracts are shared foundations,
not developer-command internals.

## Non-responsibilities

Do not put the following in `build_tools`:

- Product runtime or compiler implementation that ships to users.
- Tests used as implementation libraries. Reusable support code belongs in an
  importable tooling module; tests may consume it.
- GitHub Actions YAML plumbing that only reads workflow inputs or writes
  workflow outputs and has no reusable behavior.
- Source-of-truth product data that is naturally owned by the product,
  manifest, schema, or release policy it describes.
- A cross-product helper hidden in one product-specific module.
- Convenience wrappers that only add another name or call layer without
  creating a stable boundary.

Avoid generic dumping grounds such as `utils.py` or `common.py`. Shared code
should be named for the capability it owns, such as archive handling, source
provenance, native-binary inspection, or repository paths.

## Contracts

Code under `build_tools` should follow these contracts:

- **Importable logic, thin entry points.** Command-line modules parse inputs
  and report results; importable functions own behavior.
- **CI is a caller, not the architecture.** Logic needed by both workflows and
  local development must be executable and testable outside GitHub Actions.
- **Dependencies point toward foundations.** Entry points depend on domain
  orchestration, which depends on product mechanics and narrowly named shared
  capabilities. Shared capabilities must not depend on product policy.
- **Developer internals stay scoped.** Product packaging must not depend on a
  developer command handler merely to obtain repository paths, subprocess
  helpers, or artifact-layout constants.
- **Product policy stays with its owner.** Platform tags, payload composition,
  qualification rules, and release constraints remain in their product
  domains even when they use shared mechanics.
- **Platform mechanics remain explicit.** Linux and Windows implementations
  may share platform-neutral staging and validation, but ELF, PE, loader,
  driver, and system-runtime policy should remain visibly platform-specific.
- **Verification is first-class.** Artifact builders must expose validation
  that can run independently of artifact production. Destructive operations
  and archive extraction must validate their target boundaries.
- **Repository layout is not implicit.** Centralize stable repository and
  artifact paths instead of rediscovering them or copying constants across
  entry points.
- **No runtime imports from `tests`.** If tooling and tests need the same API,
  move that API to an importable owner and keep the test entry point thin.

Important contracts should be enforced by focused tests where practical;
documentation alone is not a substitute for validation.

## Placement guide

When adding code, use the first matching case:

1. **Does it ship as product implementation?**
   Place it in the owning product source tree, not `build_tools`.
2. **Is it source-of-truth policy or data?**
   Place it with the owning manifest, schema, product, or release policy.
3. **Is it reusable repository automation?**
   Place the importable behavior in `build_tools`.
4. **Does it implement a `dev.py` setup, build, doctor, source, or test
   workflow?**
   Place command orchestration in `build_tools/dev`.
5. **Does it build or verify one product or artifact family?**
   Place it in that product's tooling domain.
6. **Is it used across product domains?**
   Place it in a narrowly named shared capability; do not choose an arbitrary
   existing product module.
7. **Is it operating-system-specific?**
   Keep the platform-specific implementation explicit and share only the
   platform-neutral contract.
8. **Is it only GitHub Actions input/output adaptation?**
   Place a thin adapter under `.github/scripts` and call importable
   `build_tools` logic.
9. **Is it only a test or fixture?**
   Place it under `tests`; do not make production tooling import it.

Before creating a new module, check whether it establishes a durable ownership
boundary or merely fragments an existing one. Before extending an existing
module, check whether the new behavior has the same policy owner and dependency
direction.
