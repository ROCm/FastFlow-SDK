# JSON for Modern C++

- Project: JSON for Modern C++
- Upstream: https://github.com/nlohmann/json
- Version: 3.12.0
- Tag: v3.12.0
- License: MIT
- Local modifications: none

## Downloaded files

| File | Upstream source | SHA-256 |
| --- | --- | --- |
| `include/nlohmann/json.hpp` | `https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp` | `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63` |
| `LICENSE.MIT` | `https://raw.githubusercontent.com/nlohmann/json/v3.12.0/LICENSE.MIT` | `46a65cffd1ea955132d95a8dd921640714a8d6b537d2e4e482d31145ae95b603` |

## Update procedure

1. Select an upstream release tag and update the version and tag above.
2. Download `json.hpp` from that release's assets and `LICENSE.MIT` from the
   corresponding tag. Do not copy either file from another checkout.
3. Replace the vendored files without local modifications.
4. Compute each file's SHA-256 digest and update the table.
5. Update the compile-time version assertion, product manifests, license
   inventories, notices, SBOM component, and packaging tests.
6. Build and run the native CLI CTest suite on Linux and Windows.
