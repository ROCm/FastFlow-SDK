// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "fastflow_sdk/platform.h"
#include "fastflow_sdk/product.h"

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "fastflow-sdk-product-test";
  std::filesystem::remove_all(fastflow_sdk::LongPath(root));
  std::filesystem::create_directories(root / "bin");
  std::filesystem::create_directories(root / "pkg");
  std::ofstream(root / "bin" / "payload", std::ios::binary) << "abc";
  std::ofstream(root / "pkg" / "module.py");
  const std::filesystem::path manifest_path = root / "manifest.json";
  nlohmann::json manifest = {
      {"files",
       {{{"path", "bin/payload"},
         {"size", std::uint64_t{3}},
         {"sha256",
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}},
        {{"path", "pkg/module.py"},
         {"size", std::uint64_t{0}},
         {"sha256",
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85"
          "5"}}}},
  };
  nlohmann::json details;
  std::string error;
  assert(fastflow_sdk::VerifyProductManifestFiles(manifest_path, manifest,
                                                  details, error));
  assert(details.at("verified_file_count") == 2);

  std::ofstream(root / "bin" / "payload", std::ios::binary | std::ios::trunc)
      << "abd";
  assert(!fastflow_sdk::VerifyProductManifestFiles(manifest_path, manifest,
                                                   details, error));
  assert(error.find("SHA256") != std::string::npos);

  std::ofstream(root / "bin" / "payload", std::ios::binary | std::ios::trunc)
      << "abc";
  std::filesystem::remove(root / "pkg" / "module.py");
  assert(!fastflow_sdk::VerifyProductManifestFiles(manifest_path, manifest,
                                                   details, error));
  assert(error.find("missing") != std::string::npos);

  manifest["files"][0]["path"] = "../outside";
  assert(!fastflow_sdk::VerifyProductManifestFiles(manifest_path, manifest,
                                                   details, error));
  assert(error.find("unsafe") != std::string::npos);

  // Windows file APIs reject paths of MAX_PATH (260) or more unless they are
  // \\?\-prefixed, so the deep file can only be created through LongPath.
  const std::string deep_payload = "deep/" + std::string(120, 'a') + "/" +
                                   std::string(120, 'b') + "/payload";
  const std::filesystem::path long_payload =
      fastflow_sdk::LongPath(root / deep_payload);
  std::filesystem::create_directories(long_payload.parent_path());
  std::ofstream(long_payload, std::ios::binary) << "abc";
  assert(std::filesystem::absolute(root / deep_payload).native().size() >= 260);
  manifest["files"] = {
      {{"path", deep_payload},
       {"size", std::uint64_t{3}},
       {"sha256",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}}};
  assert(fastflow_sdk::VerifyProductManifestFiles(manifest_path, manifest,
                                                  details, error));
  assert(details.at("verified_file_count") == 1);

  std::filesystem::remove_all(fastflow_sdk::LongPath(root));
  return 0;
}
