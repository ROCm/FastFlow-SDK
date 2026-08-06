// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "rocm_npu/product.h"

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "rocm-npu-product-test";
  std::filesystem::remove_all(root);
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
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}}}},
  };
  nlohmann::json details;
  std::string error;
  assert(rocm_npu::VerifyProductManifestFiles(manifest_path, manifest, details,
                                              error));
  assert(details.at("verified_file_count") == 2);

  std::ofstream(root / "bin" / "payload", std::ios::binary | std::ios::trunc)
      << "abd";
  assert(!rocm_npu::VerifyProductManifestFiles(manifest_path, manifest, details,
                                               error));
  assert(error.find("SHA256") != std::string::npos);

  std::ofstream(root / "bin" / "payload", std::ios::binary | std::ios::trunc)
      << "abc";
  std::filesystem::remove(root / "pkg" / "module.py");
  assert(!rocm_npu::VerifyProductManifestFiles(manifest_path, manifest, details,
                                               error));
  assert(error.find("missing") != std::string::npos);

  manifest["files"][0]["path"] = "../outside";
  assert(!rocm_npu::VerifyProductManifestFiles(manifest_path, manifest, details,
                                               error));
  assert(error.find("unsafe") != std::string::npos);

  std::filesystem::remove_all(root);
  return 0;
}
