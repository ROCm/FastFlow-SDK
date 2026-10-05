// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <stdexcept>
#include <string>

#include "modules/dequant.hpp"
#include "modules/gemm.hpp"
#include "npu_utils/npu_utils.hpp"

int main(int argc, char** argv) {
  if (argc < 4) return 2;
  const std::string kernel = argv[1];
  const uint32_t m = static_cast<uint32_t>(std::stoul(argv[2]));
  const uint32_t k = static_cast<uint32_t>(std::stoul(argv[3]));
  LM_Config config = {};
  npu_sequence sequence(device_npu2);
  if (kernel == "dequant") {
    Dequant dequant(config);
    dequant.generate_dequant_q4_1_seq(&sequence, k, m, 0, 0);
  } else if (kernel == "mm" && argc == 5) {
    const uint32_t n = static_cast<uint32_t>(std::stoul(argv[4]));
    Gemm gemm(config);
    gemm.generate_seq(&sequence, m, k, n, 0, false, Gemm::NO_Activation, 0);
  } else {
    throw std::runtime_error("invalid shape-probe arguments");
  }
  const auto [words, word_count] = sequence.dump();
  return words && word_count ? 0 : 1;
}
