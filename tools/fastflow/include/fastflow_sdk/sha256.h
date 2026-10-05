// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fastflow_sdk {

std::string Sha256Hex(const std::vector<std::uint8_t>& bytes);

}  // namespace fastflow_sdk
