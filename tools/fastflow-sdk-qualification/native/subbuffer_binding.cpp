// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Dispatch one prebuilt FastFlowLM xclbin through hrx::bo slices. Each payload
// sits in the middle of a larger allocation. The kernel is bound at that byte
// offset via run::add_binding. Bytes outside the slice must stay unchanged.

#include "hrx_cpp/hrx_cpp.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kGuard = 4096;
constexpr uint8_t kFill = 0xA5;

std::vector<uint8_t> read_file(const char* path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(std::string("cannot read ") + path);
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
}

struct Slot {
  std::unique_ptr<hrx::ext::bo> parent;
  std::unique_ptr<hrx::bo> slice;
  size_t payload = 0;
  std::string path;
};

Slot make_slot(hrx::device& device, const char* path) {
  const std::vector<uint8_t> payload = read_file(path);
  Slot slot;
  slot.payload = payload.size();
  slot.path = path;
  slot.parent =
      std::make_unique<hrx::ext::bo>(device, payload.size() + 2 * kGuard);
  auto* mapped = slot.parent->map<uint8_t*>();
  std::memset(mapped, kFill, slot.parent->size());
  std::memcpy(mapped + kGuard, payload.data(), payload.size());
  if (hrx::hrx_report(
          hrx_buffer_flush_range(slot.parent->handle(), 0, slot.parent->size()),
          "subbuffer flush")) {
    throw std::runtime_error("failed to publish the parent allocation");
  }
  slot.slice = std::make_unique<hrx::bo>(*slot.parent, payload.size(), kGuard);
  return slot;
}

bool guards_intact(const Slot& slot) {
  if (hrx::hrx_report(hrx_buffer_invalidate_range(slot.parent->handle(), 0,
                                                  slot.parent->size()),
                      "subbuffer invalidate")) {
    return false;
  }
  const auto* mapped = slot.parent->map<const uint8_t*>();
  const size_t tail = kGuard + slot.payload;
  for (size_t index = 0; index < kGuard; ++index) {
    if (mapped[index] != kFill || mapped[tail + index] != kFill) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s xclbin transaction payload...\n", argv[0]);
    return 2;
  }
  try {
    const std::vector<uint8_t> xclbin = read_file(argv[1]);
    const std::vector<uint8_t> transaction = read_file(argv[2]);
    if (transaction.size() % sizeof(uint32_t) != 0) {
      throw std::runtime_error("transaction is not a whole number of words");
    }
    std::vector<uint32_t> words(transaction.size() / sizeof(uint32_t));
    std::memcpy(words.data(), transaction.data(), transaction.size());
    uint32_t ordinal = 0;
    hrx_executable_t executable = hrx::lookup_or_build_executable(
        xclbin, words.data(), words.size(), &ordinal, nullptr);
    if (!executable) {
      throw std::runtime_error("executable create failed");
    }
    hrx::device device(0);
    std::vector<Slot> slots;
    slots.reserve(static_cast<size_t>(argc - 3));
    for (int index = 3; index < argc; ++index) {
      slots.push_back(make_slot(device, argv[index]));
    }
    hrx::run run(executable, ordinal);
    for (const Slot& slot : slots) {
      run.add_binding(*slot.slice);
    }
    run.start();
    if (run.wait() != ERT_CMD_STATE_COMPLETED) {
      throw std::runtime_error("dispatch failed");
    }
    for (const Slot& slot : slots) {
      if (!guards_intact(slot)) {
        throw std::runtime_error(
            "a slice binding wrote outside its byte range");
      }
      const auto* mapped = slot.parent->map<const uint8_t*>();
      std::ofstream readback(slot.path + ".readback", std::ios::binary);
      readback.write(reinterpret_cast<const char*>(mapped + kGuard),
                     static_cast<std::streamsize>(slot.payload));
      if (!readback) {
        throw std::runtime_error("cannot write the slice readback");
      }
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
