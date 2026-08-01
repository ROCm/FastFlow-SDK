// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kernel_case.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "kernel_validation.hpp"

namespace {

constexpr size_t kGuardBytes = 4096;

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

hrx::device& device() {
  static hrx::device value(0);
  return value;
}

std::string content_id(const std::vector<uint8_t>& data) {
  uint64_t hash = 14695981039346656037ull;
  for (uint8_t byte : data) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  char text[17];
  std::snprintf(
      text, sizeof(text), "%016llx",
      static_cast<unsigned long long>(hash));
  return text;
}

}  // namespace

KernelExecutable::KernelExecutable(KernelExecutable&& other) noexcept
    : handle(std::exchange(other.handle, nullptr)), ordinal(other.ordinal) {}

KernelExecutable::~KernelExecutable() {
  if (handle) hrx_executable_release(handle);
}

struct KernelCase::Impl {
  Impl(
      std::string family, std::string kernel, std::string vector_id,
      std::vector<size_t> dimensions,
      const std::filesystem::path& xclbin_path,
      std::vector<uint8_t> transaction,
      std::vector<uint8_t> argument1_values,
      std::vector<uint8_t> argument2_values,
      std::vector<bf16> expected,
      std::string weight_provenance,
      std::string reference_kind,
      KernelComparison comparison,
      float absolute_tolerance,
      float relative_tolerance)
      : family(std::move(family)),
        kernel(std::move(kernel)),
        vector_id(std::move(vector_id)),
        dimensions(std::move(dimensions)),
        xclbin(read_file(xclbin_path)),
        implementation_id(content_id(xclbin)),
        transaction(std::move(transaction)),
        argument1_data(std::move(argument1_values)),
        argument2_data(std::move(argument2_values)),
        expected(std::move(expected)),
        weight_provenance(std::move(weight_provenance)),
        reference_kind(std::move(reference_kind)),
        comparison(comparison),
        absolute_tolerance(absolute_tolerance),
        relative_tolerance(relative_tolerance),
        output(device(), this->expected.size() * sizeof(bf16) +
                             2 * kGuardBytes),
        argument1(device(), argument1_data.size() + 2 * kGuardBytes),
        argument2(device(), argument2_data.size() + 2 * kGuardBytes) {
    if (xclbin.empty() || this->transaction.empty() || this->expected.empty() ||
        argument1_data.empty() || argument2_data.empty()) {
      throw std::runtime_error(
          this->family + "/" + this->kernel + "/" + this->vector_id +
          " has empty kernel case data: xclbin=" +
          std::to_string(xclbin.size()) +
          " transaction=" + std::to_string(this->transaction.size()) +
          " argument1=" + std::to_string(argument1_data.size()) +
          " argument2=" + std::to_string(argument2_data.size()) +
          " expected=" + std::to_string(this->expected.size()));
    }
    if (this->weight_provenance.empty() || this->reference_kind.empty()) {
      throw std::runtime_error(
          this->family + "/" + this->kernel + "/" + this->vector_id +
          " has incomplete validation provenance");
    }
    if (absolute_tolerance < 0.0f || relative_tolerance < 0.0f ||
        (comparison != KernelComparison::kTolerance &&
         (absolute_tolerance != 0.0f || relative_tolerance != 0.0f))) {
      throw std::runtime_error(
          this->family + "/" + this->kernel + "/" + this->vector_id +
          " has an invalid comparison tolerance");
    }
    initialize_input(argument1, argument1_data);
    initialize_input(argument2, argument2_data);
    argument1.sync_to_device();
    argument2.sync_to_device();
  }

  static void initialize_input(
      buffer<uint8_t>& value, const std::vector<uint8_t>& data) {
    value.memset(0x5A);
    std::memcpy(value.data() + kGuardBytes, data.data(), data.size());
  }

  std::string family;
  std::string kernel;
  std::string vector_id;
  std::vector<size_t> dimensions;
  std::vector<uint8_t> xclbin;
  std::string implementation_id;
  std::vector<uint8_t> transaction;
  std::vector<uint8_t> argument1_data;
  std::vector<uint8_t> argument2_data;
  std::vector<bf16> expected;
  std::string weight_provenance;
  std::string reference_kind;
  KernelComparison comparison;
  float absolute_tolerance;
  float relative_tolerance;
  KernelValidationResult validation;
  double device_seconds = 0.0;
  buffer<uint8_t> output;
  buffer<uint8_t> argument1;
  buffer<uint8_t> argument2;
};

KernelCase::KernelCase(
    std::string family, std::string kernel, std::string vector_id,
    std::vector<size_t> dimensions,
    const std::filesystem::path& xclbin_path,
    std::vector<uint8_t> transaction,
    std::vector<uint8_t> argument1,
    std::vector<uint8_t> argument2,
    std::vector<bf16> expected,
    std::string weight_provenance,
    std::string reference_kind,
    KernelComparison comparison,
    float absolute_tolerance,
    float relative_tolerance)
    : impl_(std::make_unique<Impl>(
          std::move(family), std::move(kernel), std::move(vector_id),
          std::move(dimensions), xclbin_path, std::move(transaction),
          std::move(argument1), std::move(argument2), std::move(expected),
          std::move(weight_provenance), std::move(reference_kind),
          comparison, absolute_tolerance, relative_tolerance)) {}

KernelCase::~KernelCase() = default;

std::string KernelCase::name() const {
  return impl_->family + "/" + impl_->kernel + "/" + impl_->vector_id;
}
const std::string& KernelCase::family() const { return impl_->family; }
const std::string& KernelCase::kernel() const { return impl_->kernel; }
const std::string& KernelCase::vector_id() const { return impl_->vector_id; }
const std::string& KernelCase::weight_provenance() const {
  return impl_->weight_provenance;
}
const std::string& KernelCase::reference_kind() const {
  return impl_->reference_kind;
}
const std::string& KernelCase::implementation_id() const {
  return impl_->implementation_id;
}
const char* KernelCase::comparison_name() const {
  switch (impl_->comparison) {
    case KernelComparison::kExactOrdered:
      return "exact-ordered";
    case KernelComparison::kExactMultiset:
      return "exact-multiset";
    case KernelComparison::kTolerance:
      return "tolerance";
  }
  return "unknown";
}
const std::vector<size_t>& KernelCase::dimensions() const {
  return impl_->dimensions;
}
size_t KernelCase::validated_bytes() const {
  return impl_->expected.size() * sizeof(bf16);
}
const KernelValidationResult& KernelCase::validation() const {
  return impl_->validation;
}
float KernelCase::absolute_tolerance() const {
  return impl_->absolute_tolerance;
}
float KernelCase::relative_tolerance() const {
  return impl_->relative_tolerance;
}
double KernelCase::device_seconds() const { return impl_->device_seconds; }

KernelExecutable KernelCase::create_executable() const {
  const hrx_const_byte_span_t xclbin_span = {
      impl_->xclbin.data(), impl_->xclbin.size()};
  hrx_amdxdna_executable_run_t run = hrx_amdxdna_executable_run_default();
  run.transaction = {impl_->transaction.data(), impl_->transaction.size()};
  hrx_amdxdna_executable_entry_point_t entry =
      hrx_amdxdna_executable_entry_point_default();
  entry.name = {"MLIR_AIE", 8};
  entry.runs = &run;
  entry.run_count = 1;
  hrx_amdxdna_executable_create_params_t params =
      hrx_amdxdna_executable_create_params_default();
  params.xclbins = &xclbin_span;
  params.xclbin_count = 1;
  params.entry_points = &entry;
  params.entry_point_count = 1;

  KernelExecutable executable;
  if (hrx::hrx_report(
          hrx_amdxdna_executable_create(
              hrx::rt().dev, &params, &executable.handle),
          "hrx_amdxdna_executable_create") ||
      hrx::hrx_report(
          hrx_executable_lookup_export_by_name(
              executable.handle, "MLIR_AIE", &executable.ordinal),
          "hrx_executable_lookup_export_by_name")) {
    throw std::runtime_error("failed to create executable for " + name());
  }
  return executable;
}

std::vector<bf16> KernelCase::dispatch(
    const KernelExecutable& executable) {
  const size_t output_bytes = validated_bytes();
  impl_->output.memset(0xA5);
  impl_->output.sync_to_device();
  hrx_buffer_ref_t bindings[] = {
      {impl_->output.bo().handle(), kGuardBytes, output_bytes},
      {impl_->argument1.bo().handle(), kGuardBytes,
       impl_->argument1_data.size()},
      {impl_->argument2.bo().handle(), kGuardBytes,
       impl_->argument2_data.size()},
  };
  hrx_dispatch_config_t config = {{1, 1, 1}, {1, 1, 1}, 0};
  const auto device_started = std::chrono::steady_clock::now();
  if (hrx::hrx_report(
          hrx_stream_dispatch(
              hrx::rt().stream, executable.handle, executable.ordinal,
              &config, nullptr, 0, bindings, 3, HRX_DISPATCH_FLAG_NONE),
          "hrx_stream_dispatch") ||
      hrx::hrx_report(
          hrx_stream_synchronize(hrx::rt().stream),
          "hrx_stream_synchronize")) {
    throw std::runtime_error("dispatch failed for " + name());
  }
  impl_->device_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - device_started).count();
  impl_->output.sync_from_device();
  impl_->argument1.sync_from_device();
  impl_->argument2.sync_from_device();
  std::vector<std::string> memory_errors;
  for (std::string error : {
           kernel_validation::check_guards(
               {impl_->output.data(),
                output_bytes + 2 * kGuardBytes},
               output_bytes, kGuardBytes, 0xA5, name() + "/output"),
           kernel_validation::check_immutable(
               {impl_->argument1.data(),
                impl_->argument1_data.size() + 2 * kGuardBytes},
               impl_->argument1_data, kGuardBytes, 0x5A,
               name() + "/argument1"),
           kernel_validation::check_immutable(
               {impl_->argument2.data(),
                impl_->argument2_data.size() + 2 * kGuardBytes},
               impl_->argument2_data, kGuardBytes, 0x5A,
               name() + "/argument2")}) {
    if (!error.empty()) memory_errors.push_back(std::move(error));
  }
  const auto* actual = reinterpret_cast<const bf16*>(
      impl_->output.data() + kGuardBytes);

  KernelValidationResult current;
  size_t first_mismatch = impl_->expected.size();
  if (impl_->comparison == KernelComparison::kExactMultiset) {
    if (impl_->absolute_tolerance != 0.0f ||
        impl_->relative_tolerance != 0.0f) {
      throw std::runtime_error(
          "unordered kernel comparisons must be exact");
    }
    std::array<int32_t, 1u << 16> counts = {};
    for (size_t i = 0; i < impl_->expected.size(); ++i) {
      uint16_t actual_bits;
      uint16_t expected_bits;
      std::memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
      std::memcpy(
          &expected_bits, &impl_->expected[i], sizeof(expected_bits));
      ++counts[actual_bits];
      --counts[expected_bits];
    }
    size_t unmatched = 0;
    for (int32_t count : counts) unmatched += std::abs(count);
    current.mismatches = unmatched / 2;
    if (current.mismatches) {
      current.maximum_absolute_error =
          std::numeric_limits<float>::infinity();
      first_mismatch = 0;
    }
  } else {
    for (size_t i = 0; i < impl_->expected.size(); ++i) {
      const float actual_value = static_cast<float>(actual[i]);
      const float expected_value = static_cast<float>(impl_->expected[i]);
      const float error = std::abs(actual_value - expected_value);
      current.maximum_absolute_error =
          std::max(current.maximum_absolute_error, error);
      if (expected_value != 0.0f) {
        current.maximum_relative_error = std::max(
            current.maximum_relative_error,
            error / std::abs(expected_value));
      }
      const float limit = impl_->comparison == KernelComparison::kTolerance
                              ? impl_->absolute_tolerance +
                                    impl_->relative_tolerance *
                                        std::abs(expected_value)
                              : 0.0f;
      if (limit > 0.0f) {
        current.maximum_tolerance_ratio = std::max(
            current.maximum_tolerance_ratio, error / limit);
      }
      if (!std::isfinite(actual_value) || error > limit) {
        if (first_mismatch == impl_->expected.size()) first_mismatch = i;
        ++current.mismatches;
      }
    }
  }
  impl_->validation.maximum_absolute_error =
      std::max(
          impl_->validation.maximum_absolute_error,
          current.maximum_absolute_error);
  impl_->validation.maximum_relative_error =
      std::max(
          impl_->validation.maximum_relative_error,
          current.maximum_relative_error);
  impl_->validation.maximum_tolerance_ratio =
      std::max(
          impl_->validation.maximum_tolerance_ratio,
          current.maximum_tolerance_ratio);
  impl_->validation.mismatches += current.mismatches;
  if (!memory_errors.empty() || current.mismatches) {
    std::ostringstream detail;
    for (const std::string& error : memory_errors) {
      detail << " memory=[" << error << "]";
    }
    if (current.mismatches) {
      const size_t begin = first_mismatch > 2 ? first_mismatch - 2 : 0;
      const size_t end =
          std::min(first_mismatch + 6, impl_->expected.size());
      for (size_t i = begin; i < end; ++i) {
        detail << " [" << i << ":" << static_cast<float>(actual[i])
               << "/" << static_cast<float>(impl_->expected[i]) << "]";
      }
    }
    throw std::runtime_error(
        name() + " produced " + std::to_string(current.mismatches) +
        " numeric mismatches; max abs error " +
        std::to_string(current.maximum_absolute_error) + detail.str());
  }

  return {actual, actual + impl_->expected.size()};
}
