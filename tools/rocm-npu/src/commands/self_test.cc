// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm_npu/command.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "rocm_npu/hrx_loader.h"
#include "rocm_npu/platform.h"
#include "rocm_npu/product.h"
#include "rocm_npu/sha256.h"

#include <hrx_amdxdna.h>

namespace rocm_npu {
namespace {

constexpr hrx_memory_type_t kMappedBufferMemory =
    HRX_MEMORY_TYPE_HOST_VISIBLE | HRX_MEMORY_TYPE_HOST_CACHED |
    HRX_MEMORY_TYPE_DEVICE_VISIBLE;

class CleanupGuard {
 public:
  explicit CleanupGuard(const std::function<void()>& cleanup)
      : cleanup_(cleanup) {}
  ~CleanupGuard() { cleanup_(); }

 private:
  const std::function<void()>& cleanup_;
};

std::optional<std::vector<std::uint8_t>> ReadBytes(
    const std::filesystem::path& path, std::string& error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "could not open " + PathToUtf8(path);
    return std::nullopt;
  }
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input),
                                   std::istreambuf_iterator<char>());
}

const nlohmann::json* FindFixture(const std::filesystem::path& root,
                                  const std::string& target,
                                  nlohmann::json& storage,
                                  std::filesystem::path& fixture_root,
                                  std::string& error) {
  std::error_code filesystem_error;
  for (std::filesystem::directory_iterator iterator(root, filesystem_error),
       end;
       !filesystem_error && iterator != end;
       iterator.increment(filesystem_error)) {
    if (!iterator->is_directory()) {
      continue;
    }
    const auto manifest_path = iterator->path() / "manifest.json";
    auto manifest = ReadJsonFile(manifest_path, error);
    if (!manifest) {
      if (std::filesystem::exists(manifest_path)) {
        return nullptr;
      }
      error.clear();
      continue;
    }
    const auto target_field = manifest->find("target");
    if (target_field != manifest->end() && target_field->is_string() &&
        target_field->get_ref<const std::string&>() == target) {
      storage = std::move(*manifest);
      fixture_root = iterator->path();
      return &storage;
    }
  }
  if (filesystem_error) {
    error = filesystem_error.message();
  } else {
    error = "no packaged self-test fixture matches target " + target;
  }
  return nullptr;
}

bool IsNonnegativeInteger(const nlohmann::json& value) {
  return value.is_number_unsigned() ||
         (value.is_number_integer() && value.get<std::int64_t>() >= 0);
}

bool IsInt32Integer(const nlohmann::json& value) {
  if (value.is_number_unsigned()) {
    return value.get<std::uint64_t>() <=
           static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  }
  return value.is_number_integer() &&
         value.get<std::int64_t>() >=
             static_cast<std::int64_t>(
                 std::numeric_limits<std::int32_t>::min()) &&
         value.get<std::int64_t>() <=
             static_cast<std::int64_t>(
                 std::numeric_limits<std::int32_t>::max());
}

std::optional<std::int32_t> ReadInt32Field(const nlohmann::json& object,
                                           const char* key,
                                           std::string& error) {
  const auto field = object.find(key);
  if (field == object.end() || !IsInt32Integer(*field)) {
    error = std::string("generator field ") + key + " must be an int32 value";
    return std::nullopt;
  }
  if (field->is_number_unsigned()) {
    return static_cast<std::int32_t>(field->get<std::uint64_t>());
  }
  return static_cast<std::int32_t>(field->get<std::int64_t>());
}

void AppendInt32LittleEndian(std::vector<std::uint8_t>& bytes,
                             std::int32_t value) {
  const auto bits = static_cast<std::uint32_t>(value);
  bytes.push_back(static_cast<std::uint8_t>(bits & 0xff));
  bytes.push_back(static_cast<std::uint8_t>((bits >> 8) & 0xff));
  bytes.push_back(static_cast<std::uint8_t>((bits >> 16) & 0xff));
  bytes.push_back(static_cast<std::uint8_t>((bits >> 24) & 0xff));
}

std::int32_t ReadInt32LittleEndian(const std::vector<std::uint8_t>& bytes,
                                   std::size_t offset) {
  const std::uint32_t bits =
      static_cast<std::uint32_t>(bytes[offset]) |
      (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
      (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
      (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
  if (bits <=
      static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
    return static_cast<std::int32_t>(bits);
  }
  return static_cast<std::int32_t>(static_cast<std::int64_t>(bits) -
                                   (1LL << 32));
}

std::optional<std::vector<std::uint8_t>> GenerateIotaI32(
    std::uint64_t element_count, std::int32_t start, std::string& error) {
  const std::uint64_t max_distance =
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max() -
                                 static_cast<std::int64_t>(start));
  if (element_count > max_distance + 1) {
    error = "iota_i32 generator overflows int32";
    return std::nullopt;
  }
  if (element_count >
      std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t)) {
    error = "generated buffer is too large";
    return std::nullopt;
  }
  std::vector<std::uint8_t> bytes;
  bytes.reserve(static_cast<std::size_t>(element_count) * sizeof(std::int32_t));
  for (std::uint64_t index = 0; index < element_count; ++index) {
    AppendInt32LittleEndian(
        bytes, static_cast<std::int32_t>(static_cast<std::int64_t>(start) +
                                         static_cast<std::int64_t>(index)));
  }
  return bytes;
}

std::optional<std::vector<std::uint8_t>> GenerateAddI32Scalar(
    const std::vector<std::uint8_t>& source, std::int32_t scalar,
    std::uint64_t element_count, std::string& error) {
  if (source.size() != element_count * sizeof(std::int32_t)) {
    error = "add_i32_scalar source size does not match buffer shape";
    return std::nullopt;
  }
  std::vector<std::uint8_t> bytes;
  bytes.reserve(source.size());
  for (std::uint64_t index = 0; index < element_count; ++index) {
    const std::int32_t value = ReadInt32LittleEndian(
        source, static_cast<std::size_t>(index) * sizeof(std::int32_t));
    const std::int64_t result =
        static_cast<std::int64_t>(value) + static_cast<std::int64_t>(scalar);
    if (result < static_cast<std::int64_t>(
                     std::numeric_limits<std::int32_t>::min()) ||
        result > static_cast<std::int64_t>(
                     std::numeric_limits<std::int32_t>::max())) {
      error = "add_i32_scalar generator overflows int32";
      return std::nullopt;
    }
    AppendInt32LittleEndian(bytes, static_cast<std::int32_t>(result));
  }
  return bytes;
}

std::optional<std::vector<std::uint8_t>> GenerateFixtureBuffer(
    const nlohmann::json& descriptor,
    const std::map<std::string, std::vector<std::uint8_t>>& generated_buffers,
    std::uint64_t element_count, std::string& error) {
  const auto generator_field = descriptor.find("generator");
  if (generator_field == descriptor.end() || !generator_field->is_object()) {
    error = "buffer generator must be an object";
    return std::nullopt;
  }
  const auto kind_field = generator_field->find("kind");
  if (kind_field == generator_field->end() || !kind_field->is_string()) {
    error = "buffer generator kind must be a string";
    return std::nullopt;
  }
  const std::string& kind = kind_field->get_ref<const std::string&>();
  const std::string& direction =
      descriptor["direction"].get_ref<const std::string&>();
  if (kind == "iota_i32") {
    if (direction != "input") {
      error = "iota_i32 generator is only valid for input buffers";
      return std::nullopt;
    }
    const auto start = ReadInt32Field(*generator_field, "start", error);
    if (!start) {
      return std::nullopt;
    }
    return GenerateIotaI32(element_count, *start, error);
  }
  if (kind == "add_i32_scalar") {
    if (direction != "output") {
      error = "add_i32_scalar generator is only valid for output buffers";
      return std::nullopt;
    }
    const auto source_field = generator_field->find("source");
    if (source_field == generator_field->end() || !source_field->is_string()) {
      error = "add_i32_scalar source must be a string";
      return std::nullopt;
    }
    const auto source =
        generated_buffers.find(source_field->get_ref<const std::string&>());
    if (source == generated_buffers.end()) {
      error = "add_i32_scalar source buffer has not been generated";
      return std::nullopt;
    }
    const auto scalar = ReadInt32Field(*generator_field, "scalar", error);
    if (!scalar) {
      return std::nullopt;
    }
    return GenerateAddI32Scalar(source->second, *scalar, element_count, error);
  }
  error = "unsupported buffer generator kind " + kind;
  return std::nullopt;
}

}  // namespace

bool HasValidSelfTestFixtureTypes(const nlohmann::json& fixture) {
  const auto is_string = [](const nlohmann::json& object, const char* key) {
    const auto field = object.find(key);
    return field != object.end() && field->is_string();
  };
  const auto is_unsigned = [](const nlohmann::json& object, const char* key) {
    const auto field = object.find(key);
    return field != object.end() && IsNonnegativeInteger(*field);
  };
  const auto is_int32 = [](const nlohmann::json& object, const char* key) {
    const auto field = object.find(key);
    return field != object.end() && IsInt32Integer(*field);
  };
  const auto has_generator = [&](const nlohmann::json& descriptor,
                                 const std::string& direction) {
    const auto generator = descriptor.find("generator");
    if (generator == descriptor.end() || !generator->is_object() ||
        !is_string(*generator, "kind")) {
      return false;
    }
    const std::string& kind =
        (*generator)["kind"].get_ref<const std::string&>();
    if (direction == "input") {
      return kind == "iota_i32" && is_int32(*generator, "start");
    }
    return kind == "add_i32_scalar" && is_string(*generator, "source") &&
           is_int32(*generator, "scalar");
  };
  if (!fixture.is_object() || !is_unsigned(fixture, "schema_version") ||
      !is_unsigned(fixture, "xadx_version") || !is_string(fixture, "name") ||
      !is_string(fixture, "target") || !is_string(fixture, "export") ||
      !fixture.contains("producer") || !fixture["producer"].is_object() ||
      !fixture["producer"].contains("artifacts") ||
      !fixture["producer"]["artifacts"].is_object() ||
      !fixture.contains("buffers") || !fixture["buffers"].is_array() ||
      !fixture.contains("dispatch") || !fixture["dispatch"].is_object()) {
    return false;
  }
  const auto& artifacts = fixture["producer"]["artifacts"];
  if (!is_string(artifacts, "xadx_sha256") ||
      !is_string(artifacts, "input_sha256") ||
      !is_string(artifacts, "expected_sha256")) {
    return false;
  }
  for (const auto& descriptor : fixture["buffers"]) {
    if (!descriptor.is_object() || !is_string(descriptor, "name") ||
        !is_string(descriptor, "direction") ||
        !is_string(descriptor, "dtype") ||
        !is_unsigned(descriptor, "byte_length") ||
        !descriptor.contains("shape") || !descriptor["shape"].is_array()) {
      return false;
    }
    const std::string& direction =
        descriptor["direction"].get_ref<const std::string&>();
    if ((direction != "input" && direction != "output") ||
        descriptor.contains("input") || descriptor.contains("expected") ||
        !has_generator(descriptor, direction)) {
      return false;
    }
    if (!std::all_of(descriptor["shape"].begin(), descriptor["shape"].end(),
                     [](const nlohmann::json& value) {
                       return IsNonnegativeInteger(value);
                     })) {
      return false;
    }
  }
  const auto& dispatch = fixture["dispatch"];
  const auto is_triplet = [](const nlohmann::json& value) {
    return value.is_array() && value.size() == 3 &&
           std::all_of(value.begin(), value.end(),
                       [](const nlohmann::json& item) {
                         return IsNonnegativeInteger(item);
                       });
  };
  return dispatch.contains("workgroup_count") &&
         is_triplet(dispatch["workgroup_count"]) &&
         dispatch.contains("workgroup_size") &&
         is_triplet(dispatch["workgroup_size"]) &&
         is_unsigned(dispatch, "subgroup_size") &&
         is_unsigned(dispatch, "constant_byte_length") &&
         dispatch.contains("export_binding_count") &&
         (dispatch["export_binding_count"].is_null() ||
          IsNonnegativeInteger(dispatch["export_binding_count"]));
}

Report RunSelfTest(const Invocation& invocation) {
  Report doctor = RunDoctor(invocation);
  Report report = MakeReport("self-test");
  report.checks = std::move(doctor.checks);
  report.data["doctor"] = std::move(doctor.data);
  report.status = doctor.status;
  if (doctor.status != Status::kPass) {
    AddCheck(report, {"self_test.prerequisites",
                      Status::kFail,
                      "required doctor checks did not pass",
                      {}});
    return report;
  }

  ProbeResult platform = CollectPlatformProbe();
  if (platform.target_family.empty()) {
    AddCheck(report, {"self_test.fixture",
                      Status::kFail,
                      "detected device target family is unknown",
                      {}});
    return report;
  }
  if (platform.target_family != "npu2") {
    AddCheck(report,
             {"self_test.fixture",
              Status::kFail,
              "the packaged self-test is supported only on NPU2 devices",
              {{"detected_target", platform.target_family},
               {"supported_targets", nlohmann::json::array({"npu2"})}}});
    return report;
  }

  nlohmann::json manifest;
  std::filesystem::path fixture_root;
  std::string error;
  const nlohmann::json* fixture =
      FindFixture(SelfTestDataRoot(), platform.target_family, manifest,
                  fixture_root, error);
  if (fixture == nullptr) {
    AddCheck(report, {"self_test.fixture", Status::kFail, error, {}});
    return report;
  }
  if (!HasValidSelfTestFixtureTypes(*fixture)) {
    AddCheck(report, {"self_test.fixture",
                      Status::kError,
                      "fixture manifest has invalid field types",
                      {{"path", PathToUtf8(fixture_root / "manifest.json")}}});
    return report;
  }
  const auto unsigned_field_equals = [](const nlohmann::json& object,
                                        const char* key,
                                        std::uint64_t expected) {
    const auto field = object.find(key);
    return field != object.end() && IsNonnegativeInteger(*field) &&
           field->get<std::uint64_t>() == expected;
  };
  const auto nonempty_string_field = [](const nlohmann::json& object,
                                        const char* key) {
    const auto field = object.find(key);
    return field != object.end() && field->is_string() &&
           !field->get_ref<const std::string&>().empty();
  };
  if (!unsigned_field_equals(*fixture, "schema_version", 1) ||
      !unsigned_field_equals(*fixture, "xadx_version", 1) ||
      !fixture->contains("buffers") || !(*fixture)["buffers"].is_array() ||
      (*fixture)["buffers"].size() != 2 || !fixture->contains("dispatch") ||
      !(*fixture)["dispatch"].is_object() ||
      !nonempty_string_field(*fixture, "name") ||
      !nonempty_string_field(*fixture, "export") ||
      !fixture->contains("producer") || !(*fixture)["producer"].is_object() ||
      !(*fixture)["producer"].contains("artifacts") ||
      !(*fixture)["producer"]["artifacts"].is_object()) {
    AddCheck(report, {"self_test.fixture",
                      Status::kError,
                      "fixture manifest has an invalid structure",
                      {{"path", PathToUtf8(fixture_root / "manifest.json")}}});
    return report;
  }

  auto xadx = ReadBytes(fixture_root / "kernel.xadx", error);
  if (!xadx || xadx->empty()) {
    AddCheck(report, {"self_test.load",
                      Status::kError,
                      error.empty() ? "kernel.xadx is empty" : error,
                      {}});
    return report;
  }
  const nlohmann::json& artifact_hashes = (*fixture)["producer"]["artifacts"];
  if (!nonempty_string_field(artifact_hashes, "xadx_sha256") ||
      !nonempty_string_field(artifact_hashes, "input_sha256") ||
      !nonempty_string_field(artifact_hashes, "expected_sha256") ||
      artifact_hashes["xadx_sha256"].get_ref<const std::string&>() !=
          Sha256Hex(*xadx)) {
    AddCheck(report, {"self_test.fixture",
                      Status::kError,
                      "kernel.xadx does not match fixture provenance",
                      {}});
    return report;
  }

  std::set<std::string> buffer_names;
  std::map<std::string, std::vector<std::uint8_t>> generated_buffers;
  int input_count = 0;
  int output_count = 0;
  for (const auto& descriptor : (*fixture)["buffers"]) {
    if (!descriptor.is_object()) {
      AddCheck(report, {"self_test.fixture",
                        Status::kError,
                        "fixture buffer descriptor is not an object",
                        {}});
      return report;
    }
    const auto byte_length_field = descriptor.find("byte_length");
    if (!nonempty_string_field(descriptor, "name") ||
        !nonempty_string_field(descriptor, "direction") ||
        !nonempty_string_field(descriptor, "dtype") ||
        byte_length_field == descriptor.end() ||
        !IsNonnegativeInteger(*byte_length_field) ||
        byte_length_field->get<std::uint64_t>() >
            std::numeric_limits<std::size_t>::max() ||
        !descriptor.contains("shape") || !descriptor["shape"].is_array()) {
      AddCheck(report, {"self_test.fixture",
                        Status::kError,
                        "fixture buffer descriptor has invalid field types",
                        {}});
      return report;
    }
    const std::string& name = descriptor["name"].get_ref<const std::string&>();
    const std::string& direction =
        descriptor["direction"].get_ref<const std::string&>();
    if ((direction != "input" && direction != "output") ||
        descriptor.contains("input") || descriptor.contains("expected") ||
        !descriptor.contains("generator") ||
        !descriptor["generator"].is_object()) {
      AddCheck(report, {"self_test.fixture",
                        Status::kError,
                        "fixture buffer direction or generator is invalid",
                        {{"buffer", name}}});
      return report;
    }
    const nlohmann::json& shape = descriptor["shape"];
    std::uint64_t element_count = 1;
    bool valid_shape = shape.is_array() && !shape.empty();
    if (valid_shape) {
      for (const auto& dimension : shape) {
        if (!IsNonnegativeInteger(dimension) ||
            dimension.get<std::uint64_t>() == 0 ||
            element_count > std::numeric_limits<std::uint64_t>::max() /
                                dimension.get<std::uint64_t>()) {
          valid_shape = false;
          break;
        }
        element_count *= dimension.get<std::uint64_t>();
      }
    }
    const std::size_t byte_length = byte_length_field->get<std::size_t>();
    const bool valid_descriptor =
        !name.empty() && buffer_names.insert(name).second &&
        descriptor["dtype"].get_ref<const std::string&>() == "int32" &&
        valid_shape &&
        element_count <=
            std::numeric_limits<std::uint64_t>::max() / sizeof(std::int32_t) &&
        element_count * sizeof(std::int32_t) == byte_length;
    if (!valid_descriptor) {
      AddCheck(report, {"self_test.fixture",
                        Status::kError,
                        "fixture buffer descriptor is invalid",
                        {{"buffer", name}}});
      return report;
    }
    input_count += direction == "input" ? 1 : 0;
    output_count += direction == "output" ? 1 : 0;
    std::string generator_error;
    auto bytes = GenerateFixtureBuffer(descriptor, generated_buffers,
                                       element_count, generator_error);
    const char* hash_name =
        direction == "input" ? "input_sha256" : "expected_sha256";
    if (!bytes || bytes->size() != byte_length ||
        artifact_hashes[hash_name].get_ref<const std::string&>() !=
            Sha256Hex(*bytes)) {
      AddCheck(report,
               {"self_test.fixture",
                Status::kError,
                "fixture generated buffer does not match its provenance hash",
                {{"buffer", name}, {"error", generator_error}}});
      return report;
    }
    generated_buffers.emplace(name, std::move(*bytes));
  }
  const auto& dispatch = (*fixture)["dispatch"];
  const auto valid_triplet = [](const nlohmann::json& value) {
    return value.is_array() && value.size() == 3 &&
           std::all_of(value.begin(), value.end(),
                       [](const nlohmann::json& item) {
                         return IsNonnegativeInteger(item) &&
                                item.get<std::uint64_t>() > 0 &&
                                item.get<std::uint64_t>() <=
                                    std::numeric_limits<std::uint32_t>::max();
                       });
  };
  const bool binding_count_valid =
      dispatch.contains("export_binding_count") &&
      (dispatch["export_binding_count"].is_null() ||
       (IsNonnegativeInteger(dispatch["export_binding_count"]) &&
        dispatch["export_binding_count"].get<std::uint64_t>() <=
            std::numeric_limits<std::uint32_t>::max()));
  if (input_count != 1 || output_count != 1 ||
      !dispatch.contains("workgroup_count") ||
      !valid_triplet(dispatch["workgroup_count"]) ||
      !dispatch.contains("workgroup_size") ||
      !valid_triplet(dispatch["workgroup_size"]) ||
      !dispatch.contains("constant_byte_length") ||
      !IsNonnegativeInteger(dispatch["constant_byte_length"]) ||
      dispatch["constant_byte_length"].get<std::uint64_t>() != 0 ||
      !binding_count_valid || !dispatch.contains("subgroup_size") ||
      !IsNonnegativeInteger(dispatch["subgroup_size"]) ||
      dispatch["subgroup_size"].get<std::uint64_t>() >
          std::numeric_limits<std::uint32_t>::max()) {
    AddCheck(report, {"self_test.fixture",
                      Status::kError,
                      "fixture dispatch contract is invalid",
                      {}});
    return report;
  }
  AddCheck(report, {"self_test.fixture",
                    Status::kPass,
                    "matching packaged fixture and provenance validated",
                    {{"path", PathToUtf8(fixture_root)},
                     {"target", platform.target_family}}});

  HrxLoadResult load = HrxLibrary::Load(invocation.runtime_path);
  if (!load.library) {
    AddCheck(report, {"self_test.load",
                      Status::kFail,
                      "HRX runtime could not be loaded",
                      {{"errors", load.errors},
                       {"missing_symbols", load.missing_symbols}}});
    return report;
  }

  HrxLibrary& library = *load.library;
  const HrxFunctions& functions = library.functions();
  hrx_device_t device = nullptr;
  hrx_stream_t stream = nullptr;
  hrx_executable_t executable = nullptr;
  bool initialized = false;
  struct Buffer {
    hrx_buffer_t handle = nullptr;
    void* mapping = nullptr;
    std::size_t size = 0;
    bool mapped = false;
    std::string direction;
    std::vector<std::uint8_t> expected;
  };
  std::vector<Buffer> buffers;

  const auto consume = [&](HrxStatus status, const std::string& operation) {
    return library.ConsumeStatus(status, operation);
  };
  bool cleaned = false;
  const std::function<void()> cleanup = [&]() {
    if (cleaned) {
      return;
    }
    cleaned = true;
    const bool had_resources = initialized || device != nullptr ||
                               stream != nullptr || executable != nullptr ||
                               !buffers.empty();
    std::vector<std::string> cleanup_errors;
    for (auto iterator = buffers.rbegin(); iterator != buffers.rend();
         ++iterator) {
      if (iterator->mapped) {
        std::string cleanup_error = consume(
            functions.buffer_unmap(iterator->handle), "hrx_buffer_unmap");
        if (!cleanup_error.empty()) {
          cleanup_errors.push_back(std::move(cleanup_error));
        }
      }
      if (iterator->handle != nullptr) {
        functions.buffer_release(iterator->handle);
      }
    }
    if (executable != nullptr) {
      functions.executable_release(executable);
    }
    if (stream != nullptr) {
      functions.stream_release(stream);
    }
    if (device != nullptr) {
      functions.device_release(device);
    }
    if (initialized) {
      std::string cleanup_error =
          consume(functions.gpu_shutdown(), "hrx_gpu_shutdown");
      if (!cleanup_error.empty()) {
        cleanup_errors.push_back(std::move(cleanup_error));
      }
    }
    if (had_resources) {
      AddCheck(report,
               {"self_test.cleanup",
                cleanup_errors.empty() ? Status::kPass : Status::kFail,
                cleanup_errors.empty() ? "self-test resources released cleanly"
                                       : "self-test cleanup failed",
                {{"errors", cleanup_errors}}});
    }
  };
  CleanupGuard cleanup_guard(cleanup);
  const auto fail = [&](const std::string& operation,
                        const std::string& detail) {
    AddCheck(report, {"self_test.dispatch",
                      Status::kFail,
                      detail,
                      {{"operation", operation}}});
  };

  std::string status_error =
      consume(functions.gpu_initialize(0), "hrx_gpu_initialize");
  if (!status_error.empty()) {
    fail("initialize", status_error);
    cleanup();
    return report;
  }
  initialized = true;
  status_error =
      consume(functions.gpu_device_get(0, &device), "hrx_gpu_device_get");
  if (!status_error.empty()) {
    fail("device_get", status_error);
    cleanup();
    return report;
  }
  status_error =
      consume(functions.stream_create(device, 0, &stream), "hrx_stream_create");
  if (!status_error.empty()) {
    fail("stream_create", status_error);
    cleanup();
    return report;
  }
  status_error = consume(functions.executable_load_data(
                             device, xadx->data(), xadx->size(),
                             HRX_AMDXDNA_EXECUTABLE_TARGET_FAMILY,
                             HRX_AMDXDNA_EXECUTABLE_TARGET_KEY, &executable),
                         "hrx_executable_load_data");
  if (!status_error.empty()) {
    fail("executable_load", status_error);
    cleanup();
    return report;
  }

  std::uint32_t export_ordinal = 0;
  const std::string& export_name =
      (*fixture)["export"].get_ref<const std::string&>();
  status_error = consume(functions.executable_lookup_export_by_name(
                             executable, export_name.c_str(), &export_ordinal),
                         "hrx_executable_lookup_export_by_name");
  if (!status_error.empty()) {
    fail("export_lookup", status_error);
    cleanup();
    return report;
  }
  hrx_executable_export_info_t export_info = {};
  status_error = consume(functions.executable_export_info(
                             executable, export_ordinal, &export_info),
                         "hrx_executable_export_info");
  if (!status_error.empty()) {
    fail("export_info", status_error);
    cleanup();
    return report;
  }
  const auto& workgroup_size = dispatch["workgroup_size"];
  const bool binding_count_mismatch =
      !dispatch["export_binding_count"].is_null() &&
      export_info.binding_count !=
          dispatch["export_binding_count"].get<std::uint32_t>();
  if (binding_count_mismatch ||
      export_info.constant_byte_length !=
          dispatch["constant_byte_length"].get<std::uint32_t>() ||
      !std::equal(std::begin(export_info.workgroup_size),
                  std::end(export_info.workgroup_size), workgroup_size.begin(),
                  [](std::uint32_t actual, const nlohmann::json& expected) {
                    return actual == expected.get<std::uint32_t>();
                  })) {
    AddCheck(report,
             {"self_test.dispatch",
              Status::kFail,
              "loaded export metadata does not match the fixture contract",
              {{"operation", "export_info"},
               {"actual",
                {{"binding_count", export_info.binding_count},
                 {"constant_byte_length", export_info.constant_byte_length},
                 {"workgroup_size",
                  {export_info.workgroup_size[0], export_info.workgroup_size[1],
                   export_info.workgroup_size[2]}}}},
               {"expected",
                {{"binding_count", dispatch["export_binding_count"]},
                 {"constant_byte_length", dispatch["constant_byte_length"]},
                 {"workgroup_size", workgroup_size}}}}});
    cleanup();
    return report;
  }

  buffers.reserve((*fixture)["buffers"].size());
  for (const auto& descriptor : (*fixture)["buffers"]) {
    if (!descriptor.is_object()) {
      fail("manifest", "buffer descriptor is not an object");
      cleanup();
      return report;
    }
    const std::string& direction =
        descriptor["direction"].get_ref<const std::string&>();
    const std::string& name = descriptor["name"].get_ref<const std::string&>();
    const auto bytes = generated_buffers.find(name);
    const std::size_t size = descriptor["byte_length"].get<std::size_t>();
    if (bytes == generated_buffers.end() || size == 0 ||
        bytes->second.size() != size ||
        (direction != "input" && direction != "output")) {
      fail("manifest", "generated buffer or byte_length is invalid");
      cleanup();
      return report;
    }
    Buffer buffer;
    buffer.size = size;
    buffer.direction = direction;
    if (direction == "output") {
      buffer.expected = bytes->second;
    }
    status_error = consume(
        functions.buffer_allocate(
            stream, size, kMappedBufferMemory,
            HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
            &buffer.handle),
        "hrx_buffer_allocate");
    if (!status_error.empty()) {
      fail("buffer_allocate", status_error);
      cleanup();
      return report;
    }
    buffers.push_back(std::move(buffer));
    Buffer& stored = buffers.back();
    status_error = consume(
        functions.buffer_map_with_mode(
            stored.handle, HRX_MAPPING_MODE_PERSISTENT,
            HRX_MAP_READ | HRX_MAP_WRITE, 0, stored.size, &stored.mapping),
        "hrx_buffer_map_with_mode");
    if (!status_error.empty()) {
      fail("buffer_map", status_error);
      cleanup();
      return report;
    }
    stored.mapped = true;
    if (direction == "input") {
      std::memcpy(stored.mapping, bytes->second.data(), bytes->second.size());
      status_error =
          consume(functions.buffer_flush_range(stored.handle, 0, stored.size),
                  "hrx_buffer_flush_range");
      if (!status_error.empty()) {
        fail("buffer_flush", status_error);
        cleanup();
        return report;
      }
    }
  }

  std::vector<HrxBufferRef> bindings;
  bindings.reserve(buffers.size());
  for (const Buffer& buffer : buffers) {
    bindings.push_back({buffer.handle, 0, buffer.size});
  }
  const auto& workgroup_count = dispatch["workgroup_count"];
  HrxDispatchConfig configuration = {
      {workgroup_count[0].get<std::uint32_t>(),
       workgroup_count[1].get<std::uint32_t>(),
       workgroup_count[2].get<std::uint32_t>()},
      {workgroup_size[0].get<std::uint32_t>(),
       workgroup_size[1].get<std::uint32_t>(),
       workgroup_size[2].get<std::uint32_t>()},
      dispatch["subgroup_size"].get<std::uint32_t>()};
  status_error = consume(functions.stream_dispatch(
                             stream, executable, export_ordinal, &configuration,
                             nullptr, 0, bindings.data(), bindings.size(), 0),
                         "hrx_stream_dispatch");
  if (status_error.empty()) {
    status_error =
        consume(functions.stream_synchronize(stream), "hrx_stream_synchronize");
  }
  if (!status_error.empty()) {
    fail("dispatch", status_error);
    cleanup();
    return report;
  }

  bool matches = true;
  for (Buffer& buffer : buffers) {
    if (buffer.direction != "output") {
      continue;
    }
    status_error = consume(
        functions.buffer_invalidate_range(buffer.handle, 0, buffer.size),
        "hrx_buffer_invalidate_range");
    if (!status_error.empty()) {
      fail("buffer_invalidate", status_error);
      cleanup();
      return report;
    }
    matches &=
        std::memcmp(buffer.mapping, buffer.expected.data(), buffer.size) == 0;
  }
  cleanup();
  AddCheck(report,
           {"self_test.output",
            matches ? Status::kPass : Status::kFail,
            matches ? "output bytes match the packaged golden result"
                    : "output bytes do not match the packaged golden result",
            {{"fixture", (*fixture)["name"].get_ref<const std::string&>()},
             {"target", platform.target_family}}});
  return report;
}

}  // namespace rocm_npu
