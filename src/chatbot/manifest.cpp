#include <snnbase_experiments/chatbot/manifest.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace snnbase_experiments::chatbot {
namespace {

constexpr std::array<std::uint32_t, 64> round_constants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

class Sha256 {
 public:
  void update(std::span<const std::uint8_t> bytes) {
    if (bytes.size() >
        (std::numeric_limits<std::uint64_t>::max() - byte_count_)) {
      throw std::overflow_error("SHA-256 input is too large");
    }
    byte_count_ += static_cast<std::uint64_t>(bytes.size());
    for (const auto byte : bytes) {
      buffer_[buffer_size_++] = byte;
      if (buffer_size_ == buffer_.size()) {
        transform(buffer_);
        buffer_size_ = 0;
      }
    }
  }

  Sha256Digest finish() {
    const auto bit_count = byte_count_ * 8U;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56U) {
      std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
                buffer_.end(), 0U);
      transform(buffer_);
      buffer_size_ = 0;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
              buffer_.begin() + 56, 0U);
    for (std::size_t index = 0; index < 8; ++index) {
      buffer_[63U - index] =
          static_cast<std::uint8_t>(bit_count >> (index * 8U));
    }
    transform(buffer_);

    Sha256Digest digest{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      for (std::size_t byte = 0; byte < 4; ++byte) {
        digest[word * 4U + byte] = static_cast<std::uint8_t>(
            state_[word] >> ((3U - byte) * 8U));
      }
    }
    return digest;
  }

 private:
  void transform(const std::array<std::uint8_t, 64>& block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const auto offset = index * 4U;
      words[index] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                     (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
                     (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
                     static_cast<std::uint32_t>(block[offset + 3U]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const auto s0 = std::rotr(words[index - 15U], 7) ^
                      std::rotr(words[index - 15U], 18) ^
                      (words[index - 15U] >> 3U);
      const auto s1 = std::rotr(words[index - 2U], 17) ^
                      std::rotr(words[index - 2U], 19) ^
                      (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }

    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const auto sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const auto choose = (e & f) ^ ((~e) & g);
      const auto temporary1 =
          h + sum1 + choose + round_constants[index] + words[index];
      const auto sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_{};
  std::uint64_t byte_count_{};
};

std::string digest_hex(const Sha256Digest& digest) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(digest.size() * 2U, '0');
  for (std::size_t index = 0; index < digest.size(); ++index) {
    result[index * 2U] = digits[digest[index] >> 4U];
    result[index * 2U + 1U] = digits[digest[index] & 0x0fU];
  }
  return result;
}

bool is_lower_hex(std::string_view value, std::size_t length) noexcept {
  return value.size() == length &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool is_utc_timestamp(std::string_view value) noexcept {
  if (value.size() != 20U || value[4] != '-' || value[7] != '-' ||
      value[10] != 'T' || value[13] != ':' || value[16] != ':' ||
      value[19] != 'Z') {
    return false;
  }
  for (const auto index : {0U, 1U, 2U, 3U, 5U, 6U, 8U, 9U, 11U, 12U, 14U,
                           15U, 17U, 18U}) {
    if (value[index] < '0' || value[index] > '9') {
      return false;
    }
  }
  return true;
}

std::string json_escape(std::string_view value) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(value.size());
  for (const auto character : value) {
    const auto byte = static_cast<unsigned char>(character);
    switch (character) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\b':
        result += "\\b";
        break;
      case '\f':
        result += "\\f";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          result += "\\u00";
          result.push_back(digits[byte >> 4U]);
          result.push_back(digits[byte & 0x0fU]);
        } else {
          result.push_back(character);
        }
    }
  }
  return result;
}

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::invalid_argument(std::string(message));
  }
}

void validate_file(const FileDigest& file, std::string_view label) {
  require(!file.path.empty(), std::string(label) + " path is empty");
  require(is_sha256(file.sha256),
          std::string(label) + " SHA-256 must be 64 lowercase hex digits");
}

void validate_resource(const PinnedResource& resource,
                       std::string_view label) {
  require(resource.repository == qwen3_reference_repository,
          std::string(label) + " repository is not the Phase 0 Qwen pin");
  require(resource.revision == qwen3_reference_revision,
          std::string(label) + " revision is not the Phase 0 Qwen pin");
  require(resource.license == "Apache-2.0",
          std::string(label) + " license must be recorded as Apache-2.0");
  require(!resource.files.empty(),
          std::string(label) + " must include consumed file hashes");
  std::unordered_set<std::string_view> paths;
  for (const auto& file : resource.files) {
    validate_file(file, label);
    require(paths.insert(file.path).second,
            std::string(label) + " contains a duplicate file path");
  }
}

bool contains_file(const PinnedResource& resource, std::string_view path,
                   std::string_view sha256) {
  return std::any_of(resource.files.begin(), resource.files.end(),
                     [path, sha256](const FileDigest& file) {
                       return file.path == path && file.sha256 == sha256;
                     });
}

void write_quoted(std::ostream& output, std::string_view value) {
  output << '"' << json_escape(value) << '"';
}

void write_file(std::ostream& output, const FileDigest& file) {
  output << "{\"path\":";
  write_quoted(output, file.path);
  output << ",\"sha256\":";
  write_quoted(output, file.sha256);
  output << ",\"bytes\":" << file.bytes << '}';
}

void write_repository(std::ostream& output,
                      const RepositoryState& repository) {
  output << "{\"path\":";
  write_quoted(output, repository.path);
  output << ",\"revision\":";
  write_quoted(output, repository.revision);
  output << ",\"dirty\":" << (repository.dirty ? "true" : "false") << '}';
}

void write_resource(std::ostream& output, const PinnedResource& resource) {
  output << "{\"repository\":";
  write_quoted(output, resource.repository);
  output << ",\"revision\":";
  write_quoted(output, resource.revision);
  output << ",\"license\":";
  write_quoted(output, resource.license);
  output << ",\"files\":[";
  for (std::size_t index = 0; index < resource.files.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    write_file(output, resource.files[index]);
  }
  output << "]}";
}

}  // namespace

Sha256Digest sha256_digest(std::string_view bytes) {
  Sha256 hash;
  hash.update(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
  return hash.finish();
}

std::string sha256_hex(std::string_view bytes) {
  return digest_hex(sha256_digest(bytes));
}

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open file for SHA-256: " +
                             path.string());
  }
  Sha256 hash;
  std::array<char, 64U * 1024U> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      hash.update(std::span<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t*>(buffer.data()),
          static_cast<std::size_t>(count)));
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("failed while hashing file: " + path.string());
  }
  return digest_hex(hash.finish());
}

bool is_sha256(std::string_view value) noexcept {
  return is_lower_hex(value, 64U);
}

bool is_git_revision(std::string_view value) noexcept {
  return is_lower_hex(value, 40U);
}

FileDigest digest_file(const std::filesystem::path& path) {
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error) {
    throw std::runtime_error("could not determine file size: " +
                             path.string() + ": " + error.message());
  }
  return {.path = path.string(),
          .sha256 = sha256_file(path),
          .bytes = static_cast<std::uint64_t>(size)};
}

std::string_view architecture_name(ChatbotArchitecture architecture) noexcept {
  switch (architecture) {
    case ChatbotArchitecture::ann:
      return "ann";
    case ChatbotArchitecture::snn:
      return "snn";
  }
  return "invalid";
}

std::string_view run_status_name(RunStatus status) noexcept {
  switch (status) {
    case RunStatus::planned:
      return "planned";
    case RunStatus::completed:
      return "completed";
    case RunStatus::failed:
      return "failed";
  }
  return "invalid";
}

void validate_run_manifest(const RunManifest& manifest) {
  require(manifest.schema_version == chatbot_run_schema,
          "unsupported chatbot run schema");
  require(!manifest.run_id.empty(), "run id is empty");
  require(is_utc_timestamp(manifest.created_utc),
          "created_utc must use YYYY-MM-DDTHH:MM:SSZ");
  require(architecture_name(manifest.architecture) != "invalid",
          "chatbot architecture is invalid");
  require(run_status_name(manifest.status) != "invalid", "run status is invalid");
  require(!manifest.experiments_repository.path.empty(),
          "experiments repository path is empty");
  require(!manifest.library_repository.path.empty(),
          "library repository path is empty");
  require(is_git_revision(manifest.experiments_repository.revision),
          "experiments repository revision must be a full lowercase Git SHA");
  require(is_git_revision(manifest.library_repository.revision),
          "library repository revision must be a full lowercase Git SHA");
  validate_file(manifest.executable, "executable");
  validate_file(manifest.config, "config");
  require(!manifest.dataset.name.empty(), "dataset name is empty");
  require(!manifest.dataset.version.empty(), "dataset version is empty");
  require(!manifest.dataset.source_uri.empty(), "dataset source URI is empty");
  require(!manifest.dataset.license.empty(), "dataset license is empty");
  validate_file(manifest.dataset.source, "dataset");
  require(manifest.dataset.format == chatbot_dataset_format,
          "unsupported chatbot dataset format");
  require(manifest.dataset.records > 0U, "dataset has no records");
  require(manifest.dataset.split.algorithm == chatbot_split_algorithm,
          "unsupported chatbot split algorithm");
  const auto held_out =
      static_cast<std::uint32_t>(manifest.dataset.split.validation_basis_points) +
      static_cast<std::uint32_t>(manifest.dataset.split.test_basis_points);
  require(held_out < 10000U,
          "validation and test basis points must sum to less than 10000");
  require(manifest.dataset.split.train_records +
              manifest.dataset.split.validation_records +
              manifest.dataset.split.test_records ==
          manifest.dataset.records,
          "dataset split counts do not sum to record count");
  validate_resource(manifest.reference_model, "reference model");
  validate_resource(manifest.tokenizer, "tokenizer");
  require(contains_file(
              manifest.reference_model, "config.json",
              "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59"),
          "reference model does not contain the pinned config.json hash");
  require(contains_file(
              manifest.tokenizer, "tokenizer.json",
              "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
          "tokenizer does not contain the pinned tokenizer.json hash");
  require(contains_file(
              manifest.tokenizer, "tokenizer_config.json",
              "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"),
          "tokenizer does not contain the pinned tokenizer_config.json hash");
  require(manifest.training.objective == "causal_language_modeling",
          "unsupported training objective");
  require(manifest.training.loss_scope == "assistant_content_and_eom",
          "unsupported loss scope");
  require(manifest.training.assistant_only_loss,
          "assistant-only loss must be enabled");
  require(manifest.training.ignore_index == -100,
          "assistant-only labels must use ignore index -100");
  require(!manifest.command.empty(), "run command is empty");
  for (const auto& artifact : manifest.artifacts) {
    require(!artifact.kind.empty(), "artifact kind is empty");
    validate_file(artifact.file, "artifact");
  }
  for (const auto& metric : manifest.metrics) {
    require(!metric.id.empty(), "metric id is empty");
    require(std::isfinite(metric.value), "metric value is not finite");
    require(!metric.unit.empty(), "metric unit is empty");
  }

  if (manifest.publishable) {
    require(manifest.status == RunStatus::completed,
            "publishable run must be completed");
    require(!manifest.experiments_repository.dirty,
            "publishable run has a dirty experiments repository");
    require(!manifest.library_repository.dirty,
            "publishable run has a dirty library repository");
    require(!manifest.toolchain.compiler.empty() &&
                !manifest.toolchain.cmake.empty() &&
                !manifest.toolchain.torch.empty(),
            "publishable run is missing toolchain provenance");
    require(!manifest.hardware.cpu.empty() && !manifest.hardware.gpu.empty(),
            "publishable run is missing hardware provenance");
    require(!manifest.artifacts.empty(),
            "publishable run has no hashed output artifacts");
    require(!manifest.metrics.empty(), "publishable run has no metrics");
  }
}

void write_run_manifest_json(std::ostream& output,
                             const RunManifest& manifest) {
  validate_run_manifest(manifest);
  output << "{\n  \"schema_version\": ";
  write_quoted(output, manifest.schema_version);
  output << ",\n  \"run_id\": ";
  write_quoted(output, manifest.run_id);
  output << ",\n  \"created_utc\": ";
  write_quoted(output, manifest.created_utc);
  output << ",\n  \"status\": ";
  write_quoted(output, run_status_name(manifest.status));
  output << ",\n  \"architecture\": ";
  write_quoted(output, architecture_name(manifest.architecture));
  output << ",\n  \"seed\": " << manifest.seed
         << ",\n  \"publishable\": "
         << (manifest.publishable ? "true" : "false")
         << ",\n  \"repositories\": {\"experiments\":";
  write_repository(output, manifest.experiments_repository);
  output << ",\"snnbase\":";
  write_repository(output, manifest.library_repository);
  output << "},\n  \"executable\": ";
  write_file(output, manifest.executable);
  output << ",\n  \"config\": ";
  write_file(output, manifest.config);
  output << ",\n  \"dataset\": {\"name\":";
  write_quoted(output, manifest.dataset.name);
  output << ",\"version\":";
  write_quoted(output, manifest.dataset.version);
  output << ",\"source_uri\":";
  write_quoted(output, manifest.dataset.source_uri);
  output << ",\"license\":";
  write_quoted(output, manifest.dataset.license);
  output << ",\"source\":";
  write_file(output, manifest.dataset.source);
  output << ",\"format\":";
  write_quoted(output, manifest.dataset.format);
  output << ",\"records\":" << manifest.dataset.records
         << ",\"split\":{\"algorithm\":";
  write_quoted(output, manifest.dataset.split.algorithm);
  output << ",\"seed\":" << manifest.dataset.split.seed
         << ",\"validation_basis_points\":"
         << manifest.dataset.split.validation_basis_points
         << ",\"test_basis_points\":"
         << manifest.dataset.split.test_basis_points
         << ",\"train_records\":" << manifest.dataset.split.train_records
         << ",\"validation_records\":"
         << manifest.dataset.split.validation_records
         << ",\"test_records\":" << manifest.dataset.split.test_records
         << "}},\n  \"reference_model\": ";
  write_resource(output, manifest.reference_model);
  output << ",\n  \"tokenizer\": ";
  write_resource(output, manifest.tokenizer);
  output << ",\n  \"training\": {\"objective\":";
  write_quoted(output, manifest.training.objective);
  output << ",\"loss_scope\":";
  write_quoted(output, manifest.training.loss_scope);
  output << ",\"assistant_only_loss\":"
         << (manifest.training.assistant_only_loss ? "true" : "false")
         << ",\"ignore_index\":" << manifest.training.ignore_index
         << "},\n  \"command\": [";
  for (std::size_t index = 0; index < manifest.command.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    write_quoted(output, manifest.command[index]);
  }
  output << "],\n  \"toolchain\": {\"compiler\":";
  write_quoted(output, manifest.toolchain.compiler);
  output << ",\"cmake\":";
  write_quoted(output, manifest.toolchain.cmake);
  output << ",\"torch\":";
  write_quoted(output, manifest.toolchain.torch);
  output << ",\"cuda\":";
  write_quoted(output, manifest.toolchain.cuda);
  output << "},\n  \"hardware\": {\"cpu\":";
  write_quoted(output, manifest.hardware.cpu);
  output << ",\"gpu\":";
  write_quoted(output, manifest.hardware.gpu);
  output << "},\n  \"artifacts\": [";
  for (std::size_t index = 0; index < manifest.artifacts.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << "{\"kind\":";
    write_quoted(output, manifest.artifacts[index].kind);
    output << ",\"file\":";
    write_file(output, manifest.artifacts[index].file);
    output << '}';
  }
  output << "],\n  \"metrics\": [" << std::setprecision(17);
  for (std::size_t index = 0; index < manifest.metrics.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << "{\"id\":";
    write_quoted(output, manifest.metrics[index].id);
    output << ",\"value\":" << manifest.metrics[index].value
           << ",\"unit\":";
    write_quoted(output, manifest.metrics[index].unit);
    output << '}';
  }
  output << "],\n  \"notes\": [";
  for (std::size_t index = 0; index < manifest.notes.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    write_quoted(output, manifest.notes[index]);
  }
  output << "]\n}\n";
  if (!output) {
    throw std::runtime_error("failed to write chatbot run manifest");
  }
}

}  // namespace snnbase_experiments::chatbot
