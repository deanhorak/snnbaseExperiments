#ifndef SNNBASE_EXPERIMENTS_CHATBOT_MANIFEST_HPP
#define SNNBASE_EXPERIMENTS_CHATBOT_MANIFEST_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snnbase_experiments::chatbot {

inline constexpr std::string_view chatbot_run_schema =
    "snnbase-chatbot-run-v1";
inline constexpr std::string_view chatbot_dataset_format =
    "snnbase-chatbot-jsonl-v1";
inline constexpr std::string_view chatbot_split_algorithm =
    "sha256-seed-bucket-v1";
inline constexpr std::string_view qwen3_reference_repository =
    "Qwen/Qwen3-0.6B-Base";
inline constexpr std::string_view qwen3_reference_revision =
    "da87bfb608c14b7cf20ba1ce41287e8de496c0cd";

using Sha256Digest = std::array<std::uint8_t, 32>;

[[nodiscard]] Sha256Digest sha256_digest(std::string_view bytes);
[[nodiscard]] std::string sha256_hex(std::string_view bytes);
[[nodiscard]] std::string sha256_file(const std::filesystem::path& path);
[[nodiscard]] bool is_sha256(std::string_view value) noexcept;
[[nodiscard]] bool is_git_revision(std::string_view value) noexcept;

struct FileDigest {
  std::string path{};
  std::string sha256{};
  std::uint64_t bytes{};
};

[[nodiscard]] FileDigest digest_file(const std::filesystem::path& path);

struct RepositoryState {
  std::string path{};
  std::string revision{};
  bool dirty{};
};

struct PinnedResource {
  std::string repository{};
  std::string revision{};
  std::string license{};
  std::vector<FileDigest> files{};
};

struct SplitManifest {
  std::string algorithm{chatbot_split_algorithm};
  std::uint64_t seed{};
  std::uint16_t validation_basis_points{1000};
  std::uint16_t test_basis_points{1000};
  std::size_t train_records{};
  std::size_t validation_records{};
  std::size_t test_records{};
};

struct DatasetManifest {
  std::string name{};
  std::string version{};
  std::string source_uri{};
  std::string license{};
  FileDigest source{};
  std::string format{chatbot_dataset_format};
  std::size_t records{};
  SplitManifest split{};
};

enum class ChatbotArchitecture : std::uint8_t {
  ann,
  snn,
};

enum class RunStatus : std::uint8_t {
  planned,
  completed,
  failed,
};

struct TrainingContract {
  std::string objective{"causal_language_modeling"};
  std::string loss_scope{"assistant_content_and_eom"};
  bool assistant_only_loss{true};
  std::int64_t ignore_index{-100};
};

struct ToolchainManifest {
  std::string compiler{};
  std::string cmake{};
  std::string torch{};
  std::string cuda{};
};

struct HardwareManifest {
  std::string cpu{};
  std::string gpu{};
};

struct Metric {
  std::string id{};
  double value{};
  std::string unit{};
};

struct ArtifactDigest {
  std::string kind{};
  FileDigest file{};
};

struct RunManifest {
  std::string schema_version{chatbot_run_schema};
  std::string run_id{};
  std::string created_utc{};
  RunStatus status{RunStatus::planned};
  ChatbotArchitecture architecture{ChatbotArchitecture::ann};
  std::uint64_t seed{};
  bool publishable{};
  RepositoryState experiments_repository{};
  RepositoryState library_repository{};
  FileDigest executable{};
  FileDigest config{};
  DatasetManifest dataset{};
  PinnedResource reference_model{};
  PinnedResource tokenizer{};
  TrainingContract training{};
  std::vector<std::string> command{};
  ToolchainManifest toolchain{};
  HardwareManifest hardware{};
  std::vector<ArtifactDigest> artifacts{};
  std::vector<Metric> metrics{};
  std::vector<std::string> notes{};
};

[[nodiscard]] std::string_view architecture_name(
    ChatbotArchitecture architecture) noexcept;
[[nodiscard]] std::string_view run_status_name(RunStatus status) noexcept;

// Throws std::invalid_argument for a schema/contract violation. A publishable
// manifest additionally requires a completed run, clean source repositories,
// complete environment fields, metrics, and hashed output artifacts.
void validate_run_manifest(const RunManifest& manifest);
void write_run_manifest_json(std::ostream& output,
                             const RunManifest& manifest);

}  // namespace snnbase_experiments::chatbot

#endif  // SNNBASE_EXPERIMENTS_CHATBOT_MANIFEST_HPP
