#ifndef SNNBASE_EXPERIMENTS_CHATBOT_DATASET_HPP
#define SNNBASE_EXPERIMENTS_CHATBOT_DATASET_HPP

#include <snnbase_experiments/chatbot/conversation.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace snnbase_experiments::chatbot {

struct DatasetLimits {
  std::size_t maximum_records{1'000'000};
  std::size_t maximum_line_bytes{4U * 1024U * 1024U};
  std::size_t maximum_id_bytes{256};
  std::size_t maximum_messages_per_conversation{128};
  std::size_t maximum_content_bytes_per_message{1U * 1024U * 1024U};
};

class DatasetError : public std::runtime_error {
 public:
  DatasetError(std::filesystem::path path, std::size_t line,
               std::size_t column, std::string message);

  [[nodiscard]] const std::filesystem::path& path() const noexcept;
  [[nodiscard]] std::size_t line() const noexcept;
  [[nodiscard]] std::size_t column() const noexcept;

 private:
  std::filesystem::path path_;
  std::size_t line_{};
  std::size_t column_{};
};

struct ConversationDataset {
  std::filesystem::path source_path{};
  std::string source_sha256{};
  std::vector<Conversation> conversations{};

  [[nodiscard]] std::size_t size() const noexcept;
};

// Strictly loads one object per nonempty line:
// {"id":"...","messages":[{"role":"user","content":"..."}, ...]}
// Unknown or duplicate fields, blank lines, duplicate IDs, malformed UTF-8,
// and invalid role order are rejected rather than ignored.
[[nodiscard]] ConversationDataset load_conversation_jsonl(
    const std::filesystem::path& path,
    const DatasetLimits& limits = {});

enum class DatasetPartition : std::uint8_t {
  train,
  validation,
  test,
};

struct SplitPolicy {
  std::uint64_t seed{42};
  std::uint16_t validation_basis_points{1000};
  std::uint16_t test_basis_points{1000};
};

struct DatasetSplit {
  SplitPolicy policy{};
  std::vector<std::size_t> train{};
  std::vector<std::size_t> validation{};
  std::vector<std::size_t> test{};
};

[[nodiscard]] std::string_view partition_name(
    DatasetPartition partition) noexcept;

// Assignment is SHA-256(seed encoded as eight big-endian bytes || UTF-8 id),
// first eight digest bytes interpreted big-endian, modulo 10,000. Test owns
// the first test_basis_points buckets, validation the next range, and train
// the remainder. Membership therefore does not depend on input order.
[[nodiscard]] DatasetPartition deterministic_partition(
    std::string_view conversation_id, const SplitPolicy& policy = {});
[[nodiscard]] std::uint16_t deterministic_split_bucket(
    std::string_view conversation_id, std::uint64_t seed = 42);
[[nodiscard]] DatasetSplit deterministic_split(
    const std::vector<Conversation>& conversations,
    const SplitPolicy& policy = {});

}  // namespace snnbase_experiments::chatbot

#endif  // SNNBASE_EXPERIMENTS_CHATBOT_DATASET_HPP
