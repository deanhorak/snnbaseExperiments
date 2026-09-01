#include <snnbase_experiments/chatbot/conversation.hpp>
#include <snnbase_experiments/chatbot/dataset.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using snnbase_experiments::chatbot::Conversation;
using snnbase_experiments::chatbot::DatasetError;
using snnbase_experiments::chatbot::Message;
using snnbase_experiments::chatbot::Role;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("could not create test fixture");
  }
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void expect_dataset_error(const std::filesystem::path& path,
                          const std::string& text,
                          const std::string& expected_fragment) {
  write_text(path, text);
  try {
    static_cast<void>(
        snnbase_experiments::chatbot::load_conversation_jsonl(path));
  } catch (const DatasetError& error) {
    require(error.line() > 0U || text.empty(),
            "dataset error did not retain its line number");
    require(std::string(error.what()).find(expected_fragment) !=
                std::string::npos,
            "dataset error omitted the expected diagnostic");
    return;
  }
  throw std::runtime_error("invalid JSONL fixture was accepted");
}

void test_valid_loader(const std::filesystem::path& path) {
  const std::string fixture =
      "{\"id\":\"hello-1\",\"messages\":["
      "{\"role\":\"system\",\"content\":\"Be concise.\"},"
      "{\"role\":\"user\",\"content\":\"Say \\\"hi\\\".\"},"
      "{\"role\":\"assistant\",\"content\":\"Hi!\"}]}\n"
      "{\"messages\":[{\"content\":\"2+2?\",\"role\":\"user\"},"
      "{\"content\":\"4 \\u2713\",\"role\":\"assistant\"}],"
      "\"id\":\"math-2\"}\n";
  write_text(path, fixture);
  const auto dataset =
      snnbase_experiments::chatbot::load_conversation_jsonl(path);
  require(dataset.size() == 2U, "loader returned the wrong record count");
  require(dataset.conversations[0].messages.size() == 3U,
          "loader returned the wrong message count");
  require(dataset.conversations[0].messages[0].role == Role::system,
          "loader lost the system role");
  require(dataset.conversations[1].messages[1].content == "4 \xe2\x9c\x93",
          "loader decoded a Unicode escape incorrectly");
  require(dataset.source_sha256.size() == 64U,
          "loader did not hash the source JSONL");
}

void test_strict_validation(const std::filesystem::path& path) {
  expect_dataset_error(
      path,
      "{\"id\":\"bad-role\",\"messages\":[{\"role\":\"developer\","
      "\"content\":\"x\"}]}\n",
      "unsupported message role");
  expect_dataset_error(
      path,
      "{\"id\":\"bad-order\",\"messages\":[{\"role\":\"assistant\","
      "\"content\":\"x\"}]}\n",
      "expected role user");
  expect_dataset_error(
      path,
      "{\"id\":\"unknown\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"x\"},{\"role\":\"assistant\",\"content\":\"y\"}],"
      "\"metadata\":{}}\n",
      "unknown conversation field");
  expect_dataset_error(
      path,
      "{\"id\":\"same\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"x\"},{\"role\":\"assistant\",\"content\":\"y\"}]}\n"
      "{\"id\":\"same\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"z\"},{\"role\":\"assistant\",\"content\":\"q\"}]}\n",
      "duplicate conversation id");
  expect_dataset_error(path, "\n", "blank JSONL line");
  expect_dataset_error(
      path,
      "{\"id\":\"nul\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"x\"},{\"role\":\"assistant\","
      "\"content\":\"bad\\u0000value\"}]}\n",
      "without NUL");
}

void test_assistant_loss_mask() {
  using snnbase_experiments::chatbot::MessageTokenRange;
  const std::vector<MessageTokenRange> ranges{
      {.role = Role::system, .begin = 2, .end = 4},
      {.role = Role::user, .begin = 6, .end = 9},
      {.role = Role::assistant, .begin = 11, .end = 14},
      {.role = Role::user, .begin = 16, .end = 18},
      {.role = Role::assistant, .begin = 20, .end = 22}};
  const auto mask =
      snnbase_experiments::chatbot::make_assistant_only_loss_mask(24, ranges);
  require(std::count(mask.begin(), mask.end(), std::uint8_t{1}) == 5,
          "assistant mask selected the wrong number of tokens");
  require(mask[11] == 1U && mask[13] == 1U && mask[20] == 1U &&
              mask[21] == 1U,
          "assistant mask omitted an assistant token");
  require(mask[6] == 0U && mask[10] == 0U && mask[22] == 0U,
          "assistant mask included prompt or control tokens");

  std::vector<std::int64_t> tokens(24);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    tokens[index] = static_cast<std::int64_t>(100U + index);
  }
  const auto labels =
      snnbase_experiments::chatbot::make_assistant_only_labels(tokens, mask);
  require(labels[10] == -100 && labels[11] == tokens[11] &&
              labels[21] == tokens[21] && labels[22] == -100,
          "assistant labels do not match the binary mask");

  bool threw = false;
  try {
    const std::vector<MessageTokenRange> overlap{
        {.role = Role::user, .begin = 1, .end = 4},
        {.role = Role::assistant, .begin = 3, .end = 5}};
    static_cast<void>(
        snnbase_experiments::chatbot::make_assistant_only_loss_mask(8, overlap));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "assistant mask accepted overlapping token ranges");
}

Conversation conversation(std::string id) {
  return {.id = std::move(id),
          .messages = {{.role = Role::user, .content = "prompt"},
                       {.role = Role::assistant, .content = "answer"}}};
}

void test_deterministic_split() {
  std::vector<Conversation> records;
  for (std::size_t index = 0; index < 250U; ++index) {
    records.push_back(conversation("conversation-" + std::to_string(index)));
  }
  const snnbase_experiments::chatbot::SplitPolicy policy{
      .seed = 42, .validation_basis_points = 1500, .test_basis_points = 1000};
  const auto vectors_path = std::filesystem::path(__FILE__).parent_path() /
                            "fixtures/chatbot/split_vectors_v1.tsv";
  std::ifstream vectors(vectors_path);
  require(static_cast<bool>(vectors),
          "could not open shared split-contract vectors");
  std::string line;
  require(static_cast<bool>(std::getline(vectors, line)) &&
              line == "seed\tconversation_id\tbucket",
          "split-contract vectors have an invalid header");
  std::size_t vector_count = 0U;
  while (std::getline(vectors, line)) {
    std::istringstream row(line);
    std::string seed_text;
    std::string identifier;
    std::string bucket_text;
    require(static_cast<bool>(std::getline(row, seed_text, '\t')) &&
                static_cast<bool>(std::getline(row, identifier, '\t')) &&
                static_cast<bool>(std::getline(row, bucket_text)) &&
                !seed_text.empty() && !identifier.empty() &&
                !bucket_text.empty(),
            "split-contract vector row is invalid");
    const auto seed = std::stoull(seed_text);
    const auto bucket = std::stoul(bucket_text);
    require(snnbase_experiments::chatbot::deterministic_split_bucket(
                identifier, seed) == bucket,
            "split bucket does not match a shared SHA-256 vector");
    ++vector_count;
  }
  require(vector_count >= 5U, "too few shared split-contract vectors");
  const auto first =
      snnbase_experiments::chatbot::deterministic_split(records, policy);
  require(first.train.size() + first.validation.size() + first.test.size() ==
              records.size(),
          "split lost dataset records");
  require(!first.train.empty() && !first.validation.empty() &&
              !first.test.empty(),
          "split failed to populate a configured partition");

  std::map<std::string, std::string> assignments;
  for (std::size_t index = 0; index < records.size(); ++index) {
    assignments.emplace(
        records[index].id,
        std::string(snnbase_experiments::chatbot::partition_name(
            snnbase_experiments::chatbot::deterministic_partition(
                records[index].id, policy))));
  }
  std::reverse(records.begin(), records.end());
  const auto second =
      snnbase_experiments::chatbot::deterministic_split(records, policy);
  require(second.train.size() == first.train.size() &&
              second.validation.size() == first.validation.size() &&
              second.test.size() == first.test.size(),
          "split membership changed when input order changed");
  for (const auto& record : records) {
    require(assignments.at(record.id) ==
                snnbase_experiments::chatbot::partition_name(
                    snnbase_experiments::chatbot::deterministic_partition(
                        record.id, policy)),
            "record partition changed when input order changed");
  }

  bool seed_changed_assignment = false;
  auto other_policy = policy;
  other_policy.seed = 43;
  for (const auto& record : records) {
    seed_changed_assignment =
        seed_changed_assignment ||
        snnbase_experiments::chatbot::deterministic_partition(record.id,
                                                               policy) !=
            snnbase_experiments::chatbot::deterministic_partition(record.id,
                                                                   other_policy);
  }
  require(seed_changed_assignment, "split seed had no effect on assignments");
}

}  // namespace

int main() {
  const auto directory = std::filesystem::temp_directory_path() /
                         "snnbase-chatbot-dataset-tests";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto fixture = directory / "conversations.jsonl";
  try {
    test_valid_loader(fixture);
    test_strict_validation(fixture);
    test_assistant_loss_mask();
    test_deterministic_split();
  } catch (...) {
    std::filesystem::remove_all(directory);
    throw;
  }
  std::filesystem::remove_all(directory);
}
