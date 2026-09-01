#include <snnbase_experiments/chatbot/dataset.hpp>

#include <snnbase_experiments/chatbot/manifest.hpp>

#include <array>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_set>

namespace snnbase_experiments::chatbot {
namespace {

class JsonLineParser {
 public:
  explicit JsonLineParser(std::string_view line) : line_(line) {}

  Conversation parse() {
    Conversation conversation;
    bool saw_id = false;
    bool saw_messages = false;
    skip_whitespace();
    expect('{');
    skip_whitespace();
    if (consume('}')) {
      fail("conversation object is empty");
    }
    while (true) {
      const auto key = parse_string();
      skip_whitespace();
      expect(':');
      skip_whitespace();
      if (key == "id") {
        if (saw_id) {
          fail("duplicate id field");
        }
        conversation.id = parse_string();
        saw_id = true;
      } else if (key == "messages") {
        if (saw_messages) {
          fail("duplicate messages field");
        }
        conversation.messages = parse_messages();
        saw_messages = true;
      } else {
        fail("unknown conversation field: " + key);
      }
      skip_whitespace();
      if (consume('}')) {
        break;
      }
      expect(',');
      skip_whitespace();
    }
    skip_whitespace();
    if (position_ != line_.size()) {
      fail("trailing characters after conversation object");
    }
    if (!saw_id || !saw_messages) {
      fail("conversation requires id and messages fields");
    }
    return conversation;
  }

  [[nodiscard]] std::size_t column() const noexcept { return position_ + 1U; }

 private:
  std::vector<Message> parse_messages() {
    std::vector<Message> messages;
    expect('[');
    skip_whitespace();
    if (consume(']')) {
      return messages;
    }
    while (true) {
      messages.push_back(parse_message());
      skip_whitespace();
      if (consume(']')) {
        break;
      }
      expect(',');
      skip_whitespace();
    }
    return messages;
  }

  Message parse_message() {
    Message message;
    bool saw_role = false;
    bool saw_content = false;
    expect('{');
    skip_whitespace();
    if (consume('}')) {
      fail("message object is empty");
    }
    while (true) {
      const auto key = parse_string();
      skip_whitespace();
      expect(':');
      skip_whitespace();
      if (key == "role") {
        if (saw_role) {
          fail("duplicate message role field");
        }
        const auto value = parse_string();
        const auto role = parse_role(value);
        if (!role.has_value()) {
          fail("unsupported message role: " + value);
        }
        message.role = *role;
        saw_role = true;
      } else if (key == "content") {
        if (saw_content) {
          fail("duplicate message content field");
        }
        message.content = parse_string();
        saw_content = true;
      } else {
        fail("unknown message field: " + key);
      }
      skip_whitespace();
      if (consume('}')) {
        break;
      }
      expect(',');
      skip_whitespace();
    }
    if (!saw_role || !saw_content) {
      fail("message requires role and content fields");
    }
    return message;
  }

  static std::uint32_t hex_digit(char character) {
    if (character >= '0' && character <= '9') {
      return static_cast<std::uint32_t>(character - '0');
    }
    if (character >= 'a' && character <= 'f') {
      return static_cast<std::uint32_t>(character - 'a' + 10);
    }
    if (character >= 'A' && character <= 'F') {
      return static_cast<std::uint32_t>(character - 'A' + 10);
    }
    throw std::invalid_argument("invalid hexadecimal digit");
  }

  std::uint32_t parse_hex_quad() {
    if (position_ + 4U > line_.size()) {
      fail("truncated Unicode escape");
    }
    std::uint32_t value = 0;
    for (std::size_t count = 0; count < 4; ++count) {
      try {
        value = (value << 4U) | hex_digit(line_[position_++]);
      } catch (const std::invalid_argument&) {
        fail("invalid Unicode escape");
      }
    }
    return value;
  }

  static void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7fU) {
      output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
      output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
      output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
      output.push_back(
          static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
      output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
      output.push_back(
          static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
      output.push_back(
          static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
  }

  void parse_unicode_escape(std::string& output) {
    auto codepoint = parse_hex_quad();
    if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
      if (position_ + 2U > line_.size() || line_[position_] != '\\' ||
          line_[position_ + 1U] != 'u') {
        fail("high surrogate is not followed by a low surrogate");
      }
      position_ += 2U;
      const auto low = parse_hex_quad();
      if (low < 0xdc00U || low > 0xdfffU) {
        fail("invalid low surrogate");
      }
      codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) +
                  (low - 0xdc00U);
    } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
      fail("unpaired low surrogate");
    }
    append_utf8(output, codepoint);
  }

  std::string parse_string() {
    expect('"');
    std::string value;
    while (position_ < line_.size()) {
      const auto character = line_[position_++];
      if (character == '"') {
        if (!is_valid_utf8(value)) {
          fail("JSON string is not valid UTF-8");
        }
        return value;
      }
      if (static_cast<unsigned char>(character) < 0x20U) {
        fail("unescaped control character in JSON string");
      }
      if (character != '\\') {
        value.push_back(character);
        continue;
      }
      if (position_ == line_.size()) {
        fail("truncated JSON escape");
      }
      const auto escaped = line_[position_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          value.push_back(escaped);
          break;
        case 'b':
          value.push_back('\b');
          break;
        case 'f':
          value.push_back('\f');
          break;
        case 'n':
          value.push_back('\n');
          break;
        case 'r':
          value.push_back('\r');
          break;
        case 't':
          value.push_back('\t');
          break;
        case 'u':
          parse_unicode_escape(value);
          break;
        default:
          fail("unsupported JSON escape");
      }
    }
    fail("unterminated JSON string");
  }

  void skip_whitespace() noexcept {
    while (position_ < line_.size() &&
           (line_[position_] == ' ' || line_[position_] == '\t' ||
            line_[position_] == '\r' || line_[position_] == '\n')) {
      ++position_;
    }
  }

  bool consume(char expected) noexcept {
    if (position_ < line_.size() && line_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void expect(char expected) {
    if (!consume(expected)) {
      fail(std::string("expected '") + expected + "'");
    }
  }

  [[noreturn]] void fail(std::string message) const {
    throw std::invalid_argument(std::move(message));
  }

  std::string_view line_;
  std::size_t position_{};
};

void validate_limits(const DatasetLimits& limits) {
  if (limits.maximum_records == 0U || limits.maximum_line_bytes == 0U ||
      limits.maximum_id_bytes == 0U ||
      limits.maximum_messages_per_conversation == 0U ||
      limits.maximum_content_bytes_per_message == 0U) {
    throw std::invalid_argument("all chatbot dataset limits must be positive");
  }
}

void validate_policy(const SplitPolicy& policy) {
  const auto held_out = static_cast<std::uint32_t>(policy.test_basis_points) +
                        policy.validation_basis_points;
  if (held_out >= 10000U) {
    throw std::invalid_argument(
        "validation and test basis points must sum to less than 10000");
  }
}

std::string error_message(const std::filesystem::path& path, std::size_t line,
                          std::size_t column, std::string_view message) {
  std::ostringstream output;
  output << path.string();
  if (line != 0U) {
    output << ':' << line;
    if (column != 0U) {
      output << ':' << column;
    }
  }
  output << ": " << message;
  return output.str();
}

}  // namespace

DatasetError::DatasetError(std::filesystem::path path, std::size_t line,
                           std::size_t column, std::string message)
    : std::runtime_error(error_message(path, line, column, message)),
      path_(std::move(path)),
      line_(line),
      column_(column) {}

const std::filesystem::path& DatasetError::path() const noexcept {
  return path_;
}

std::size_t DatasetError::line() const noexcept { return line_; }

std::size_t DatasetError::column() const noexcept { return column_; }

std::size_t ConversationDataset::size() const noexcept {
  return conversations.size();
}

ConversationDataset load_conversation_jsonl(const std::filesystem::path& path,
                                            const DatasetLimits& limits) {
  validate_limits(limits);
  std::string hash_before;
  try {
    hash_before = sha256_file(path);
  } catch (const std::runtime_error& error) {
    throw DatasetError(path, 0, 0, error.what());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw DatasetError(path, 0, 0, "could not open JSONL dataset");
  }

  ConversationDataset dataset{.source_path = path};
  std::unordered_set<std::string> ids;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line == "\r") {
      throw DatasetError(path, line_number, 1, "blank JSONL line is not allowed");
    }
    if (line.size() > limits.maximum_line_bytes) {
      throw DatasetError(path, line_number, 1,
                         "JSONL line exceeds maximum_line_bytes");
    }
    if (dataset.conversations.size() >= limits.maximum_records) {
      throw DatasetError(path, line_number, 1,
                         "dataset exceeds maximum_records");
    }

    JsonLineParser parser(line);
    Conversation conversation;
    try {
      conversation = parser.parse();
      validate_conversation(conversation);
    } catch (const std::invalid_argument& error) {
      throw DatasetError(path, line_number, parser.column(), error.what());
    }
    if (conversation.id.size() > limits.maximum_id_bytes) {
      throw DatasetError(path, line_number, 1,
                         "conversation id exceeds maximum_id_bytes");
    }
    if (conversation.messages.size() >
        limits.maximum_messages_per_conversation) {
      throw DatasetError(path, line_number, 1,
                         "conversation exceeds maximum message count");
    }
    for (const auto& message : conversation.messages) {
      if (message.content.size() > limits.maximum_content_bytes_per_message) {
        throw DatasetError(path, line_number, 1,
                           "message content exceeds configured byte limit");
      }
    }
    if (!ids.insert(conversation.id).second) {
      throw DatasetError(path, line_number, 1,
                         "duplicate conversation id: " + conversation.id);
    }
    dataset.conversations.push_back(std::move(conversation));
  }
  if (!input.eof()) {
    throw DatasetError(path, line_number, 0, "failed while reading JSONL dataset");
  }
  if (dataset.conversations.empty()) {
    throw DatasetError(path, 0, 0, "JSONL dataset contains no records");
  }
  try {
    dataset.source_sha256 = sha256_file(path);
  } catch (const std::runtime_error& error) {
    throw DatasetError(path, 0, 0, error.what());
  }
  if (dataset.source_sha256 != hash_before) {
    throw DatasetError(path, 0, 0,
                       "JSONL dataset changed while it was being loaded");
  }
  return dataset;
}

std::string_view partition_name(DatasetPartition partition) noexcept {
  switch (partition) {
    case DatasetPartition::train:
      return "train";
    case DatasetPartition::validation:
      return "validation";
    case DatasetPartition::test:
      return "test";
  }
  return "invalid";
}

DatasetPartition deterministic_partition(std::string_view conversation_id,
                                         const SplitPolicy& policy) {
  validate_policy(policy);
  const auto bucket = deterministic_split_bucket(conversation_id, policy.seed);
  if (bucket < policy.test_basis_points) {
    return DatasetPartition::test;
  }
  if (bucket < static_cast<std::uint32_t>(policy.test_basis_points) +
                   policy.validation_basis_points) {
    return DatasetPartition::validation;
  }
  return DatasetPartition::train;
}

std::uint16_t deterministic_split_bucket(std::string_view conversation_id,
                                         std::uint64_t seed) {
  if (conversation_id.empty() || !is_valid_utf8(conversation_id)) {
    throw std::invalid_argument(
        "split assignment requires a nonempty UTF-8 conversation id");
  }
  std::string input(8U, '\0');
  for (std::size_t index = 0; index < 8U; ++index) {
    input[index] = static_cast<char>(seed >> ((7U - index) * 8U));
  }
  input.append(conversation_id);
  const auto digest = sha256_digest(input);
  std::uint64_t prefix = 0;
  for (std::size_t index = 0; index < 8U; ++index) {
    prefix = (prefix << 8U) | digest[index];
  }
  return static_cast<std::uint16_t>(prefix % 10000U);
}

DatasetSplit deterministic_split(const std::vector<Conversation>& conversations,
                                 const SplitPolicy& policy) {
  validate_policy(policy);
  DatasetSplit split{.policy = policy};
  split.train.reserve(conversations.size());
  split.validation.reserve(conversations.size() / 10U);
  split.test.reserve(conversations.size() / 10U);
  std::unordered_set<std::string_view> ids;
  ids.reserve(conversations.size());
  for (std::size_t index = 0; index < conversations.size(); ++index) {
    validate_conversation(conversations[index]);
    if (!ids.insert(conversations[index].id).second) {
      throw std::invalid_argument(
          "deterministic split received duplicate conversation id: " +
          conversations[index].id);
    }
    switch (deterministic_partition(conversations[index].id, policy)) {
      case DatasetPartition::train:
        split.train.push_back(index);
        break;
      case DatasetPartition::validation:
        split.validation.push_back(index);
        break;
      case DatasetPartition::test:
        split.test.push_back(index);
        break;
    }
  }
  return split;
}

}  // namespace snnbase_experiments::chatbot
