#include <snnbase_experiments/chatbot/conversation.hpp>

#include <algorithm>
#include <stdexcept>

namespace snnbase_experiments::chatbot {
namespace {

bool is_continuation(unsigned char value) noexcept {
  return (value & 0xc0U) == 0x80U;
}

std::string message_prefix(std::size_t index) {
  return "conversation message " + std::to_string(index) + ": ";
}

}  // namespace

std::string_view role_name(Role role) noexcept {
  switch (role) {
    case Role::system:
      return "system";
    case Role::user:
      return "user";
    case Role::assistant:
      return "assistant";
  }
  return "invalid";
}

std::optional<Role> parse_role(std::string_view role) noexcept {
  if (role == "system") {
    return Role::system;
  }
  if (role == "user") {
    return Role::user;
  }
  if (role == "assistant") {
    return Role::assistant;
  }
  return std::nullopt;
}

bool is_valid_utf8(std::string_view text) noexcept {
  std::size_t index = 0;
  while (index < text.size()) {
    const auto first = static_cast<unsigned char>(text[index]);
    if (first <= 0x7fU) {
      ++index;
      continue;
    }

    std::size_t count = 0;
    std::uint32_t codepoint = 0;
    std::uint32_t minimum = 0;
    if ((first & 0xe0U) == 0xc0U) {
      count = 2;
      codepoint = first & 0x1fU;
      minimum = 0x80U;
    } else if ((first & 0xf0U) == 0xe0U) {
      count = 3;
      codepoint = first & 0x0fU;
      minimum = 0x800U;
    } else if ((first & 0xf8U) == 0xf0U) {
      count = 4;
      codepoint = first & 0x07U;
      minimum = 0x10000U;
    } else {
      return false;
    }
    if (index + count > text.size()) {
      return false;
    }
    for (std::size_t offset = 1; offset < count; ++offset) {
      const auto next = static_cast<unsigned char>(text[index + offset]);
      if (!is_continuation(next)) {
        return false;
      }
      codepoint = (codepoint << 6U) | (next & 0x3fU);
    }
    if (codepoint < minimum || codepoint > 0x10ffffU ||
        (codepoint >= 0xd800U && codepoint <= 0xdfffU)) {
      return false;
    }
    index += count;
  }
  return true;
}

void validate_conversation(const Conversation& conversation) {
  if (conversation.id.empty()) {
    throw std::invalid_argument("conversation id must not be empty");
  }
  if (!is_valid_utf8(conversation.id) ||
      conversation.id.find('\0') != std::string::npos) {
    throw std::invalid_argument("conversation id must be valid UTF-8 without NUL");
  }
  if (conversation.messages.empty()) {
    throw std::invalid_argument("conversation must contain messages");
  }

  std::size_t first_turn = 0;
  if (conversation.messages.front().role == Role::system) {
    first_turn = 1;
  }
  if (first_turn == conversation.messages.size()) {
    throw std::invalid_argument(
        "conversation must contain a user/assistant turn after system");
  }

  for (std::size_t index = 0; index < conversation.messages.size(); ++index) {
    const auto& message = conversation.messages[index];
    if (message.content.empty()) {
      throw std::invalid_argument(message_prefix(index) +
                                  "content must not be empty");
    }
    if (!is_valid_utf8(message.content) ||
        message.content.find('\0') != std::string::npos) {
      throw std::invalid_argument(message_prefix(index) +
                                  "content must be valid UTF-8 without NUL");
    }
    if (message.role != Role::system && message.role != Role::user &&
        message.role != Role::assistant) {
      throw std::invalid_argument(message_prefix(index) + "role is invalid");
    }
    if (message.role == Role::system && index != 0) {
      throw std::invalid_argument(
          message_prefix(index) + "system role is permitted only at index 0");
    }
  }

  for (std::size_t index = first_turn; index < conversation.messages.size();
       ++index) {
    const auto expected = ((index - first_turn) % 2U == 0U)
                              ? Role::user
                              : Role::assistant;
    if (conversation.messages[index].role != expected) {
      throw std::invalid_argument(message_prefix(index) + "expected role " +
                                  std::string(role_name(expected)));
    }
  }
  if (conversation.messages.back().role != Role::assistant) {
    throw std::invalid_argument(
        "conversation must end with an assistant response");
  }
}

std::vector<std::uint8_t> make_assistant_only_loss_mask(
    std::size_t token_count, std::span<const MessageTokenRange> ranges) {
  std::vector<std::uint8_t> mask(token_count, 0U);
  std::size_t previous_end = 0;
  bool first = true;
  for (const auto& range : ranges) {
    if (range.begin >= range.end) {
      throw std::invalid_argument("message token ranges must be nonempty");
    }
    if (range.end > token_count) {
      throw std::invalid_argument("message token range exceeds token count");
    }
    if (!first && range.begin < previous_end) {
      throw std::invalid_argument(
          "message token ranges must be ordered and non-overlapping");
    }
    if (range.role != Role::system && range.role != Role::user &&
        range.role != Role::assistant) {
      throw std::invalid_argument("message token range has an invalid role");
    }
    if (range.role == Role::assistant) {
      std::fill(mask.begin() + static_cast<std::ptrdiff_t>(range.begin),
                mask.begin() + static_cast<std::ptrdiff_t>(range.end), 1U);
    }
    previous_end = range.end;
    first = false;
  }
  return mask;
}

std::vector<std::int64_t> make_assistant_only_labels(
    std::span<const std::int64_t> token_ids,
    std::span<const std::uint8_t> loss_mask, std::int64_t ignore_index) {
  if (token_ids.size() != loss_mask.size()) {
    throw std::invalid_argument("token IDs and loss mask sizes differ");
  }
  std::vector<std::int64_t> labels(token_ids.size(), ignore_index);
  for (std::size_t index = 0; index < token_ids.size(); ++index) {
    if (loss_mask[index] > 1U) {
      throw std::invalid_argument("assistant loss mask must be binary");
    }
    if (loss_mask[index] == 1U) {
      labels[index] = token_ids[index];
    }
  }
  return labels;
}

}  // namespace snnbase_experiments::chatbot
