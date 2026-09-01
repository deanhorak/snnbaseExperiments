#ifndef SNNBASE_EXPERIMENTS_CHATBOT_CONVERSATION_HPP
#define SNNBASE_EXPERIMENTS_CHATBOT_CONVERSATION_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snnbase_experiments::chatbot {

// The dataset contract deliberately excludes tool and developer messages. New
// roles require a schema version change instead of being silently mapped to a
// training role.
enum class Role : std::uint8_t {
  system,
  user,
  assistant,
};

struct Message {
  Role role{Role::user};
  std::string content{};

  [[nodiscard]] bool operator==(const Message&) const = default;
};

struct Conversation {
  std::string id{};
  std::vector<Message> messages{};

  [[nodiscard]] bool operator==(const Conversation&) const = default;
};

[[nodiscard]] std::string_view role_name(Role role) noexcept;
[[nodiscard]] std::optional<Role> parse_role(std::string_view role) noexcept;
[[nodiscard]] bool is_valid_utf8(std::string_view text) noexcept;

// Throws std::invalid_argument on a contract violation. A valid training
// conversation has an optional first system message followed by one or more
// user/assistant turns, starts with user, and ends with assistant.
void validate_conversation(const Conversation& conversation);

// Ranges refer to the final, already-tokenized sequence. The tokenizer adapter
// must exclude role/header control tokens from these ranges. An assistant range
// should include the assistant content and its end-of-message token so the
// model learns when to stop.
struct MessageTokenRange {
  Role role{Role::user};
  std::size_t begin{};
  std::size_t end{};  // Half-open [begin, end).
};

// Returns one byte per token: 1 only for tokens in assistant ranges, 0 for all
// prompt/control/system/user tokens. Ranges must be ordered, nonempty,
// non-overlapping, and within token_count.
[[nodiscard]] std::vector<std::uint8_t> make_assistant_only_loss_mask(
    std::size_t token_count, std::span<const MessageTokenRange> ranges);

// Converts a binary mask to the labels convention used by causal-language-
// model losses. The model is expected to perform the usual one-token shift;
// this function does not shift token IDs itself.
[[nodiscard]] std::vector<std::int64_t> make_assistant_only_labels(
    std::span<const std::int64_t> token_ids,
    std::span<const std::uint8_t> loss_mask,
    std::int64_t ignore_index = -100);

}  // namespace snnbase_experiments::chatbot

#endif  // SNNBASE_EXPERIMENTS_CHATBOT_CONVERSATION_HPP
