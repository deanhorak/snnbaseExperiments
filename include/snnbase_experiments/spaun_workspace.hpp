#ifndef SNNBASE_EXPERIMENTS_SPAUN_WORKSPACE_HPP
#define SNNBASE_EXPERIMENTS_SPAUN_WORKSPACE_HPP

#include <snnbase/semantic_pointer.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace snnbase_experiments::spaun::workspace {

inline constexpr std::size_t semantic_dimensions = 32;
inline constexpr std::size_t decimal_digit_count = 10;
inline constexpr std::size_t spaun_task_count = 8;
inline constexpr std::size_t query_kind_count = 2;
inline constexpr std::size_t position_count = 9;
inline constexpr std::size_t counting_number_count = 19;

// This vocabulary is process-global, immutable after construction, and uses a
// fixed seed.  Its names are D0--D9, A0--A7, P, K, and POS1--POS9.
[[nodiscard]] const snnbase::semantic_pointer::Vocabulary& vocabulary();
[[nodiscard]] std::string_view digit_name(std::size_t digit);
[[nodiscard]] std::string_view task_name(std::size_t task);
[[nodiscard]] std::string_view query_name(char query);
[[nodiscard]] std::string_view position_name(std::size_t one_based_position);

enum class TokenKind : std::uint8_t {
  digit,
  task,
  query,
  group_boundary,
  sequence_restart,
};

// A token event owns its already-encoded semantic value.  The kind is routing
// metadata, analogous to a visual/task-control channel; it never contains a
// desired response or a task solution.  Boundary/restart events have no value.
struct EncodedTokenEvent {
  TokenKind kind{TokenKind::digit};
  std::optional<snnbase::semantic_pointer::Pointer> semantic{};
};

[[nodiscard]] EncodedTokenEvent digit_event(std::size_t digit);
[[nodiscard]] EncodedTokenEvent task_event(std::size_t task);
[[nodiscard]] EncodedTokenEvent query_event(char query);
[[nodiscard]] EncodedTokenEvent group_boundary_event();
[[nodiscard]] EncodedTokenEvent sequence_restart_event();

enum class Pathway : std::uint8_t {
  encoding_to_working_memory,
  working_memory_recurrence,
  working_memory_to_transformation,
  transformation_action_to_decoding,
  decoding_to_motor,
};

struct Config {
  std::size_t maximum_groups{9};
  std::size_t maximum_list_length{9};
  std::size_t neurons_per_sign{1};
  double semantic_input_gain{6.0};
  double projection_weight_gain{1.1};
  double memory_noise_standard_deviation{};
  double memory_recurrent_weight_gain{1.1};
  double primacy_recurrent_weight_gain{1.1};
  double recency_recurrent_weight_gain{1.1};
  double cleanup_similarity_threshold{0.75};
  std::size_t counting_delay_ticks{42};
  std::uint32_t seed{42};
};

// A compact Spaun-style semantic workspace.  Every list item has a distinct
// recurrent LIF population.  Digit outputs are obtained exclusively by
// cleaning up the current motor population activity after the selected memory
// population has driven the complete spiking output pathway.
class Workspace {
 public:
  explicit Workspace(Config config = {});
  ~Workspace();
  Workspace(Workspace&&) noexcept;
  Workspace& operator=(Workspace&&) noexcept;
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  void accept(const EncodedTokenEvent& event);
  void advance(std::size_t steps = 1);

  [[nodiscard]] std::optional<int> recall_digit(std::size_t group,
                                                 std::size_t item) const;
  [[nodiscard]] std::vector<int> recall_group(std::size_t group) const;
  [[nodiscard]] std::optional<int> task_state() const;
  [[nodiscard]] std::optional<char> query_state() const;

  // route_recalled_digit never decodes the selected working-memory slot on the
  // host.  Its semantic spikes traverse WM -> transformation -> action ->
  // decoding -> motor, and only the motor activity is cleaned up.
  [[nodiscard]] std::optional<int> route_recalled_digit(std::size_t group,
                                                         std::size_t item);

  // The planned pointer is first written into a recurrent plan buffer through
  // the encoding population, then follows the same complete output pathway.
  [[nodiscard]] std::optional<int> route_planned_digit(
      const snnbase::semantic_pointer::Pointer& planned_digit);

  // Counting is a delayed feed-forward successor chain. The start value enters
  // from a recurrent working-memory slot, each configured delay advances one
  // learned semantic successor, and the selected stage traverses the same
  // transformation/action/decoding pathway as every other response.
  [[nodiscard]] bool begin_counting_from_memory(std::size_t group,
                                                 std::size_t item);
  [[nodiscard]] std::optional<int> route_counting_result(
      std::size_t increments);
  [[nodiscard]] std::optional<int> counting_stage_state(
      std::size_t increments) const;
  [[nodiscard]] std::size_t counting_delay_ticks() const noexcept;

  void set_pathway_enabled(Pathway pathway, bool enabled);
  [[nodiscard]] bool pathway_enabled(Pathway pathway) const;

  // reset clears all neural and sequence state while preserving lesion state.
  void reset();

  [[nodiscard]] std::size_t current_group() const noexcept;
  [[nodiscard]] std::size_t current_item() const noexcept;
  [[nodiscard]] std::size_t stored_item_count(std::size_t group) const;
  [[nodiscard]] std::size_t maximum_groups() const noexcept;
  [[nodiscard]] std::size_t maximum_list_length() const noexcept;
  [[nodiscard]] std::size_t neuron_count() const noexcept;
  [[nodiscard]] std::size_t population_count() const noexcept;
  [[nodiscard]] std::size_t projection_count() const noexcept;
  [[nodiscard]] std::size_t synapse_count() const noexcept;
  [[nodiscard]] std::size_t total_spikes() const noexcept;
  [[nodiscard]] std::uint64_t time() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace snnbase_experiments::spaun::workspace

#endif  // SNNBASE_EXPERIMENTS_SPAUN_WORKSPACE_HPP
