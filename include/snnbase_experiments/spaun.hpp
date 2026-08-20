#ifndef SNNBASE_EXPERIMENTS_SPAUN_HPP
#define SNNBASE_EXPERIMENTS_SPAUN_HPP

#include <snnbase/snnbase.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snnbase_experiments::spaun {

enum class Task : std::uint8_t {
  copy_drawing = 0,
  image_recognition = 1,
  reinforcement_learning = 2,
  serial_working_memory = 3,
  counting = 4,
  question_answering = 5,
  rapid_variable_creation = 6,
  fluid_reasoning = 7,
};

enum class Module : std::uint8_t {
  visual = 0,
  encoding,
  working_memory,
  transformation,
  reward,
  action_selection,
  decoding,
  motor,
  arm,
};

// Explicit lesion points used by the behavioral benchmark. Disabling a
// pathway changes the causal graph; it is not a telemetry-only switch.
enum class ModelPathway : std::uint8_t {
  retina_to_visual = 0,
  visual_to_encoding,
  encoding_to_working_memory,
  working_memory_recurrence,
  working_memory_to_transformation,
  reward_plasticity,
  transformation_action_to_decoding,
  decoding_to_motor,
  motor_to_arm,
};

inline constexpr std::size_t task_count = 8;
inline constexpr std::size_t module_count = 9;
inline constexpr std::size_t glyph_rows = 7;
inline constexpr std::size_t glyph_columns = 5;

struct Config {
  std::size_t neurons_per_module{64};
  std::size_t stimulus_ticks{15};
  std::size_t blank_ticks{15};
  std::size_t motor_ticks_per_target{5};
  double tick_seconds{0.01};
  double memory_noise_standard_deviation{0.0105};
  // Cross-seed protocol sweep: this gain/noise pair preserves Spaun's
  // primacy/recency signature instead of optimizing away the serial-position
  // effect with ceiling-level recall.
  double memory_recurrent_weight_gain{1.020};
  double primacy_recurrent_weight_gain{1.05};
  double recency_recurrent_weight_gain{1.05};
  double memory_cleanup_similarity_threshold{0.75};
  std::size_t counting_delay_ticks{42};
  std::uint32_t seed{42};
};

struct Trial {
  Task task{Task::image_recognition};
  std::string name;
  std::string stimulus_stream;
  std::vector<std::vector<int>> groups;
  char query_kind{};
  int query_value{};
  std::array<double, 3> reward_probabilities{0.12, 0.12, 0.72};
  std::vector<int> expected;
  // Optional externally supplied A2 contingency blocks. Empty selects the
  // published three-block default; nonempty runs must provide exactly three
  // valid three-arm probability vectors.
  std::vector<std::array<double, 3>> reward_probability_blocks;
};

struct ArmState {
  double shoulder{};
  double elbow{};
  double shoulder_velocity{};
  double elbow_velocity{};
  double elbow_x{};
  double elbow_y{};
  double pen_x{};
  double pen_y{};
  bool pen_down{};
};

struct PenPoint {
  double x{};
  double y{};
  bool stroke_start{};
};

struct ProbeFrame {
  std::size_t tick{};
  double time_seconds{};
  char stimulus{' '};
  char recognized{' '};
  std::string phase;
  std::string working_memory;
  std::string output;
  std::string selected_action;
  std::array<double, module_count> activity{};
  std::array<std::size_t, module_count> spikes{};
  std::array<double, 3> action_utilities{};
  double reward{};
  ArmState arm;
  std::vector<PenPoint> pen_trace;
};

struct Result {
  Task task{Task::image_recognition};
  std::string task_name;
  std::string stimulus_stream;
  std::vector<int> expected;
  // Decoded cognitive answer, retained even when the motor plant is lesioned.
  std::vector<int> output;
  // Behavioral correctness additionally requires observable arm ink whenever
  // the trial has a nonempty expected answer.
  bool correct{};
  std::size_t network_neurons{};
  std::size_t network_synapses{};
  std::size_t total_spikes{};
  double simulated_seconds{};
  std::vector<ProbeFrame> frames;
};

[[nodiscard]] std::string_view task_name(Task task) noexcept;
[[nodiscard]] std::string_view module_name(Module module) noexcept;
[[nodiscard]] Task parse_task(std::string_view value);
[[nodiscard]] std::array<Task, task_count> all_tasks() noexcept;
[[nodiscard]] Trial canonical_trial(Task task);
[[nodiscard]] std::array<std::string_view, glyph_rows> glyph(char symbol);
[[nodiscard]] snnbase::SpikeEvent glyph_event(char symbol);

class Model {
 public:
  explicit Model(Config config = {});
  ~Model();
  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] Result run(const Trial& trial);
  void reset();
  void set_pathway_enabled(ModelPathway pathway, bool enabled);
  [[nodiscard]] bool pathway_enabled(ModelPathway pathway) const;

  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] std::size_t neuron_count() const noexcept;
  [[nodiscard]] std::size_t synapse_count() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string digits_string(std::span<const int> digits);

}  // namespace snnbase_experiments::spaun

#endif  // SNNBASE_EXPERIMENTS_SPAUN_HPP
