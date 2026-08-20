#ifndef SNNBASE_EXPERIMENTS_SPAUN_VISUAL_HPP
#define SNNBASE_EXPERIMENTS_SPAUN_VISUAL_HPP

#include <snnbase/population.hpp>
#include <snnbase/spikeevent.hpp>
#include <snnbase_experiments/spaun.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace snnbase_experiments::spaun {

inline constexpr std::size_t visual_symbol_count = 16;
inline constexpr std::size_t retina_pixel_count = glyph_rows * glyph_columns;
inline constexpr std::size_t opponent_retina_neuron_count =
    retina_pixel_count * 2;

using Retina = std::array<std::uint8_t, retina_pixel_count>;

enum class VisualModule : std::uint8_t {
  retina = 0,
  visual,
  encoding,
};

enum class VisualPathway : std::uint8_t {
  retina_to_visual = 0,
  visual_to_encoding,
};

inline constexpr std::size_t visual_module_count = 3;
inline constexpr std::size_t visual_pathway_count = 2;

struct VisualConfig {
  std::size_t stimulus_ticks{15};
  std::size_t blank_ticks{15};
  std::size_t tolerated_bit_flips{1};
  double retina_current{1.0};
  double match_weight{1.0};
  double encoding_weight{1.0};
  snnbase::TimeStep refractory_period{1};
  std::uint32_t seed{42};
};

struct VisualModuleDiagnostics {
  std::size_t spikes{};
  std::size_t cumulative_spikes{};
  double mean_activity{};
};

struct VisualStep {
  snnbase::TimeStep time{};
  bool input_onset{};
  snnbase::SpikeEvent encoded_event;
  std::optional<char> recognition;
  std::array<VisualModuleDiagnostics, visual_module_count> modules{};
};

// Returns the fixed labels assigned to the 16 visual and encoding neurons.
[[nodiscard]] std::array<char, visual_symbol_count> visual_vocabulary()
    noexcept;

// Rasterizes the public 5x7 Spaun glyph. Values are binary (zero or one).
[[nodiscard]] Retina glyph_retina(char symbol);

// A spike-causal 5x7 visual front end. Recognition is produced only by an
// encoding-population spike after traversing both projections. Host code only
// rasterizes stimuli, suppresses duplicate onset pulses, and decodes the index
// of an emitted encoding neuron; it never compares an input with a prototype.
class VisualFrontend {
 public:
  explicit VisualFrontend(VisualConfig config = {});
  ~VisualFrontend();
  VisualFrontend(VisualFrontend&&) noexcept;
  VisualFrontend& operator=(VisualFrontend&&) noexcept;
  VisualFrontend(const VisualFrontend&) = delete;
  VisualFrontend& operator=(const VisualFrontend&) = delete;

  // Advances one neural tick with a public glyph or a caller-provided retina.
  // Repeating an unchanged retina is a sustained presentation and does not
  // create another retinal onset pulse.
  [[nodiscard]] VisualStep step(char symbol);
  [[nodiscard]] VisualStep step(const Retina& retina);
  [[nodiscard]] VisualStep step_blank();

  // Runs the configured stimulus/blank schedule and drains the two-projection
  // pipeline when necessary. The Retina overload is the causal test boundary:
  // no symbol label enters the circuit.
  [[nodiscard]] std::optional<char> present(char symbol);
  [[nodiscard]] std::optional<char> present(const Retina& retina);

  void reset();
  void set_projection_enabled(VisualPathway pathway, bool enabled);
  [[nodiscard]] bool projection_enabled(VisualPathway pathway) const;

  [[nodiscard]] const VisualConfig& config() const noexcept;
  [[nodiscard]] std::size_t neuron_count() const noexcept;
  [[nodiscard]] std::size_t synapse_count() const noexcept;
  [[nodiscard]] std::size_t projection_count() const noexcept;
  [[nodiscard]] std::size_t module_neuron_count(VisualModule module) const;
  [[nodiscard]] std::size_t cumulative_spikes(VisualModule module) const;
  [[nodiscard]] std::span<const std::uint8_t> module_spikes(
      VisualModule module) const;
  [[nodiscard]] std::span<const double> module_activity(
      VisualModule module) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace snnbase_experiments::spaun

#endif  // SNNBASE_EXPERIMENTS_SPAUN_VISUAL_HPP
