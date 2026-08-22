#include <snnbase_experiments/spaun_visual.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace snnbase_experiments::spaun {
namespace {

constexpr std::array<char, visual_symbol_count> symbols{
    '0', '1', '2', '3', '4', '5', '6', '7',
    '8', '9', 'A', 'P', 'K', '[', ']', '?'};
constexpr std::size_t pipeline_delay = 2;

static_assert(snnbase::SpikeEvent::payload_width() >= visual_symbol_count,
              "SpikeEvent payload cannot represent the visual vocabulary");

std::size_t module_index(const VisualModule module) {
  const auto index = static_cast<std::size_t>(module);
  if (index >= visual_module_count) {
    throw std::out_of_range("visual module is out of range");
  }
  return index;
}

std::size_t pathway_index(const VisualPathway pathway) {
  const auto index = static_cast<std::size_t>(pathway);
  if (index >= visual_pathway_count) {
    throw std::out_of_range("visual pathway is out of range");
  }
  return index;
}

std::size_t hamming_distance(const Retina& left, const Retina& right) {
  std::size_t distance = 0;
  for (std::size_t pixel = 0; pixel < retina_pixel_count; ++pixel) {
    distance += left[pixel] == right[pixel] ? 0U : 1U;
  }
  return distance;
}

std::size_t maximum_unique_bit_flips(
    const std::array<Retina, visual_symbol_count>& prototypes) {
  auto minimum_distance = retina_pixel_count;
  for (std::size_t left = 0; left < prototypes.size(); ++left) {
    for (std::size_t right = left + 1; right < prototypes.size(); ++right) {
      minimum_distance = std::min(
          minimum_distance,
          hamming_distance(prototypes[left], prototypes[right]));
    }
  }
  return (minimum_distance - 1U) / 2U;
}

void validate_config(const VisualConfig& config,
                     const std::size_t maximum_bit_flips) {
  if (config.stimulus_ticks == 0 || config.blank_ticks == 0) {
    throw std::invalid_argument(
        "visual stimulus and blank durations must be positive");
  }
  if (config.tolerated_bit_flips > maximum_bit_flips) {
    throw std::invalid_argument(
        "visual bit-flip tolerance exceeds the vocabulary's unique decoding "
        "radius");
  }
  if (!std::isfinite(config.retina_current) ||
      !(config.retina_current > 0.5)) {
    throw std::invalid_argument(
        "visual retinal current must be finite and exceed 0.5");
  }
  if (!std::isfinite(config.match_weight) ||
      !(config.match_weight > 0.0) ||
      !std::isfinite(config.encoding_weight) ||
      !(config.encoding_weight > 0.0)) {
    throw std::invalid_argument(
        "visual projection weights must be finite and positive");
  }
}

std::size_t count_spikes(const std::span<const std::uint8_t> spikes) {
  return std::accumulate(spikes.begin(), spikes.end(), std::size_t{});
}

double mean_activity(const std::span<const double> activity) {
  if (activity.empty()) {
    return 0.0;
  }
  return std::accumulate(activity.begin(), activity.end(), 0.0) /
         static_cast<double>(activity.size());
}

}  // namespace

std::array<char, visual_symbol_count> visual_vocabulary() noexcept {
  return symbols;
}

Retina glyph_retina(const char symbol) {
  const auto rows = glyph(symbol);
  Retina retina{};
  for (std::size_t row = 0; row < glyph_rows; ++row) {
    for (std::size_t column = 0; column < glyph_columns; ++column) {
      retina[row * glyph_columns + column] =
          rows[row][column] == '#' ? 1U : 0U;
    }
  }
  return retina;
}

struct VisualFrontend::Impl {
  explicit Impl(VisualConfig selected_config)
      : config(std::move(selected_config)) {
    for (std::size_t index = 0; index < symbols.size(); ++index) {
      prototypes[index] = glyph_retina(symbols[index]);
    }
    validate_config(config, maximum_unique_bit_flips(prototypes));
    build_circuit();
  }

  void build_circuit() {
    using snnbase::population::PopulationConfig;
    using snnbase::population::ProjectionConfig;
    using snnbase::population::ProjectionLayout;

    populations[module_index(VisualModule::retina)] = network.add_population(
        PopulationConfig{.size = opponent_retina_neuron_count,
                         .threshold = 0.5,
                         .leak = 0.0,
                         .bias = 0.0,
                         .noise_standard_deviation = 0.0,
                         .activity_decay = 0.5,
                         .refractory_period = config.refractory_period,
                         .seed = config.seed});

    const auto accepted_matches =
        retina_pixel_count - config.tolerated_bit_flips;
    const auto visual_threshold =
        (static_cast<double>(accepted_matches) - 0.5) * config.match_weight;
    populations[module_index(VisualModule::visual)] = network.add_population(
        PopulationConfig{.size = visual_symbol_count,
                         .threshold = visual_threshold,
                         .leak = 0.0,
                         .bias = 0.0,
                         .noise_standard_deviation = 0.0,
                         .activity_decay = 0.5,
                         .refractory_period = config.refractory_period,
                         .seed = config.seed + 1U});
    populations[module_index(VisualModule::encoding)] = network.add_population(
        PopulationConfig{.size = visual_symbol_count,
                         .threshold = config.encoding_weight * 0.5,
                         .leak = 0.0,
                         .bias = 0.0,
                         .noise_standard_deviation = 0.0,
                         .activity_decay = 0.5,
                         .refractory_period = config.refractory_period,
                         .seed = config.seed + 2U});

    std::vector<double> visual_weights(
        visual_symbol_count * opponent_retina_neuron_count, 0.0);
    for (std::size_t target = 0; target < visual_symbol_count; ++target) {
      for (std::size_t pixel = 0; pixel < retina_pixel_count; ++pixel) {
        const auto opponent =
            2U * pixel + (prototypes[target][pixel] == 0U ? 1U : 0U);
        visual_weights[target * opponent_retina_neuron_count + opponent] =
            config.match_weight;
      }
    }

    pathways[pathway_index(VisualPathway::retina_to_visual)] = network.connect(
        ProjectionConfig{
            .source = populations[module_index(VisualModule::retina)],
            .target = populations[module_index(VisualModule::visual)],
            .layout = ProjectionLayout::dense,
            .weights = std::move(visual_weights),
            .sparse_weights = {},
            .delay = 1,
            .synaptic_decay = 0.0,
            .plasticity = snnbase::population::PlasticityRule::fixed,
            .learning_rate = 0.01,
            .eligibility_decay = 0.95,
            .minimum_weight = 0.0,
            .maximum_weight = std::max(4.0, config.match_weight),
            .enabled = true});
    pathways[pathway_index(VisualPathway::visual_to_encoding)] =
        network.connect(
            ProjectionConfig{
                .source = populations[module_index(VisualModule::visual)],
                .target = populations[module_index(VisualModule::encoding)],
                .layout = ProjectionLayout::diagonal,
                .weights = {config.encoding_weight},
                .sparse_weights = {},
                .delay = 1,
                .synaptic_decay = 0.0,
                .plasticity = snnbase::population::PlasticityRule::fixed,
                .learning_rate = 0.01,
                .eligibility_decay = 0.95,
                .minimum_weight = 0.0,
                .maximum_weight = std::max(4.0, config.encoding_weight),
                .enabled = true});
  }

  VisualStep advance(const Retina& retina) {
    const auto input_onset = !previous_retina.has_value() ||
                             previous_retina.value() != retina;
    previous_retina = retina;

    std::vector<double> retinal_current(opponent_retina_neuron_count, 0.0);
    std::array<snnbase::population::PopulationInput, 1> inputs{};
    std::span<const snnbase::population::PopulationInput> supplied;
    if (input_onset) {
      for (std::size_t pixel = 0; pixel < retina_pixel_count; ++pixel) {
        const auto opponent =
            2U * pixel + (retina[pixel] == 0U ? 1U : 0U);
        retinal_current[opponent] = config.retina_current;
      }
      inputs[0] = {.population =
                       populations[module_index(VisualModule::retina)],
                   .current = retinal_current};
      supplied = inputs;
    }

    const auto summary = network.step(supplied);
    VisualStep result;
    result.time = summary.time;
    result.input_onset = input_onset;

    for (std::size_t module = 0; module < visual_module_count; ++module) {
      const auto spikes = network.spikes(populations[module]);
      cumulative[module] += count_spikes(spikes);
      result.modules[module] = {
          .spikes = count_spikes(spikes),
          .cumulative_spikes = cumulative[module],
          .mean_activity =
              mean_activity(network.filtered_activity(populations[module]))};
    }

    const auto encoding_spikes =
        network.spikes(populations[module_index(VisualModule::encoding)]);
    snnbase::SpikeEvent::Storage event_bits = 0;
    std::size_t encoded_count = 0;
    std::size_t encoded_index = 0;
    for (std::size_t index = 0; index < encoding_spikes.size(); ++index) {
      if (encoding_spikes[index] != 0U) {
        event_bits |= snnbase::SpikeEvent::payload_bit(index);
        encoded_index = index;
        ++encoded_count;
      }
    }
    result.encoded_event = snnbase::SpikeEvent(
        event_bits, populations[module_index(VisualModule::encoding)],
        summary.time);
    if (encoded_count == 1) {
      result.recognition = symbols[encoded_index];
    }
    return result;
  }

  std::optional<char> run_presentation(const Retina& retina) {
    std::optional<char> recognition;
    const auto capture = [&recognition](const VisualStep& step) {
      if (step.recognition.has_value()) {
        recognition = step.recognition;
      }
    };

    for (std::size_t tick = 0; tick < config.stimulus_ticks; ++tick) {
      capture(advance(retina));
    }
    const auto blank = glyph_retina(' ');
    for (std::size_t tick = 0; tick < config.blank_ticks; ++tick) {
      capture(advance(blank));
    }
    for (std::size_t tick = 0;
         tick < pipeline_delay && !recognition.has_value(); ++tick) {
      capture(advance(blank));
    }
    return recognition;
  }

  VisualConfig config;
  std::array<Retina, visual_symbol_count> prototypes{};
  snnbase::population::Network network;
  std::array<snnbase::population::PopulationId, visual_module_count>
      populations{};
  std::array<snnbase::population::ProjectionId, visual_pathway_count>
      pathways{};
  std::array<std::size_t, visual_module_count> cumulative{};
  std::optional<Retina> previous_retina;
};

VisualFrontend::VisualFrontend(VisualConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
VisualFrontend::~VisualFrontend() = default;
VisualFrontend::VisualFrontend(VisualFrontend&&) noexcept = default;
VisualFrontend& VisualFrontend::operator=(VisualFrontend&&) noexcept = default;

VisualStep VisualFrontend::step(const char symbol) {
  return impl_->advance(glyph_retina(symbol));
}

VisualStep VisualFrontend::step(const Retina& retina) {
  for (const auto pixel : retina) {
    if (pixel > 1U) {
      throw std::invalid_argument("visual retina values must be zero or one");
    }
  }
  return impl_->advance(retina);
}

VisualStep VisualFrontend::step_blank() { return step(' '); }

std::optional<char> VisualFrontend::present(const char symbol) {
  return present(glyph_retina(symbol));
}

std::optional<char> VisualFrontend::present(const Retina& retina) {
  for (const auto pixel : retina) {
    if (pixel > 1U) {
      throw std::invalid_argument("visual retina values must be zero or one");
    }
  }
  return impl_->run_presentation(retina);
}

void VisualFrontend::reset() {
  impl_->network.reset_state();
  impl_->cumulative.fill(0);
  impl_->previous_retina.reset();
}

void VisualFrontend::set_projection_enabled(const VisualPathway pathway,
                                            const bool enabled) {
  impl_->network.set_projection_enabled(
      impl_->pathways[pathway_index(pathway)], enabled);
}

bool VisualFrontend::projection_enabled(const VisualPathway pathway) const {
  return impl_->network.projection_enabled(
      impl_->pathways[pathway_index(pathway)]);
}

const VisualConfig& VisualFrontend::config() const noexcept {
  return impl_->config;
}

std::size_t VisualFrontend::neuron_count() const noexcept {
  return impl_->network.neuron_count();
}

std::size_t VisualFrontend::synapse_count() const noexcept {
  std::size_t count = 0;
  for (const auto pathway : impl_->pathways) {
    count += impl_->network.projection_weights(pathway).size();
  }
  return count;
}

std::size_t VisualFrontend::projection_count() const noexcept {
  return impl_->network.projection_count();
}

std::size_t VisualFrontend::module_neuron_count(
    const VisualModule module) const {
  return impl_->network
      .population_config(impl_->populations[module_index(module)])
      .size;
}

std::size_t VisualFrontend::cumulative_spikes(const VisualModule module) const {
  return impl_->cumulative[module_index(module)];
}

std::span<const std::uint8_t> VisualFrontend::module_spikes(
    const VisualModule module) const {
  return impl_->network.spikes(impl_->populations[module_index(module)]);
}

std::span<const double> VisualFrontend::module_activity(
    const VisualModule module) const {
  return impl_->network.filtered_activity(
      impl_->populations[module_index(module)]);
}

}  // namespace snnbase_experiments::spaun
