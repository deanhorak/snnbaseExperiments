#include <snnbase_experiments/spaun_visual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using snnbase_experiments::spaun::Retina;
using snnbase_experiments::spaun::VisualConfig;
using snnbase_experiments::spaun::VisualFrontend;
using snnbase_experiments::spaun::VisualModule;
using snnbase_experiments::spaun::VisualPathway;
using snnbase_experiments::spaun::VisualStep;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

VisualConfig fast_config() {
  return {.stimulus_ticks = 1,
          .blank_ticks = 1,
          .tolerated_bit_flips = 1,
          .retina_current = 1.0,
          .match_weight = 1.0,
          .encoding_weight = 1.0,
          .refractory_period = 1,
          .seed = 91};
}

void test_structure_and_exact_vocabulary() {
  using namespace snnbase_experiments::spaun;
  VisualFrontend frontend(fast_config());
  require(frontend.neuron_count() == 70U + 16U + 16U,
          "visual front end neuron count is incorrect");
  require(frontend.synapse_count() == 70U * 16U + 16U,
          "visual front end synapse count is incorrect");
  require(frontend.projection_count() == 2,
          "visual front end must have exactly two projections");
  require(frontend.module_neuron_count(VisualModule::retina) == 70,
          "retina must have one ON/OFF opponent pair per pixel");

  for (const auto symbol : visual_vocabulary()) {
    frontend.reset();
    const auto retina = glyph_retina(symbol);
    const auto recognized = frontend.present(retina);
    require(recognized == std::optional<char>{symbol},
            "retina-only recognition failed for symbol " +
                std::string(1, symbol));
    require(frontend.cumulative_spikes(VisualModule::retina) > 0,
            "retina did not spike");
    require(frontend.cumulative_spikes(VisualModule::visual) == 1,
            "exact glyph did not create exactly one visual spike");
    require(frontend.cumulative_spikes(VisualModule::encoding) == 1,
            "exact glyph did not create exactly one encoding spike");
  }
}

void test_registered_bit_flip_robustness_and_rejection() {
  using namespace snnbase_experiments::spaun;
  VisualFrontend frontend(fast_config());
  std::size_t flip = 0;
  for (const auto symbol : visual_vocabulary()) {
    auto retina = glyph_retina(symbol);
    retina[flip % retina.size()] ^= 1U;
    frontend.reset();
    require(frontend.present(retina) == std::optional<char>{symbol},
            "one-bit retinal corruption was not corrected for symbol " +
                std::string(1, symbol));
    ++flip;
  }

  // K is at least 13 bits from every other vocabulary glyph. Two flips leave
  // this raster outside every registered radius-one neural decision region.
  auto rejected = glyph_retina('K');
  rejected[0] ^= 1U;
  rejected[1] ^= 1U;
  frontend.reset();
  require(!frontend.present(rejected).has_value(),
          "a raster beyond the registered bit-flip budget was accepted");

  auto invalid_config = fast_config();
  invalid_config.tolerated_bit_flips = 2;
  bool rejected_config = false;
  try {
    VisualFrontend invalid(invalid_config);
  } catch (const std::invalid_argument&) {
    rejected_config = true;
  }
  require(rejected_config,
          "ambiguous bit-flip tolerance was not rejected at registration");
}

void test_causal_pathway_lesions() {
  using namespace snnbase_experiments::spaun;
  VisualFrontend frontend(fast_config());
  const auto retina = glyph_retina('8');

  frontend.set_projection_enabled(VisualPathway::retina_to_visual, false);
  require(!frontend.present(retina).has_value(),
          "recognition survived a retina-to-visual lesion");
  require(frontend.cumulative_spikes(VisualModule::retina) > 0,
          "lesioned trial did not drive the retinal boundary");
  require(frontend.cumulative_spikes(VisualModule::visual) == 0,
          "retina-to-visual lesion leaked visual spikes");
  require(frontend.cumulative_spikes(VisualModule::encoding) == 0,
          "retina-to-visual lesion leaked encoding spikes");

  frontend.set_projection_enabled(VisualPathway::retina_to_visual, true);
  frontend.set_projection_enabled(VisualPathway::visual_to_encoding, false);
  frontend.reset();
  require(!frontend.present(retina).has_value(),
          "recognition survived a visual-to-encoding lesion");
  require(frontend.cumulative_spikes(VisualModule::visual) == 1,
          "visual classifier did not spike upstream of encoding lesion");
  require(frontend.cumulative_spikes(VisualModule::encoding) == 0,
          "visual-to-encoding lesion leaked an output spike");
}

struct TracePoint {
  bool onset{};
  snnbase::SpikeEvent::Storage bits{};
  std::array<std::size_t, 3> spikes{};
};

std::vector<TracePoint> trace_sequence(VisualFrontend& frontend) {
  using namespace snnbase_experiments::spaun;
  std::vector<TracePoint> trace;
  for (const auto symbol : std::array<char, 7>{'6', '6', '6', ' ', ' ',
                                               '6', '6'}) {
    const auto step = frontend.step(symbol);
    trace.push_back(
        {.onset = step.input_onset,
         .bits = step.encoded_event.bits(),
         .spikes = {step.modules[0].spikes, step.modules[1].spikes,
                    step.modules[2].spikes}});
  }
  return trace;
}

void test_deterministic_reset_and_onset_pulses() {
  using namespace snnbase_experiments::spaun;
  VisualFrontend frontend(fast_config());
  const auto first = trace_sequence(frontend);
  frontend.reset();
  const auto second = trace_sequence(frontend);
  require(first.size() == second.size(), "deterministic trace size changed");
  for (std::size_t index = 0; index < first.size(); ++index) {
    require(first[index].onset == second[index].onset &&
                first[index].bits == second[index].bits &&
                first[index].spikes == second[index].spikes,
            "reset did not reproduce the visual spike trace");
  }

  frontend.reset();
  std::size_t recognition_pulses = 0;
  std::size_t onset_pulses = 0;
  std::size_t retinal_spikes = 0;
  for (std::size_t tick = 0; tick < 8; ++tick) {
    const VisualStep step = frontend.step('3');
    recognition_pulses += step.recognition.has_value() ? 1U : 0U;
    onset_pulses += step.input_onset ? 1U : 0U;
    retinal_spikes += step.modules[0].spikes;
  }
  require(onset_pulses == 1,
          "sustained presentation created duplicate retinal onsets");
  require(retinal_spikes == 35,
          "opponent retina did not emit exactly one spike per pixel");
  require(recognition_pulses == 1,
          "sustained presentation created duplicate token outputs");
}

}  // namespace

int main() {
  try {
    test_structure_and_exact_vocabulary();
    test_registered_bit_flip_robustness_and_rejection();
    test_causal_pathway_lesions();
    test_deterministic_reset_and_onset_pulses();
    std::cout << "spaun visual tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "spaun visual tests failed: " << error.what() << '\n';
    return 1;
  }
}
