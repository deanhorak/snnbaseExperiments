#ifndef SNNBASE_EXPERIMENTS_TEMPORAL_CHAT_MODEL_HPP
#define SNNBASE_EXPERIMENTS_TEMPORAL_CHAT_MODEL_HPP
#include <snnbase/population.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace temporal_chat {
using Token = std::uint32_t;
struct Example { std::vector<Token> prompt, answer; };
struct Metrics { double loss{}, accuracy{}; std::size_t tokens{}; };

// Fixed spiking reservoir with an explicitly engineered categorical readout.
// All readout inputs are causal traces of spikes emitted by snnbase neurons.
class Model {
 public:
  static constexpr std::size_t input_size = 128, excitatory_size = 192,
      inhibitory_size = 32, steps_per_token = 6;
  static constexpr std::size_t channels = excitatory_size + inhibitory_size;
  static constexpr std::size_t features = 1 + channels * 2;
  Model(std::vector<Token> vocabulary, std::uint32_t seed = 42, bool lesion = false)
      : vocabulary_(std::move(vocabulary)), seed_(seed),
        fast_(channels), slow_(channels), lesion_(lesion) {
    if (vocabulary_.size() < 2 || vocabulary_.size() > 512 ||
        !std::is_sorted(vocabulary_.begin(), vocabulary_.end()) ||
        std::adjacent_find(vocabulary_.begin(), vocabulary_.end()) != vocabulary_.end())
      throw std::invalid_argument("output vocabulary must have 2..512 sorted unique tokens");
    weights_.resize(vocabulary_.size() * features);
    using namespace snnbase::population;
    PopulationConfig input; input.size = input_size; input.leak = 0.0;
    input.activity_decay = 0.0; input.seed = seed;
    input_ = network_.add_population(input);
    PopulationConfig exc; exc.size = excitatory_size; exc.leak = 0.65;
    exc.threshold = 1.0; exc.refractory_period = 1; exc.seed = seed;
    exc.dendritic_branches = 2; exc.dendritic_decay = 0.65;
    exc.dendritic_threshold = 1.0; exc.dendritic_output_gain = 0.8;
    exc.adaptation_increment = 0.08; exc.adaptation_decay = 0.94;
    excitatory_ = network_.add_population(exc);
    PopulationConfig inh; inh.size = inhibitory_size; inh.leak = 0.5;
    inh.threshold = 0.9; inh.refractory_period = 1; inh.seed = seed;
    inhibitory_ = network_.add_population(inh);
    PopulationConfig clock; clock.size = 1; clock.leak = 0.0;
    pacemaker_ = network_.add_population(clock);
    std::mt19937 random(seed);
    const auto sparse = [&](PopulationId source, PopulationId target,
                            std::size_t ns, std::size_t nt, std::size_t degree,
                            double weight, std::optional<std::size_t> branch,
                            snnbase::TimeStep delay) {
      ProjectionConfig p; p.source = source; p.target = target;
      p.layout = ProjectionLayout::sparse; p.delay = delay;
      p.target_branch = branch;
      if (source == input_ && target == excitatory_) {
        p.plasticity = PlasticityRule::stdp;
        p.learning_enabled = false;
        p.learning_rate = 0.00005;
        p.stdp_potentiation = 1.0; p.stdp_depression = 1.05;
      }
      p.polarity = weight > 0 ? ProjectionPolarity::excitatory : ProjectionPolarity::inhibitory;
      p.minimum_weight = weight > 0 ? 0.0 : -4.0;
      p.maximum_weight = weight > 0 ? 4.0 : 0.0;
      for (std::size_t t = 0; t < nt; ++t) {
        std::vector<std::size_t> selected;
        while (selected.size() < degree) {
          const auto s = static_cast<std::size_t>(random()) % ns;
          if (std::find(selected.begin(), selected.end(), s) != selected.end()) continue;
          selected.push_back(s); p.sparse_weights.push_back({s, t, weight});
        }
      }
      return network_.connect(std::move(p));
    };
    // Separate branches detect coincident token input and recurrent context.
    input_projection_ = sparse(input_, excitatory_, input_size, excitatory_size, 14, 0.6, 0, 1);
    static_cast<void>(sparse(excitatory_, excitatory_, excitatory_size, excitatory_size, 6, 0.20, 1, 2));
    static_cast<void>(sparse(excitatory_, inhibitory_, excitatory_size, inhibitory_size, 12, 0.22, {}, 1));
    static_cast<void>(sparse(inhibitory_, excitatory_, inhibitory_size, excitatory_size, 4, -0.28, {}, 1));
    // A population supplies common periodic spikes, not direct phase resets.
    static_cast<void>(sparse(pacemaker_, excitatory_, 1, excitatory_size, 1, 0.26, 1, 1));
    static_cast<void>(sparse(pacemaker_, inhibitory_, 1, inhibitory_size, 1, 0.15, {}, 1));
  }
  double warmup(const std::vector<Example>& examples) {
    if (frozen_) throw std::logic_error("readout is frozen");
    const auto initial = network_.projection_weights(input_projection_);
    const std::vector<double> before(initial.begin(), initial.end());
    network_.set_projection_learning_enabled(input_projection_, true);
    for (const auto& example : examples) {
      reset();
      for (auto token : example.prompt) static_cast<void>(observe(token));
      for (auto token : example.answer) static_cast<void>(observe(token));
    }
    network_.set_projection_learning_enabled(input_projection_, false);
    const auto after = network_.projection_weights(input_projection_);
    double change = 0.0;
    for (std::size_t i = 0; i < before.size(); ++i) change += std::abs(after[i] - before[i]);
    reset(); return change;
  }
  void reset() {
    network_.reset_state();
    std::fill(fast_.begin(), fast_.end(), 0.0);
    std::fill(slow_.begin(), slow_.end(), 0.0);
  }
  std::vector<double> observe(Token token) {
    std::array<double, input_size> current{};
    // Stable 16-of-128 distributed code; accepts unseen upstream tokens.
    std::array<bool, input_size> selected{};
    std::uint64_t bits = (static_cast<std::uint64_t>(token) << 32) ^ seed_;
    for (std::size_t count = 0; count < 16;) {
      bits += 0x9e3779b97f4a7c15ULL;
      auto z = bits; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      const auto i = static_cast<std::size_t>((z ^ (z >> 31)) % input_size);
      if (!selected[i]) { selected[i] = true; ++count; }
    }
    for (std::size_t step = 0; step < steps_per_token; ++step) {
      for (std::size_t i = 0; i < input_size; ++i)
        current[i] = selected[i] && (step == i % 3 || step == i % 3 + 3) ? 1.05 : 0.0;
      const std::array<double, 1> clock{step == 0 ? 1.05 : 0.0};
      const std::array<snnbase::population::PopulationInput, 2> inputs{{{input_, current}, {pacemaker_, clock}}};
      static_cast<void>(network_.step(inputs));
      std::size_t offset = 0;
      for (auto pop : {excitatory_, inhibitory_}) {
        for (const auto spike : network_.spikes(pop)) {
          fast_[offset] = 0.55 * fast_[offset] + static_cast<double>(spike);
          slow_[offset] = 0.985 * slow_[offset] + 0.10 * static_cast<double>(spike);
          ++offset;
        }
      }
    }
    std::vector<double> x(features, 1.0);
    std::copy(fast_.begin(), fast_.end(), x.begin() + 1);
    std::copy(slow_.begin(), slow_.end(), x.begin() + 1 + channels);
    if (lesion_) std::fill(x.begin() + 1, x.end(), 0.0);
    return x;
  }
  std::vector<double> probabilities(const std::vector<double>& x) const {
    if (x.size() != features) throw std::invalid_argument("wrong feature size");
    std::vector<double> p(vocabulary_.size());
    for (std::size_t k = 0; k < p.size(); ++k)
      p[k] = std::inner_product(x.begin(), x.end(), weights_.begin() + k * features, 0.0);
    const auto max = *std::max_element(p.begin(), p.end());
    double sum = 0.0;
    for (auto& value : p) { value = std::exp(value - max); sum += value; }
    for (auto& value : p) value /= sum;
    return p;
  }
  struct Sample { std::vector<double> x; std::size_t target; };
  std::vector<Sample> encode(const std::vector<Example>& examples) {
    std::vector<Sample> samples;
    for (const auto& example : examples) {
      if (example.prompt.empty() || example.answer.empty()) throw std::invalid_argument("empty example");
      reset(); std::vector<double> x;
      for (auto token : example.prompt) x = observe(token);
      for (auto token : example.answer) {
        const auto found = std::lower_bound(vocabulary_.begin(), vocabulary_.end(), token);
        if (found == vocabulary_.end() || *found != token) throw std::invalid_argument("target outside training vocabulary");
        samples.push_back({x, static_cast<std::size_t>(found - vocabulary_.begin())});
        x = observe(token); // Teacher forcing; the target never enters its own features.
      }
    }
    return samples;
  }
  // Local three-factor readout: dw = eta * presynaptic trace * (teacher-p).
  // Softmax and teacher error are engineered supervision, not cortical claims.
  void fit(const std::vector<Sample>& samples, std::size_t epochs) {
    if (frozen_) throw std::logic_error("readout is frozen");
    std::vector<std::size_t> order(samples.size()); std::iota(order.begin(), order.end(), 0);
    std::mt19937 random(seed_);
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
      std::shuffle(order.begin(), order.end(), random);
      for (auto index : order) {
        const auto& sample = samples[index]; const auto p = probabilities(sample.x);
        const double norm = std::inner_product(sample.x.begin(), sample.x.end(), sample.x.begin(), 1.0);
        // Cap the step for the bias-only lesion, whose feature norm is tiny.
        const double eta = std::min(0.25, 3.0 / std::sqrt(norm));
        for (std::size_t k = 0; k < p.size(); ++k) {
          const double error = (sample.target == k ? 1.0 : 0.0) - p[k];
          for (std::size_t j = 0; j < features; ++j)
            weights_[k * features + j] += eta * error * sample.x[j];
        }
      }
    }
  }
  Metrics evaluate(const std::vector<Sample>& samples) const {
    Metrics m; m.tokens = samples.size();
    for (const auto& sample : samples) {
      const auto p = probabilities(sample.x);
      m.loss -= std::log(std::max(p[sample.target], 1e-300));
      m.accuracy += static_cast<std::size_t>(std::max_element(p.begin(), p.end()) - p.begin()) == sample.target ? 1.0 : 0.0;
    }
    if (m.tokens) { m.loss /= m.tokens; m.accuracy /= m.tokens; }
    return m;
  }
  std::vector<Token> generate(const std::vector<Token>& prompt, Token stop, std::size_t maximum) {
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    if (!frozen_) throw std::logic_error("freeze before generation");
    reset(); std::vector<double> x;
    for (auto token : prompt) x = observe(token);
    std::vector<Token> output;
    for (std::size_t step = 0; step < maximum; ++step) {
      const auto p = probabilities(x);
      const auto token = vocabulary_[static_cast<std::size_t>(std::max_element(p.begin(), p.end()) - p.begin())];
      output.push_back(token);
      if (token == stop) break;
      x = observe(token); // Actual generated tokens re-enter the spiking circuit.
    }
    return output;
  }
  void freeze() { frozen_ = true; }
  const std::vector<double>& weights() const { return weights_; }
  const std::vector<Token>& vocabulary() const { return vocabulary_; }
  std::size_t spike_count() const { return network_.total_spikes(); }
  std::size_t neuron_count() const { return network_.neuron_count(); }
  std::size_t synapse_count() const { return network_.synapse_count(); }
 private:
  std::vector<Token> vocabulary_; std::uint32_t seed_;
  std::vector<double> weights_, fast_, slow_; bool lesion_, frozen_ = false;
  snnbase::population::Network network_;
  snnbase::population::ProjectionId input_projection_{};
  snnbase::population::PopulationId input_{}, excitatory_{}, inhibitory_{}, pacemaker_{};
};
} // namespace temporal_chat
#endif
