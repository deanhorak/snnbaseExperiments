#include <snnbase_experiments/spaun.hpp>
#include <snnbase_experiments/spaun_visual.hpp>

#include <snnbase/spiking_conv.hpp>
#include <snnbase_experiments/spaun_workspace.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace snnbase_experiments::spaun {
namespace {

using Glyph = std::array<std::string_view, glyph_rows>;

constexpr Glyph blank{".....", ".....", ".....", ".....", ".....",
                      ".....", "....."};
constexpr Glyph zero{".###.", "#...#", "#..##", "#.#.#", "##..#",
                     "#...#", ".###."};
constexpr Glyph one{"..#..", ".##..", "..#..", "..#..", "..#..",
                    "..#..", ".###."};
constexpr Glyph two{".###.", "#...#", "....#", "...#.", "..#..",
                    ".#...", "#####"};
constexpr Glyph three{"####.", "....#", "....#", ".###.", "....#",
                      "....#", "####."};
constexpr Glyph four{"...#.", "..##.", ".#.#.", "#..#.", "#####",
                     "...#.", "...#."};
constexpr Glyph five{"#####", "#....", "#....", "####.", "....#",
                     "....#", "####."};
constexpr Glyph six{".###.", "#....", "#....", "####.", "#...#",
                    "#...#", ".###."};
constexpr Glyph seven{"#####", "....#", "...#.", "..#..", ".#...",
                      ".#...", ".#..."};
constexpr Glyph eight{".###.", "#...#", "#...#", ".###.", "#...#",
                      "#...#", ".###."};
constexpr Glyph nine{".###.", "#...#", "#...#", ".####", "....#",
                     "....#", ".###."};
constexpr Glyph letter_a{".###.", "#...#", "#...#", "#####", "#...#",
                         "#...#", "#...#"};
constexpr Glyph letter_p{"####.", "#...#", "#...#", "####.", "#....",
                         "#....", "#...."};
constexpr Glyph letter_k{"#...#", "#..#.", "#.#..", "##...", "#.#..",
                         "#..#.", "#...#"};
constexpr Glyph left_bracket{".###.", ".#...", ".#...", ".#...", ".#...",
                             ".#...", ".###."};
constexpr Glyph right_bracket{".###.", "...#.", "...#.", "...#.", "...#.",
                              "...#.", ".###."};
constexpr Glyph question{".###.", "#...#", "....#", "...#.", "..#..",
                         ".....", "..#.."};

constexpr std::array<char, 16> vocabulary{
    '0', '1', '2', '3', '4', '5', '6', '7',
    '8', '9', 'A', 'P', 'K', '[', ']', '?'};

constexpr std::array<std::pair<Module, Module>, 12> pathways{{
    {Module::visual, Module::encoding},
    {Module::encoding, Module::working_memory},
    {Module::encoding, Module::transformation},
    {Module::encoding, Module::reward},
    {Module::working_memory, Module::transformation},
    {Module::working_memory, Module::decoding},
    {Module::transformation, Module::action_selection},
    {Module::reward, Module::action_selection},
    {Module::action_selection, Module::working_memory},
    {Module::action_selection, Module::decoding},
    {Module::decoding, Module::motor},
    {Module::motor, Module::arm},
}};

std::size_t module_index(Module module) {
  return static_cast<std::size_t>(module);
}

std::vector<int> number_digits(int value) {
  if (value == 0) {
    return {0};
  }
  std::vector<int> result;
  for (value = std::abs(value); value > 0; value /= 10) {
    result.push_back(value % 10);
  }
  std::reverse(result.begin(), result.end());
  return result;
}

double clamp_unit(double value) {
  return std::clamp(value, -1.0, 1.0);
}

using StrokePoint = std::array<double, 2>;
using Stroke = std::vector<StrokePoint>;
using DigitStrokes = std::vector<Stroke>;

DigitStrokes digit_strokes(int digit) {
  switch (digit) {
    case 0:
      return {{{0.25, 1.00}, {0.75, 1.00}, {1.00, 0.80},
               {1.00, 0.20}, {0.75, 0.00}, {0.25, 0.00},
               {0.00, 0.20}, {0.00, 0.80}, {0.25, 1.00}}};
    case 1:
      return {{{0.20, 0.75}, {0.50, 1.00}, {0.50, 0.00}},
              {{0.18, 0.00}, {0.82, 0.00}}};
    case 2:
      return {{{0.00, 0.78}, {0.18, 0.96}, {0.72, 1.00},
               {0.96, 0.84}, {1.00, 0.68}, {0.82, 0.52},
               {0.08, 0.08}, {0.00, 0.00}, {1.00, 0.00}}};
    case 3:
      return {{{0.04, 0.91}, {0.24, 1.00}, {0.78, 0.98},
               {1.00, 0.80}, {0.88, 0.64}, {0.55, 0.52},
               {0.88, 0.46}, {1.00, 0.24}, {0.80, 0.04},
               {0.22, 0.00}, {0.00, 0.10}}};
    case 4:
      return {{{0.82, 0.00}, {0.82, 1.00}, {0.00, 0.35},
               {1.00, 0.35}}};
    case 5:
      return {{{1.00, 1.00}, {0.08, 1.00}, {0.00, 0.55},
               {0.72, 0.55}, {0.96, 0.40}, {1.00, 0.20},
               {0.80, 0.02}, {0.20, 0.00}, {0.00, 0.10}}};
    case 6:
      return {{{0.94, 0.90}, {0.72, 1.00}, {0.28, 0.94},
               {0.04, 0.70}, {0.00, 0.25}, {0.18, 0.04},
               {0.72, 0.00}, {0.98, 0.20}, {0.90, 0.48},
               {0.68, 0.58}, {0.08, 0.54}}};
    case 7:
      return {{{0.00, 1.00}, {1.00, 1.00}, {0.66, 0.68},
               {0.34, 0.00}}};
    case 8:
      return {{{0.50, 0.50}, {0.16, 0.64}, {0.02, 0.84},
               {0.22, 1.00}, {0.78, 1.00}, {0.98, 0.84},
               {0.84, 0.64}, {0.50, 0.50}, {0.16, 0.36},
               {0.02, 0.16}, {0.22, 0.00}, {0.78, 0.00},
               {0.98, 0.16}, {0.84, 0.36}, {0.50, 0.50}}};
    case 9:
      return {{{0.90, 0.46}, {0.30, 0.42}, {0.06, 0.58},
               {0.02, 0.82}, {0.24, 1.00}, {0.76, 0.98},
               {0.98, 0.76}, {0.92, 0.30}, {0.72, 0.04},
               {0.28, 0.00}, {0.06, 0.10}}};
    default:
      throw std::invalid_argument("motor plan requires a decimal digit");
  }
}

}  // namespace

std::string_view task_name(Task task) noexcept {
  constexpr std::array<std::string_view, task_count> names{
      "copy drawing",          "image recognition",
      "reinforcement learning", "serial working memory",
      "counting",              "question answering",
      "rapid variable creation", "fluid reasoning"};
  const auto index = static_cast<std::size_t>(task);
  return index < names.size() ? names[index] : "unknown";
}

std::string_view module_name(Module module) noexcept {
  constexpr std::array<std::string_view, module_count> names{
      "visual", "encoding", "working memory", "transformation", "reward",
      "action selection", "decoding", "motor", "arm"};
  const auto index = module_index(module);
  return index < names.size() ? names[index] : "unknown";
}

Task parse_task(std::string_view value) {
  constexpr std::array<std::string_view, task_count> short_names{
      "copy", "recognition", "bandit", "memory", "counting", "question",
      "variable", "reasoning"};
  for (std::size_t index = 0; index < task_count; ++index) {
    if (value == short_names[index] || value == task_name(static_cast<Task>(index)) ||
        value == std::to_string(index) || value == "A" + std::to_string(index)) {
      return static_cast<Task>(index);
    }
  }
  throw std::invalid_argument("unknown Spaun task: " + std::string(value));
}

std::array<Task, task_count> all_tasks() noexcept {
  return {Task::copy_drawing,
          Task::image_recognition,
          Task::reinforcement_learning,
          Task::serial_working_memory,
          Task::counting,
          Task::question_answering,
          Task::rapid_variable_creation,
          Task::fluid_reasoning};
}

Trial canonical_trial(Task task) {
  switch (task) {
    case Task::copy_drawing:
      return {task, "copy handwritten digit", "A0[2]?", {{2}}, {}, 0,
              {0.72, 0.12, 0.12}, {2}, {}};
    case Task::image_recognition:
      return {task, "recognize digit", "A1[7]?", {{7}}, {}, 0,
              {0.72, 0.12, 0.12}, {7}, {}};
    case Task::reinforcement_learning:
      return {task, "switching three-armed bandit", "A2?", {}, {}, 0,
              {0.12, 0.12, 0.72}, {0}, {}};
    case Task::serial_working_memory:
      return {task, "recall a six-item list", "A3[015873]?",
              {{0, 1, 5, 8, 7, 3}}, {}, 0, {0.72, 0.12, 0.12},
              {0, 1, 5, 8, 7, 3}, {}};
    case Task::counting:
      return {task, "count five from three", "A4[3][5]?", {{3}, {5}}, {}, 0,
              {0.72, 0.12, 0.12}, {8}, {}};
    case Task::question_answering:
      return {task, "second item in a list", "A5[015873][P][2]?",
              {{0, 1, 5, 8, 7, 3}, {2}}, 'P', 2, {0.72, 0.12, 0.12}, {1}, {}};
    case Task::rapid_variable_creation:
      return {task,
              "infer a suffix transformation",
              "A6[0014][14][0094][94][0074]?",
              {{0, 0, 1, 4}, {1, 4}, {0, 0, 9, 4}, {9, 4}, {0, 0, 7, 4}},
              {},
              0,
              {0.72, 0.12, 0.12},
              {7, 4}, {}};
    case Task::fluid_reasoning:
      return {task,
              "complete a repeated-item progression",
              "A7[1][11][111][4][44][444][5][55]?",
              {{1}, {1, 1}, {1, 1, 1}, {4}, {4, 4}, {4, 4, 4}, {5}, {5, 5}},
              {},
              0,
              {0.72, 0.12, 0.12},
              {5, 5, 5}, {}};
  }
  throw std::invalid_argument("invalid Spaun task");
}

std::array<std::string_view, glyph_rows> glyph(char symbol) {
  switch (symbol) {
    case '0': return zero;
    case '1': return one;
    case '2': return two;
    case '3': return three;
    case '4': return four;
    case '5': return five;
    case '6': return six;
    case '7': return seven;
    case '8': return eight;
    case '9': return nine;
    case 'A': return letter_a;
    case 'P': return letter_p;
    case 'K': return letter_k;
    case '[': return left_bracket;
    case ']': return right_bracket;
    case '?': return question;
    case ' ': return blank;
    default: throw std::invalid_argument("unsupported Spaun glyph");
  }
}

snnbase::SpikeEvent glyph_event(char symbol) {
  const auto rows = glyph(symbol);
  snnbase::SpikeEvent::Storage bits = 0;
  for (std::size_t row = 0; row < rows.size(); ++row) {
    for (std::size_t column = 0; column < rows[row].size(); ++column) {
      if (rows[row][column] == '#') {
        bits |= snnbase::SpikeEvent::payload_bit(row * glyph_columns + column);
      }
    }
  }
  return snnbase::SpikeEvent(bits);
}

std::string digits_string(std::span<const int> digits) {
  std::string result;
  result.reserve(digits.size());
  for (const auto digit : digits) {
    if (digit < 0 || digit > 9) {
      result += '?';
    } else {
      result += static_cast<char>('0' + digit);
    }
  }
  return result;
}

struct Model::Impl {
  explicit Impl(Config selected_config)
      : config(std::move(selected_config)),
        visual_frontend(VisualConfig{.stimulus_ticks = config.stimulus_ticks,
                                     .blank_ticks = config.blank_ticks,
                                     .tolerated_bit_flips = 1,
                                     .retina_current = 1.0,
                                     .match_weight = 1.0,
                                     .encoding_weight = 1.0,
                                     .refractory_period = 0,
                                     .seed = config.seed}),
        semantic_workspace(workspace::Config{
            .maximum_groups = 9,
            .maximum_list_length = 9,
            .neurons_per_sign = 1,
            .semantic_input_gain = 6.0,
            .projection_weight_gain = 1.1,
            .memory_noise_standard_deviation =
                config.memory_noise_standard_deviation,
            .memory_recurrent_weight_gain =
                config.memory_recurrent_weight_gain,
            .primacy_recurrent_weight_gain =
                config.primacy_recurrent_weight_gain,
            .recency_recurrent_weight_gain =
                config.recency_recurrent_weight_gain,
            .cleanup_similarity_threshold =
                config.memory_cleanup_similarity_threshold,
            .counting_delay_ticks = config.counting_delay_ticks,
            .seed = config.seed}),
        generator(config.seed),
        reward_selector(
            snnbase::action::RewardSelectorConfig{.seed = config.seed}) {
    if (config.neurons_per_module == 0 || config.stimulus_ticks == 0 ||
        config.blank_ticks == 0 || config.motor_ticks_per_target == 0 ||
        !(config.tick_seconds > 0.0)) {
      throw std::invalid_argument("invalid Spaun configuration");
    }
    build_circuit();
  }

  void set_learned_digit_checkpoint(const std::filesystem::path& path) {
    if (path.empty()) {
      learned_digit_classifier.reset();
      return;
    }
    learned_digit_classifier = std::make_unique<snnbase::spiking_conv::Classifier>(
          snnbase::spiking_conv::ModelConfig{
              .input_rows = 28,
              .input_columns = 28,
              .input_channels = 1,
              .class_count = 10,
              .first_convolution = {12, 5, 2, false},
              .second_convolution = {24, 3, 2, false},
              .neuron_threshold = 0.5F,
              .channel_normalization = false,
              .second_residual = snnbase::spiking_conv::ResidualMerge::none},
          snnbase::spiking_conv::TrainingConfig{
              .epochs = 1,
              .batch_size = 64,
              .time_steps = 8,
              .learning_rate = 0.001F,
              .seed = config.seed,
              .augment = false});
    learned_digit_classifier->load_checkpoint(path);
  }

  [[nodiscard]] std::size_t learned_digit(const char symbol) const {
    std::vector<std::uint8_t> pixels(28 * 28);
    const auto bitmap = glyph(symbol);
    for (std::size_t row = 0; row < 28; ++row) {
      for (std::size_t column = 0; column < 28; ++column) {
        const auto source_row = row * glyph_rows / 28;
        const auto source_column = column * glyph_columns / 28;
        pixels[row * 28 + column] =
            bitmap[source_row][source_column] == '#' ? std::uint8_t{255} : 0;
      }
    }
    return learned_digit_classifier->predict({28, 28, pixels});
  }

  void build_circuit() {
    for (std::size_t module = 0; module < module_count; ++module) {
      auto& population = populations[module];
      population.reserve(config.neurons_per_module);
      for (std::size_t neuron = 0; neuron < config.neurons_per_module; ++neuron) {
        const auto bit = (module * config.neurons_per_module + neuron) %
                         snnbase::SpikeEvent::payload_width();
        const auto id = network.add_neuron(
            {.model = snnbase::NeuronModel::leaky_integrate_and_fire,
             .theta = 0.85F,
             .membrane_leak = 0.72F,
             .output_pattern = snnbase::SpikeEvent::payload_pattern({bit}),
             .refractory_period = 1});
        population.push_back(id);
        neuron_modules.push_back(module);
      }
      for (std::size_t neuron = 1; neuron < population.size(); ++neuron) {
        static_cast<void>(network.add_synapse(
            {.source = population[neuron - 1],
             .target = population[neuron],
             .weight = 0.12,
             .delay = 1}));
      }
    }
    for (const auto& [source_module, target_module] : pathways) {
      const auto& sources = populations[module_index(source_module)];
      const auto& targets = populations[module_index(target_module)];
      for (std::size_t neuron = 0; neuron < config.neurons_per_module; ++neuron) {
        static_cast<void>(network.add_synapse(
            {.source = sources[neuron],
             .target = targets[neuron],
             .weight = 0.42,
             .delay = 1}));
      }
    }
  }

  ProbeFrame step(std::initializer_list<Module> driven, char stimulus,
                  char recognized, std::string phase,
                  const std::string& memory, const std::string& output,
                  const std::string& action, double reward = 0.0,
                  const VisualStep* visual_step = nullptr) {
    std::vector<double> currents(network.neuron_count(), 0.0);
    for (const auto module : driven) {
      for (const auto neuron : populations[module_index(module)]) {
        currents[neuron] += 1.05;
      }
    }
    const auto emitted = network.step(currents);
    ProbeFrame frame;
    frame.tick = static_cast<std::size_t>(network.time() - 1);
    frame.time_seconds = static_cast<double>(frame.tick) * config.tick_seconds;
    frame.stimulus = stimulus;
    frame.recognized = recognized;
    frame.phase = std::move(phase);
    frame.working_memory = memory;
    frame.output = output;
    frame.selected_action = action;
    frame.reward = reward;
    frame.arm = arm;
    frame.pen_trace = pen_trace;
    for (const auto& spike : emitted) {
      const auto module = neuron_modules.at(spike.neuron);
      ++frame.spikes[module];
      ++total_spikes;
    }
    if (visual_step != nullptr) {
      const auto retinal = visual_step->modules[
          static_cast<std::size_t>(VisualModule::retina)].spikes;
      const auto visual = visual_step->modules[
          static_cast<std::size_t>(VisualModule::visual)].spikes;
      const auto encoding = visual_step->modules[
          static_cast<std::size_t>(VisualModule::encoding)].spikes;
      frame.spikes[module_index(Module::visual)] += retinal + visual;
      frame.spikes[module_index(Module::encoding)] += encoding;
      total_spikes += retinal + visual + encoding;
    }
    for (std::size_t module = 0; module < module_count; ++module) {
      double membrane = 0.0;
      for (const auto neuron : populations[module]) {
        membrane += std::max(0.0, network.neuron_state(neuron).membrane_potential);
      }
      const auto rate = static_cast<double>(frame.spikes[module]) /
                        static_cast<double>(config.neurons_per_module);
      frame.activity[module] =
          std::clamp(0.65 * rate + 0.35 * membrane /
                                      static_cast<double>(config.neurons_per_module),
                     0.0, 1.0);
    }
    return frame;
  }

  std::string present_stream(const Trial& trial, Result& result) {
    std::string observed;
    std::string memory;
    bool inside_group = false;
    for (const auto symbol : trial.stimulus_stream) {
      char recognized = ' ';
      const auto capture = [&](const VisualStep& visual_step,
                               const char displayed,
                               const std::string_view phase) {
        const auto can_use_learned_digit =
            learned_digit_classifier != nullptr && inside_group &&
            displayed >= '0' && displayed <= '9';
        if ((visual_step.recognition.has_value() || can_use_learned_digit) &&
            recognized == ' ') {
          recognized = can_use_learned_digit
                           ? static_cast<char>('0' + learned_digit(displayed))
                           : *visual_step.recognition;
          observed.push_back(recognized);
          if (recognized == '[') {
            inside_group = true;
          } else if (recognized == ']') {
            inside_group = false;
            semantic_workspace.accept(workspace::group_boundary_event());
          } else if (observed.size() == 2 && observed.front() == 'A' &&
                     recognized >= '0' && recognized <= '7') {
            semantic_workspace.accept(workspace::task_event(
                static_cast<std::size_t>(recognized - '0')));
          } else if (inside_group && recognized >= '0' &&
                     recognized <= '9') {
            semantic_workspace.accept(workspace::digit_event(
                static_cast<std::size_t>(recognized - '0')));
          } else if (inside_group &&
                     (recognized == 'P' || recognized == 'K')) {
            semantic_workspace.accept(workspace::query_event(recognized));
          }
          if (recognized >= '0' && recognized <= '9' && observed.size() > 2) {
            memory.push_back(recognized);
          }
        }
        result.frames.push_back(step(
            phase == "visual encoding"
                ? std::initializer_list<Module>{Module::visual,
                                                Module::encoding}
                : std::initializer_list<Module>{Module::working_memory},
            displayed, recognized, std::string(phase), memory, {},
            phase == "visual encoding" ? "route visual input"
                                        : "maintain context",
            0.0, &visual_step));
      };
      for (std::size_t tick = 0; tick < config.stimulus_ticks; ++tick) {
        capture(visual_frontend.step(symbol), symbol, "visual encoding");
      }
      for (std::size_t tick = 0; tick < config.blank_ticks; ++tick) {
        capture(visual_frontend.step_blank(), ' ', "inter-stimulus blank");
      }
      for (std::size_t drain = 0;
           drain < 2 && recognized == ' '; ++drain) {
        capture(visual_frontend.step_blank(), ' ', "visual pipeline drain");
      }
    }
    return observed;
  }

  Trial route_from_visual_stream(const Trial& source,
                                 const std::string& observed) const {
    if (observed.size() < 3 || observed.front() != 'A' ||
        observed[1] < '0' || observed[1] > '7') {
      throw std::runtime_error(
          "Spaun could not decode a valid visual task cue; encoded stream='" +
          observed + "'");
    }
    Trial routed = source;
    const auto neural_task = semantic_workspace.task_state();
    if (!neural_task.has_value()) {
      throw std::runtime_error(
          "Spaun task state was not sustained by its recurrent population");
    }
    routed.task = static_cast<Task>(*neural_task);
    routed.groups.clear();
    routed.query_kind = semantic_workspace.query_state().value_or(char{});
    routed.query_value = 0;
    for (std::size_t group = 0;
         group < semantic_workspace.maximum_groups(); ++group) {
      if (semantic_workspace.stored_item_count(group) == 0) {
        continue;
      }
      std::vector<int> recalled;
      const auto item_count = semantic_workspace.stored_item_count(group);
      recalled.reserve(item_count);
      for (std::size_t item = 0; item < item_count; ++item) {
        recalled.push_back(
            semantic_workspace.recall_digit(group, item).value_or(-1));
      }
      routed.groups.push_back(recalled);
    }
    if ((routed.query_kind == 'P' || routed.query_kind == 'K') &&
        routed.groups.size() >= 2 && !routed.groups.back().empty()) {
      routed.query_value = routed.groups.back().front();
    }
    return routed;
  }

  std::vector<int> solve_bandit(const Trial& trial, Result& result) {
    constexpr std::array<std::array<double, 3>, 3> published_probabilities{{
        {{0.12, 0.12, 0.72}},
        {{0.12, 0.72, 0.12}},
        {{0.72, 0.12, 0.12}},
    }};
    auto probabilities = published_probabilities;
    if (!trial.reward_probability_blocks.empty()) {
      if (trial.reward_probability_blocks.size() != probabilities.size()) {
        throw std::invalid_argument(
            "A2 requires exactly three reward-probability blocks");
      }
      for (std::size_t block = 0; block < probabilities.size(); ++block) {
        probabilities[block] = trial.reward_probability_blocks[block];
        for (const auto probability : probabilities[block]) {
          if (!std::isfinite(probability) || probability < 0.0 ||
              probability > 1.0) {
            throw std::invalid_argument(
                "A2 reward probabilities must lie in [0, 1]");
          }
        }
      }
    }
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (std::size_t trial_index = 0; trial_index < 60; ++trial_index) {
      const auto block = trial_index / 20;
      const auto& block_probabilities = probabilities[block];
      const auto best_arm = static_cast<std::size_t>(
          std::max_element(block_probabilities.begin(),
                           block_probabilities.end()) -
          block_probabilities.begin());
      const auto decision = reward_selector.select();
      const auto choice = decision.action;
      const auto rewarded = unit(generator) < probabilities[block][choice];
      const double reward = rewarded ? 1.0 : 0.0;
      reward_selector.apply_reward(reward);
      auto frame = step({Module::reward, Module::action_selection}, ' ', ' ',
                        "reward learning", {}, {},
                        "choose arm " + std::to_string(choice) +
                            "; best arm " + std::to_string(best_arm),
                        reward);
      std::copy(reward_selector.utility_weights().begin(),
                reward_selector.utility_weights().end(),
                frame.action_utilities.begin());
      result.frames.push_back(std::move(frame));
    }
    const auto utilities = reward_selector.utility_weights();
    return {static_cast<int>(
        std::max_element(utilities.begin(), utilities.end()) -
        utilities.begin())};
  }

  std::vector<int> solve(const Trial& trial,
                         const std::vector<std::vector<int>>& groups,
                         Result& result) {
    switch (trial.task) {
      case Task::copy_drawing:
      case Task::image_recognition:
      case Task::serial_working_memory:
        return groups.empty() ? std::vector<int>{} : groups.front();
      case Task::reinforcement_learning:
        return solve_bandit(trial, result);
      case Task::counting:
        if (groups.size() < 2 || groups[0].empty() || groups[1].empty() ||
            groups[0].front() < 0 || groups[1].front() < 0) {
          return {};
        }
        if (!semantic_workspace.begin_counting_from_memory(0, 0)) {
          return {};
        }
        for (std::size_t increment = 0;
             increment < static_cast<std::size_t>(groups[1].front());
             ++increment) {
          for (std::size_t tick = 0; tick < config.counting_delay_ticks;
               ++tick) {
            semantic_workspace.advance();
            result.frames.push_back(step(
                {Module::working_memory, Module::transformation,
                 Module::action_selection},
                ' ', ' ', "counting successor", digits_string(groups[0]), {},
                "advance neural accumulator " + std::to_string(increment + 1U)));
          }
        }
        if (const auto counted = semantic_workspace.route_counting_result(
                static_cast<std::size_t>(groups[1].front()));
            counted.has_value()) {
          return number_digits(*counted);
        }
        return {};
      case Task::question_answering: {
        if (groups.empty() || trial.query_value < 0) {
          return {};
        }
        const auto& list = groups.front();
        if (trial.query_kind == 'P') {
          const auto position = trial.query_value > 0
                                    ? static_cast<std::size_t>(trial.query_value - 1)
                                    : list.size();
          return position < list.size() ? std::vector<int>{list[position]}
                                        : std::vector<int>{};
        }
        const auto found = std::find(list.begin(), list.end(), trial.query_value);
        return found == list.end()
                   ? std::vector<int>{}
                   : std::vector<int>{static_cast<int>(found - list.begin() + 1)};
      }
      case Task::rapid_variable_creation: {
        if (groups.size() < 3) {
          return {};
        }
        std::size_t prefix = 0;
        bool initialized = false;
        for (std::size_t index = 0; index + 1 < groups.size() - 1; index += 2) {
          const auto& input = groups[index];
          const auto& output = groups[index + 1];
          if (output.size() > input.size()) {
            return {};
          }
          for (std::size_t suffix = 0; suffix < output.size(); ++suffix) {
            const auto output_value = output[output.size() - 1U - suffix];
            const auto input_value = input[input.size() - 1U - suffix];
            if (output_value >= 0 && input_value >= 0 &&
                output_value != input_value) {
              return {};
            }
          }
          const auto candidate = input.size() - output.size();
          if (initialized && candidate != prefix) {
            return {};
          }
          prefix = candidate;
          initialized = true;
        }
        const auto& query = groups.back();
        return prefix <= query.size()
                   ? std::vector<int>(query.begin() + static_cast<std::ptrdiff_t>(prefix),
                                      query.end())
                   : std::vector<int>{};
      }
      case Task::fluid_reasoning: {
        if (groups.size() < 2 || groups.back().empty()) {
          return {};
        }
        std::size_t target_length = groups.back().size() + 1;
        if (groups.size() >= 6) {
          target_length = groups[2].size();
        }
        return std::vector<int>(target_length, groups.back().back());
      }
    }
    return {};
  }

  DigitStrokes motor_strokes(int digit) const {
    constexpr double drawing_bottom = 0.36;
    constexpr double glyph_width = 0.50;
    constexpr double glyph_height = 0.72;

    auto strokes = digit_strokes(digit);
    for (auto& stroke : strokes) {
      for (auto& point : stroke) {
        point[0] = (point[0] - 0.5) * glyph_width;
        point[1] = drawing_bottom + point[1] * glyph_height;
      }
    }
    return strokes;
  }

  std::array<double, 2> inverse_kinematics(double target_x,
                                           double target_y) const {
    constexpr double link = 0.72;
    const auto radius_squared = std::clamp(
        target_x * target_x + target_y * target_y, 0.02,
        4.0 * link * link - 1e-4);
    const auto elbow_target =
        std::acos(clamp_unit((radius_squared - 2.0 * link * link) /
                             (2.0 * link * link)));
    const auto shoulder_target =
        std::atan2(target_y, target_x) -
        std::atan2(link * std::sin(elbow_target),
                   link + link * std::cos(elbow_target));
    return {shoulder_target, elbow_target};
  }

  void update_arm_coordinates() {
    constexpr double link = 0.72;
    arm.elbow_x = link * std::cos(arm.shoulder);
    arm.elbow_y = link * std::sin(arm.shoulder);
    arm.pen_x = arm.elbow_x + link * std::cos(arm.shoulder + arm.elbow);
    arm.pen_y = arm.elbow_y + link * std::sin(arm.shoulder + arm.elbow);
  }

  void set_arm_pose(double target_x, double target_y) {
    const auto target = inverse_kinematics(target_x, target_y);
    arm.shoulder = target[0];
    arm.elbow = target[1];
    arm.shoulder_velocity = 0.0;
    arm.elbow_velocity = 0.0;
    arm.pen_down = false;
    update_arm_coordinates();
  }

  void advance_arm(double target_x, double target_y, bool pen_down) {
    const auto target = inverse_kinematics(target_x, target_y);
    constexpr double stiffness = 225.0;
    constexpr double damping = 30.0;
    const auto shoulder_acceleration =
        stiffness * (target[0] - arm.shoulder) -
        damping * arm.shoulder_velocity;
    const auto elbow_acceleration =
        stiffness * (target[1] - arm.elbow) - damping * arm.elbow_velocity;
    arm.shoulder_velocity += shoulder_acceleration * config.tick_seconds;
    arm.elbow_velocity += elbow_acceleration * config.tick_seconds;
    arm.shoulder += arm.shoulder_velocity * config.tick_seconds;
    arm.elbow += arm.elbow_velocity * config.tick_seconds;
    update_arm_coordinates();
    const auto stroke_start = pen_down && !arm.pen_down;
    arm.pen_down = pen_down;
    if (pen_down) {
      const auto should_record = [&] {
        if (stroke_start || pen_trace.empty()) {
          return true;
        }
        const auto delta_x = arm.pen_x - pen_trace.back().x;
        const auto delta_y = arm.pen_y - pen_trace.back().y;
        return std::hypot(delta_x, delta_y) >= 0.004;
      }();
      if (should_record) {
        pen_trace.push_back({arm.pen_x, arm.pen_y, stroke_start});
      }
    }
  }

  void append_motor_frame(Result& result, std::string_view phase,
                          std::string_view output_text,
                          std::string action) {
    result.frames.push_back(step(
        {Module::decoding, Module::motor, Module::arm}, ' ', ' ',
        std::string(phase), std::string(output_text), std::string(output_text),
        std::move(action)));
  }

  void position_pen(const StrokePoint& target, int digit, Result& result,
                    std::string_view output_text) {
    const auto minimum_ticks = config.motor_ticks_per_target;
    const auto settling_ticks = static_cast<std::size_t>(
        std::ceil(0.8 / config.tick_seconds));
    const auto maximum_ticks = std::max(minimum_ticks, settling_ticks);
    for (std::size_t tick = 0; tick < maximum_ticks; ++tick) {
      advance_arm(target[0], target[1], false);
      append_motor_frame(result, "pen positioning", output_text,
                         "position pen for digit " + std::to_string(digit));
      const auto position_error =
          std::hypot(target[0] - arm.pen_x, target[1] - arm.pen_y);
      const auto joint_speed =
          std::hypot(arm.shoulder_velocity, arm.elbow_velocity);
      if (tick + 1 >= minimum_ticks && position_error < 0.012 &&
          joint_speed < 0.10) {
        break;
      }
    }
  }

  void trace_stroke(const Stroke& stroke, int digit, Result& result,
                    std::string_view output_text) {
    if (stroke.empty()) {
      return;
    }
    position_pen(stroke.front(), digit, result, output_text);
    for (std::size_t tick = 0; tick < config.motor_ticks_per_target; ++tick) {
      advance_arm(stroke.front()[0], stroke.front()[1], true);
      append_motor_frame(result, "motor execution", output_text,
                         "draw digit " + std::to_string(digit));
    }

    const auto target_spacing = std::clamp(
        0.012 * static_cast<double>(config.motor_ticks_per_target), 0.012,
        0.08);
    for (std::size_t index = 1; index < stroke.size(); ++index) {
      const auto& start = stroke[index - 1];
      const auto& end = stroke[index];
      const auto distance = std::hypot(end[0] - start[0], end[1] - start[1]);
      const auto subdivisions = std::max<std::size_t>(
          1, static_cast<std::size_t>(std::ceil(distance / target_spacing)));
      for (std::size_t subdivision = 1; subdivision <= subdivisions;
           ++subdivision) {
        const auto fraction = static_cast<double>(subdivision) /
                              static_cast<double>(subdivisions);
        const auto target_x = start[0] + (end[0] - start[0]) * fraction;
        const auto target_y = start[1] + (end[1] - start[1]) * fraction;
        for (std::size_t tick = 0; tick < config.motor_ticks_per_target;
             ++tick) {
          advance_arm(target_x, target_y, true);
          append_motor_frame(result, "motor execution", output_text,
                             "draw digit " + std::to_string(digit));
        }
      }
    }

    const auto endpoint_ticks = static_cast<std::size_t>(
        std::ceil(0.4 / config.tick_seconds));
    for (std::size_t tick = 0; tick < endpoint_ticks; ++tick) {
      advance_arm(stroke.back()[0], stroke.back()[1], true);
      append_motor_frame(result, "motor execution", output_text,
                         "complete digit " + std::to_string(digit));
      const auto position_error =
          std::hypot(stroke.back()[0] - arm.pen_x,
                     stroke.back()[1] - arm.pen_y);
      const auto joint_speed =
          std::hypot(arm.shoulder_velocity, arm.elbow_velocity);
      if (position_error < 0.010 && joint_speed < 0.08) {
        break;
      }
    }

    for (std::size_t tick = 0; tick < config.motor_ticks_per_target; ++tick) {
      advance_arm(stroke.back()[0], stroke.back()[1], false);
      append_motor_frame(result, "pen lift", output_text,
                         "finish stroke for digit " + std::to_string(digit));
    }
  }

  void display_completed_digit(int digit, std::size_t digit_index,
                               std::size_t digit_count, Result& result,
                               std::string_view output_text) {
    const auto display_ticks = std::max<std::size_t>(
        config.motor_ticks_per_target,
        static_cast<std::size_t>(std::ceil(0.35 / config.tick_seconds)));
    const auto action =
        "display digit " + std::to_string(digit) + " (" +
        std::to_string(digit_index + 1) + "/" +
        std::to_string(digit_count) + ")";
    for (std::size_t tick = 0; tick < display_ticks; ++tick) {
      advance_arm(arm.pen_x, arm.pen_y, false);
      append_motor_frame(result, "digit display", output_text, action);
    }
  }

  void erase_completed_digit(int digit, int next_digit, Result& result,
                             std::string_view output_text) {
    pen_trace.clear();
    const auto erase_ticks = std::max<std::size_t>(
        config.motor_ticks_per_target,
        static_cast<std::size_t>(std::ceil(0.15 / config.tick_seconds)));
    const auto action = "erase digit " + std::to_string(digit) +
                        " before digit " + std::to_string(next_digit);
    for (std::size_t tick = 0; tick < erase_ticks; ++tick) {
      advance_arm(0.0, 0.90, false);
      append_motor_frame(result, "surface erase", output_text, action);
    }
  }

  void draw_output(const std::vector<int>& output, Result& result) {
    if (!motor_to_arm_enabled) {
      return;
    }
    const auto output_text = digits_string(output);
    for (std::size_t digit_index = 0; digit_index < output.size();
         ++digit_index) {
      const auto digit = output[digit_index];
      if (digit < 0 || digit > 9) {
        continue;
      }
      for (const auto& stroke : motor_strokes(digit)) {
        trace_stroke(stroke, digit, result, output_text);
      }
      display_completed_digit(digit, digit_index, output.size(), result,
                              output_text);
      if (digit_index + 1 < output.size()) {
        erase_completed_digit(digit, output[digit_index + 1], result,
                              output_text);
      }
    }
  }

  Result run(const Trial& trial) {
    reset_runtime();
    Result result;
    result.stimulus_stream = trial.stimulus_stream;
    result.expected = trial.expected;
    result.network_neurons = network.neuron_count() +
                             visual_frontend.neuron_count() +
                             semantic_workspace.neuron_count() +
                             reward_selector.network().neuron_count();
    result.network_synapses = network.synapse_count() +
                              visual_frontend.synapse_count() +
                              semantic_workspace.synapse_count() +
                              reward_selector.network().synapse_count();
    const auto observed = present_stream(trial, result);
    const auto routed = route_from_visual_stream(trial, observed);
    result.task = routed.task;
    result.task_name = std::string(task_name(routed.task));
    const auto& recalled = routed.groups;
    const auto memory_text = recalled.empty() ? std::string{} : digits_string(recalled.front());
    for (std::size_t tick = 0; tick < config.stimulus_ticks; ++tick) {
      result.frames.push_back(step(
          {Module::working_memory, Module::transformation,
           Module::action_selection},
          ' ', ' ', "cognitive transformation", memory_text, {},
          "select " + std::string(task_name(routed.task))));
    }
    result.output.clear();
    const auto route_recalled = routed.task == Task::copy_drawing ||
                                routed.task == Task::image_recognition ||
                                routed.task == Task::serial_working_memory;
    const auto route_position_query =
        routed.task == Task::question_answering && routed.query_kind == 'P' &&
        routed.query_value > 0;
    if (route_recalled) {
      const auto item_count = semantic_workspace.stored_item_count(0);
      result.output.reserve(item_count);
      for (std::size_t item = 0; item < item_count; ++item) {
        result.output.push_back(
            semantic_workspace.route_recalled_digit(0, item).value_or(-1));
      }
    } else if (route_position_query) {
      result.output.push_back(
          semantic_workspace
              .route_recalled_digit(
                  0, static_cast<std::size_t>(routed.query_value - 1))
              .value_or(-1));
    } else {
      const auto planned_output = solve(routed, recalled, result);
      result.output.reserve(planned_output.size());
      for (const auto digit : planned_output) {
        if (digit < 0 || digit > 9) {
          result.output.push_back(-1);
          continue;
        }
        result.output.push_back(
            semantic_workspace
                .route_planned_digit(workspace::vocabulary().get(
                    workspace::digit_name(static_cast<std::size_t>(digit))))
                .value_or(-1));
      }
    }
    for (std::size_t tick = 0; tick < config.stimulus_ticks; ++tick) {
      result.frames.push_back(step(
          {Module::action_selection, Module::decoding}, ' ', ' ',
          "response decoding", memory_text, digits_string(result.output),
          "release motor plan"));
    }
    draw_output(result.output, result);
    const auto produced_observable_ink = std::any_of(
        result.frames.begin(), result.frames.end(), [](const auto& frame) {
          return !frame.pen_trace.empty();
        });
    result.correct = result.output == result.expected &&
                     (result.expected.empty() || produced_observable_ink);
    result.total_spikes = total_spikes + semantic_workspace.total_spikes() +
                          reward_selector.network().total_spikes();
    result.simulated_seconds =
        static_cast<double>(network.time()) * config.tick_seconds;
    return result;
  }

  void reset_runtime() {
    network.reset();
    visual_frontend.reset();
    semantic_workspace.reset();
    generator.seed(config.seed);
    reward_selector.reset_state();
    reward_selector.reset_learning();
    total_spikes = 0;
    arm = {};
    pen_trace.clear();
    set_arm_pose(0.0, 0.90);
  }

  void set_pathway_enabled(const ModelPathway pathway, const bool enabled) {
    switch (pathway) {
      case ModelPathway::retina_to_visual:
        visual_frontend.set_projection_enabled(
            VisualPathway::retina_to_visual, enabled);
        break;
      case ModelPathway::visual_to_encoding:
        visual_frontend.set_projection_enabled(
            VisualPathway::visual_to_encoding, enabled);
        break;
      case ModelPathway::encoding_to_working_memory:
        semantic_workspace.set_pathway_enabled(
            workspace::Pathway::encoding_to_working_memory, enabled);
        break;
      case ModelPathway::working_memory_recurrence:
        semantic_workspace.set_pathway_enabled(
            workspace::Pathway::working_memory_recurrence, enabled);
        break;
      case ModelPathway::working_memory_to_transformation:
        semantic_workspace.set_pathway_enabled(
            workspace::Pathway::working_memory_to_transformation, enabled);
        break;
      case ModelPathway::reward_plasticity:
        reward_selector.set_learning_enabled(enabled);
        break;
      case ModelPathway::transformation_action_to_decoding:
        semantic_workspace.set_pathway_enabled(
            workspace::Pathway::transformation_action_to_decoding, enabled);
        break;
      case ModelPathway::decoding_to_motor:
        semantic_workspace.set_pathway_enabled(
            workspace::Pathway::decoding_to_motor, enabled);
        break;
      case ModelPathway::motor_to_arm:
        motor_to_arm_enabled = enabled;
        break;
    }
  }

  bool pathway_enabled(const ModelPathway pathway) const {
    switch (pathway) {
      case ModelPathway::retina_to_visual:
        return visual_frontend.projection_enabled(
            VisualPathway::retina_to_visual);
      case ModelPathway::visual_to_encoding:
        return visual_frontend.projection_enabled(
            VisualPathway::visual_to_encoding);
      case ModelPathway::encoding_to_working_memory:
        return semantic_workspace.pathway_enabled(
            workspace::Pathway::encoding_to_working_memory);
      case ModelPathway::working_memory_recurrence:
        return semantic_workspace.pathway_enabled(
            workspace::Pathway::working_memory_recurrence);
      case ModelPathway::working_memory_to_transformation:
        return semantic_workspace.pathway_enabled(
            workspace::Pathway::working_memory_to_transformation);
      case ModelPathway::reward_plasticity:
        return reward_selector.learning_enabled();
      case ModelPathway::transformation_action_to_decoding:
        return semantic_workspace.pathway_enabled(
            workspace::Pathway::transformation_action_to_decoding);
      case ModelPathway::decoding_to_motor:
        return semantic_workspace.pathway_enabled(
            workspace::Pathway::decoding_to_motor);
      case ModelPathway::motor_to_arm:
        return motor_to_arm_enabled;
    }
    return false;
  }

  Config config;
  VisualFrontend visual_frontend;
  std::unique_ptr<snnbase::spiking_conv::Classifier> learned_digit_classifier;
  workspace::Workspace semantic_workspace;
  snnbase::Network network;
  std::array<std::vector<snnbase::NeuronId>, module_count> populations;
  std::vector<std::size_t> neuron_modules;
  std::mt19937 generator;
  snnbase::action::RewardModulatedSelector reward_selector;
  std::size_t total_spikes{};
  ArmState arm;
  std::vector<PenPoint> pen_trace;
  bool motor_to_arm_enabled{true};
};

Model::Model(Config config) : impl_(std::make_unique<Impl>(std::move(config))) {}
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

void Model::set_learned_digit_checkpoint(const std::filesystem::path& path) {
  impl_->set_learned_digit_checkpoint(path);
}

Result Model::run(const Trial& trial) {
  return impl_->run(trial);
}

void Model::reset() {
  impl_->reset_runtime();
}

void Model::set_pathway_enabled(const ModelPathway pathway,
                                const bool enabled) {
  impl_->set_pathway_enabled(pathway, enabled);
}

bool Model::pathway_enabled(const ModelPathway pathway) const {
  return impl_->pathway_enabled(pathway);
}

const Config& Model::config() const noexcept {
  return impl_->config;
}

std::size_t Model::neuron_count() const noexcept {
  return impl_->network.neuron_count() + impl_->visual_frontend.neuron_count() +
         impl_->semantic_workspace.neuron_count() +
         impl_->reward_selector.network().neuron_count();
}

std::size_t Model::synapse_count() const noexcept {
  return impl_->network.synapse_count() + impl_->visual_frontend.synapse_count() +
         impl_->semantic_workspace.synapse_count() +
         impl_->reward_selector.network().synapse_count();
}

}  // namespace snnbase_experiments::spaun
