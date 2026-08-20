#include <snnbase_experiments/spaun.hpp>
#include <snnbase_experiments/spaun_visual.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_vocabulary_and_task_protocol() {
  using namespace snnbase_experiments::spaun;
  require(parse_task("A0") == Task::copy_drawing, "A0 parser mismatch");
  require(parse_task("memory") == Task::serial_working_memory,
          "task short-name parser mismatch");
  require(parse_task("7") == Task::fluid_reasoning,
          "numeric task parser mismatch");
  require(glyph_event('8').count() > 0, "digit glyph emitted no event bits");
  require(glyph_event(' ').empty(), "blank glyph must emit no payload bits");
  require(glyph_event('2').bits() != glyph_event('3').bits(),
          "digit glyphs must have distinct events");
  for (const auto task : all_tasks()) {
    const auto trial = canonical_trial(task);
    require(trial.stimulus_stream.starts_with(
                "A" + std::to_string(static_cast<int>(task))),
            "canonical task cue must enter through the visual stream");
    require(trial.stimulus_stream.ends_with('?'),
            "canonical task must end with a visual question mark");
  }
}

void test_network_structure_and_battery() {
  using namespace snnbase_experiments::spaun;
  constexpr std::size_t neurons_per_module = 4;
  Model model({.neurons_per_module = neurons_per_module,
               .stimulus_ticks = 2,
               .blank_ticks = 1,
               .motor_ticks_per_target = 2,
               .tick_seconds = 0.01,
               .memory_noise_standard_deviation = 0.0,
               .memory_recurrent_weight_gain = 1.1,
               .primacy_recurrent_weight_gain = 1.1,
               .recency_recurrent_weight_gain = 1.1,
               .seed = 42});
  require(model.neuron_count() >= module_count * neurons_per_module +
                                      opponent_retina_neuron_count +
                                      2 * visual_symbol_count,
          "Spaun functional and visual population count mismatch");
  require(model.synapse_count() >= 12 * neurons_per_module +
                                      opponent_retina_neuron_count *
                                          visual_symbol_count +
                                      visual_symbol_count,
          "Spaun functional pathways are incomplete");

  std::array<std::size_t, module_count> module_spikes{};
  for (const auto task : all_tasks()) {
    const auto result = model.run(canonical_trial(task));
    require(result.correct,
            "canonical task failed: A" +
                std::to_string(static_cast<int>(task)));
    require(result.output == result.expected,
            "task output did not match its behavioral contract");
    require(!result.frames.empty(), "task produced no probe frames");
    require(result.total_spikes > 0, "task produced no circuit spikes");
    require(result.simulated_seconds > 0.0,
            "task has no simulated duration");
    require(result.frames.back().output == digits_string(result.output),
            "final GUI snapshot does not expose motor output");
    for (const auto& frame : result.frames) {
      require(std::isfinite(frame.arm.shoulder) &&
                  std::isfinite(frame.arm.elbow) &&
                  std::isfinite(frame.arm.pen_x) &&
                  std::isfinite(frame.arm.pen_y),
              "arm state became non-finite");
      require(std::abs(frame.arm.pen_x) < 2.0 &&
                  std::abs(frame.arm.pen_y) < 2.0,
              "arm left its physical workspace");
      for (std::size_t module = 0; module < module_count; ++module) {
        module_spikes[module] += frame.spikes[module];
      }
    }
  }
  for (std::size_t module = 0; module < module_count; ++module) {
    require(module_spikes[module] > 0,
            "functional module never spiked: " +
                std::string(module_name(static_cast<Module>(module))));
  }
}

void test_question_k_and_multi_digit_count() {
  using namespace snnbase_experiments::spaun;
  Model model({.neurons_per_module = 3,
               .stimulus_ticks = 1,
               .blank_ticks = 1,
               .motor_ticks_per_target = 1,
               .memory_noise_standard_deviation = 0.0,
               .memory_recurrent_weight_gain = 1.1,
               .primacy_recurrent_weight_gain = 1.1,
               .recency_recurrent_weight_gain = 1.1,
               .seed = 9});

  auto question = canonical_trial(Task::question_answering);
  question.name = "position of item five";
  question.stimulus_stream = "A5[015873][K][5]?";
  question.groups = {{0, 1, 5, 8, 7, 3}, {5}};
  question.query_kind = 'K';
  question.query_value = 5;
  question.expected = {3};
  require(model.run(question).correct, "K-style question answering failed");

  auto counting = canonical_trial(Task::counting);
  counting.stimulus_stream = "A4[7][5]?";
  counting.groups = {{7}, {5}};
  counting.expected = {1, 2};
  require(model.run(counting).correct, "multi-digit counting output failed");

  auto visual_routing = canonical_trial(Task::image_recognition);
  visual_routing.task = Task::copy_drawing;
  visual_routing.groups = {{1}};
  const auto routed = model.run(visual_routing);
  require(routed.task == Task::image_recognition && routed.output == std::vector<int>{7},
          "task or operand metadata bypassed the visual stream");
}

void test_deterministic_replay() {
  using namespace snnbase_experiments::spaun;
  const Config config{.neurons_per_module = 3,
                      .stimulus_ticks = 1,
                      .blank_ticks = 1,
                      .motor_ticks_per_target = 1,
                      .seed = 123};
  Model first(config);
  Model second(config);
  const auto first_result =
      first.run(canonical_trial(Task::reinforcement_learning));
  const auto second_result =
      second.run(canonical_trial(Task::reinforcement_learning));
  require(first_result.output == second_result.output,
          "seeded behavioral replay is not deterministic");
  require(first_result.total_spikes == second_result.total_spikes,
          "seeded spike-count replay is not deterministic");
  require(first_result.frames.size() == second_result.frames.size(),
          "seeded probe replay changed length");
  require(first_result.frames.at(10).action_utilities ==
              second_result.frames.at(10).action_utilities,
          "seeded reward utility trace is not deterministic");
}

void test_external_reward_contingencies_drive_a2() {
  using namespace snnbase_experiments::spaun;
  auto trial = canonical_trial(Task::reinforcement_learning);
  trial.reward_probability_blocks = {
      std::array<double, 3>{0.0, 0.0, 1.0},
      std::array<double, 3>{0.0, 0.0, 1.0},
      std::array<double, 3>{0.0, 0.0, 1.0}};
  trial.expected = {2};
  Model model({.neurons_per_module = 3,
               .stimulus_ticks = 1,
               .blank_ticks = 1,
               .motor_ticks_per_target = 1,
               .seed = 42});
  const auto result = model.run(trial);
  require(result.output == std::vector<int>{2} && result.correct,
          "A2 ignored externally supplied reward contingencies");
  const auto reward_frames_are_externally_scoped = std::all_of(
      result.frames.begin(), result.frames.end(), [](const auto& frame) {
        return frame.phase != "reward learning" ||
               frame.selected_action.find("best arm 2") != std::string::npos;
      });
  require(reward_frames_are_externally_scoped,
          "A2 telemetry retained the built-in best-arm order");
}

void test_recognizable_drawing_geometry() {
  using namespace snnbase_experiments::spaun;
  Model model({.neurons_per_module = 3,
               .stimulus_ticks = 1,
               .blank_ticks = 1,
               .motor_ticks_per_target = 5,
               .tick_seconds = 0.01,
               .memory_noise_standard_deviation = 0.0,
               .memory_recurrent_weight_gain = 1.1,
               .primacy_recurrent_weight_gain = 1.1,
               .recency_recurrent_weight_gain = 1.1,
               .seed = 42});

  const auto copy = model.run(canonical_trial(Task::copy_drawing));
  const auto& trace = copy.frames.back().pen_trace;
  require(trace.size() > 40, "copy task produced an undersampled pen trace");

  double minimum_x = trace.front().x;
  double maximum_x = trace.front().x;
  double minimum_y = trace.front().y;
  double maximum_y = trace.front().y;
  std::size_t stroke_count = 0;
  bool top_left = false;
  bool top_right = false;
  bool bottom_left = false;
  bool bottom_right = false;
  for (std::size_t index = 0; index < trace.size(); ++index) {
    const auto& point = trace[index];
    require(std::isfinite(point.x) && std::isfinite(point.y),
            "pen trace contains a non-finite point");
    require(point.x >= -0.75 && point.x <= 0.75 && point.y >= 0.25 &&
                point.y <= 1.20,
            "pen trace left the drawing surface");
    minimum_x = std::min(minimum_x, point.x);
    maximum_x = std::max(maximum_x, point.x);
    minimum_y = std::min(minimum_y, point.y);
    maximum_y = std::max(maximum_y, point.y);
    stroke_count += point.stroke_start ? 1U : 0U;
    top_left = top_left || (point.x < -0.12 && point.y > 0.98);
    top_right = top_right || (point.x > 0.12 && point.y > 0.98);
    bottom_left = bottom_left || (point.x < -0.12 && point.y < 0.44);
    bottom_right = bottom_right || (point.x > 0.12 && point.y < 0.44);
    if (index > 0 && !point.stroke_start) {
      const auto distance = std::hypot(point.x - trace[index - 1].x,
                                       point.y - trace[index - 1].y);
      require(distance < 0.08,
              "continuous ink contains a nonphysical jump");
    }
  }
  require(stroke_count == 1, "digit two should be one continuous stroke");
  require(maximum_x - minimum_x > 0.40 && maximum_y - minimum_y > 0.60,
          "digit two trace collapsed to an unrecognizable extent");
  require(top_left && top_right && bottom_left && bottom_right,
          "digit two trace is missing a defining corner");

  for (int digit = 0; digit <= 9; ++digit) {
    auto trial = canonical_trial(Task::copy_drawing);
    trial.stimulus_stream = "A0[" + std::to_string(digit) + "]?";
    trial.expected = {digit};
    const auto result = model.run(trial);
    require(result.correct, "copy drawing failed for digit " +
                                std::to_string(digit));
    const auto& digit_trace = result.frames.back().pen_trace;
    require(digit_trace.size() > 20,
            "digit stroke is too sparse for digit " +
                std::to_string(digit));
    double digit_minimum_x = digit_trace.front().x;
    double digit_maximum_x = digit_trace.front().x;
    double digit_minimum_y = digit_trace.front().y;
    double digit_maximum_y = digit_trace.front().y;
    std::size_t digit_strokes = 0;
    for (const auto& point : digit_trace) {
      digit_minimum_x = std::min(digit_minimum_x, point.x);
      digit_maximum_x = std::max(digit_maximum_x, point.x);
      digit_minimum_y = std::min(digit_minimum_y, point.y);
      digit_maximum_y = std::max(digit_maximum_y, point.y);
      digit_strokes += point.stroke_start ? 1U : 0U;
    }
    require(digit_strokes >= 1,
            "digit has no explicit stroke boundary: " +
                std::to_string(digit));
    require(digit_maximum_x - digit_minimum_x > 0.20 &&
                digit_maximum_y - digit_minimum_y > 0.55,
            "digit trace collapsed to an unrecognizable shape: " +
                std::to_string(digit));
  }

  const auto multi = model.run(canonical_trial(Task::rapid_variable_creation));
  const auto& multi_trace = multi.frames.back().pen_trace;
  require(!multi_trace.empty(), "multi-digit answer produced no ink");
  const ProbeFrame* first_digit_display = nullptr;
  bool saw_surface_erase = false;
  bool saw_new_stroke_after_erase = false;
  std::size_t erase_transitions = 0;
  std::string_view previous_phase;
  for (const auto& frame : multi.frames) {
    if (frame.phase == "digit display" &&
        frame.selected_action.find("display digit 7 (1/2)") !=
            std::string::npos) {
      first_digit_display = &frame;
    }
    if (frame.phase == "surface erase") {
      if (previous_phase != "surface erase") {
        ++erase_transitions;
      }
      saw_surface_erase = true;
      require(frame.pen_trace.empty(),
              "surface erase frame retained the previous digit's ink");
    } else if (saw_surface_erase && !frame.pen_trace.empty() &&
               !saw_new_stroke_after_erase) {
      saw_new_stroke_after_erase = true;
      require(frame.pen_trace.front().stroke_start,
              "next digit reused the previous digit's open stroke");
      require(frame.pen_trace.size() <= 2,
              "next digit began with stale ink on the surface");
    }
    previous_phase = frame.phase;
  }
  require(first_digit_display != nullptr &&
              !first_digit_display->pen_trace.empty(),
          "first digit was not held visibly before erasing");
  require(erase_transitions == 1 && saw_new_stroke_after_erase,
          "multi-digit output did not erase exactly once between digits");

  double first_minimum_x = first_digit_display->pen_trace.front().x;
  double first_maximum_x = first_minimum_x;
  for (const auto& point : first_digit_display->pen_trace) {
    first_minimum_x = std::min(first_minimum_x, point.x);
    first_maximum_x = std::max(first_maximum_x, point.x);
  }
  require(first_maximum_x - first_minimum_x > 0.40,
          "first sequential digit was not drawn full-size");

  double final_minimum_x = multi_trace.front().x;
  double final_maximum_x = final_minimum_x;
  std::size_t final_strokes = 0;
  for (const auto& point : multi_trace) {
    final_minimum_x = std::min(final_minimum_x, point.x);
    final_maximum_x = std::max(final_maximum_x, point.x);
    final_strokes += point.stroke_start ? 1U : 0U;
  }
  require(final_strokes == 1,
          "final surface retained strokes from an earlier digit");
  require(final_maximum_x - final_minimum_x > 0.40 &&
              final_maximum_x - final_minimum_x < 0.65,
          "final sequential digit was not centered at full size");
}

void test_invalid_configuration() {
  using namespace snnbase_experiments::spaun;
  bool threw = false;
  try {
    Model invalid({.neurons_per_module = 0});
    static_cast<void>(invalid);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "zero-neuron Spaun configuration was accepted");
}

void test_end_to_end_causal_lesions() {
  using namespace snnbase_experiments::spaun;
  const Config config{.neurons_per_module = 3,
                      .stimulus_ticks = 1,
                      .blank_ticks = 1,
                      .motor_ticks_per_target = 1,
                      .tick_seconds = 0.01,
                      .seed = 42};

  Model visual_lesion(config);
  visual_lesion.set_pathway_enabled(ModelPathway::visual_to_encoding,
                                    false);
  require(!visual_lesion.pathway_enabled(
              ModelPathway::visual_to_encoding),
          "visual lesion state was not applied");
  bool failed_without_visual_encoding = false;
  try {
    static_cast<void>(
        visual_lesion.run(canonical_trial(Task::image_recognition)));
  } catch (const std::runtime_error&) {
    failed_without_visual_encoding = true;
  }
  require(failed_without_visual_encoding,
          "behavior survived the visual-to-encoding lesion");

  Model memory_lesion(config);
  memory_lesion.set_pathway_enabled(
      ModelPathway::working_memory_recurrence, false);
  bool failed_without_memory = false;
  try {
    static_cast<void>(
        memory_lesion.run(canonical_trial(Task::serial_working_memory)));
  } catch (const std::runtime_error&) {
    failed_without_memory = true;
  }
  require(failed_without_memory,
          "serial recall survived loss of recurrent working memory");

  Model transform_lesion(config);
  transform_lesion.set_pathway_enabled(
      ModelPathway::working_memory_to_transformation, false);
  const auto transformed =
      transform_lesion.run(canonical_trial(Task::image_recognition));
  const auto has_valid_digit = [](const Result& result) {
    return std::any_of(result.output.begin(), result.output.end(),
                       [](const int digit) { return digit >= 0 && digit <= 9; });
  };
  require(!has_valid_digit(transformed) && !transformed.correct,
          "motor answer survived the transformation lesion");

  Model motor_lesion(config);
  motor_lesion.set_pathway_enabled(ModelPathway::decoding_to_motor, false);
  const auto motor =
      motor_lesion.run(canonical_trial(Task::image_recognition));
  require(!has_valid_digit(motor) && !motor.correct,
          "answer survived the decoding-to-motor lesion");

  Model arm_lesion(config);
  arm_lesion.set_pathway_enabled(ModelPathway::motor_to_arm, false);
  const auto arm = arm_lesion.run(canonical_trial(Task::image_recognition));
  require(arm.output == arm.expected,
          "motor decoding was unexpectedly destroyed by a plant lesion");
  require(!arm.correct,
          "a decoded answer without observable arm ink was scored correct");
  require(arm.frames.back().pen_trace.empty(),
          "arm produced ink after its motor pathway was lesioned");
}

}  // namespace

int main() {
  try {
    test_vocabulary_and_task_protocol();
    test_network_structure_and_battery();
    test_question_k_and_multi_digit_count();
    test_deterministic_replay();
    test_external_reward_contingencies_drive_a2();
    test_recognizable_drawing_geometry();
    test_end_to_end_causal_lesions();
    test_invalid_configuration();
    std::cout << "spaun_tests: passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "spaun_tests: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
