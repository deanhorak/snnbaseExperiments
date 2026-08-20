#include <snnbase_experiments/spaun_benchmark_runner.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace snnbase_experiments::spaun::benchmark {
namespace {

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

std::string bracketed(std::span<const int> digits) {
  return "[" + digits_string(digits) + "]";
}

Config effective_config(const RunnerOptions& options, std::uint32_t seed) {
  auto config = options.model_config;
  config.seed = seed;
  if (options.quick) {
    config.neurons_per_module =
        std::min<std::size_t>(config.neurons_per_module, 8U);
    config.stimulus_ticks = 1;
    config.blank_ticks = 1;
    config.motor_ticks_per_target = 1;
  }
  return config;
}

double response_onset(const Result& result) {
  const ProbeFrame* last_blank = nullptr;
  const ProbeFrame* first_motor = nullptr;
  for (const auto& frame : result.frames) {
    if (frame.phase == "inter-stimulus blank") {
      last_blank = &frame;
    }
    if (first_motor == nullptr && frame.phase == "pen positioning") {
      first_motor = &frame;
    }
  }
  if (last_blank == nullptr || first_motor == nullptr ||
      first_motor->time_seconds < last_blank->time_seconds) {
    return 0.0;
  }
  return first_motor->time_seconds - last_blank->time_seconds;
}

MetricSummary binary_summary(std::span<const std::uint8_t> values,
                             const RunnerOptions& options,
                             std::uint32_t salt) {
  return bootstrap_proportion(values, options.bootstrap_samples,
                              options.seed ^ salt);
}

MetricRecord binary_metric(std::string id,
                           std::span<const std::uint8_t> values,
                           const RunnerOptions& options, std::uint32_t salt,
                           std::string provenance) {
  return {.metric_id = std::move(id),
          .summary = binary_summary(values, options, salt),
          .unit = "proportion",
          .provenance = std::move(provenance)};
}

MetricRecord mean_metric(std::string id, std::span<const double> values,
                         const RunnerOptions& options, std::uint32_t salt,
                         std::string unit, std::string provenance) {
  return {.metric_id = std::move(id),
          .summary = bootstrap_mean(values, options.bootstrap_samples,
                                    options.seed ^ salt),
          .unit = std::move(unit),
          .provenance = std::move(provenance)};
}

TrialRecord record_trial(const PlannedTrial& planned, const Trial& trial,
                         const Result& result) {
  return {.schedule_index = planned.trial_index,
          .simulated_participant = planned.simulated_participant,
          .condition_index = planned.condition_index,
          .trial_seed = planned.trial_seed,
          .visual_stream = trial.stimulus_stream,
          .expected = trial.expected,
          .output = result.output,
          // Result::correct is the behavioral score: for a nonempty expected
          // answer it already requires both decoded equality and visible ink.
          .correct = result.correct && result.output == trial.expected,
          .response_onset_seconds = response_onset(result),
          .spikes = result.total_spikes,
          .simulated_seconds = result.simulated_seconds};
}

std::vector<PlannedTrial> selected_trials(const TrialSchedule& schedule,
                                         bool quick) {
  if (!quick) {
    return schedule.trials;
  }
  std::vector<PlannedTrial> selected;
  switch (schedule.task) {
    case Task::serial_working_memory: {
      std::array<std::size_t, 4> counts{};
      for (const auto& trial : schedule.trials) {
        if (trial.condition_index < counts.size() &&
            counts[trial.condition_index] < 2U) {
          selected.push_back(trial);
          ++counts[trial.condition_index];
        }
      }
      break;
    }
    case Task::counting:
    case Task::question_answering:
      std::copy_if(schedule.trials.begin(), schedule.trials.end(),
                   std::back_inserter(selected), [](const auto& trial) {
                     return trial.simulated_participant == 0U;
                   });
      break;
    default:
      selected = schedule.trials;
      break;
  }
  return selected;
}

int parsed_choice(std::string_view action) {
  constexpr std::string_view prefix = "choose arm ";
  const auto begin = action.find(prefix);
  if (begin == std::string_view::npos || begin + prefix.size() >= action.size()) {
    return -1;
  }
  const auto symbol = action[begin + prefix.size()];
  return symbol >= '0' && symbol <= '2' ? symbol - '0' : -1;
}

TaskBenchmarkResult unavailable(Task task, const RunnerOptions& options,
                                TaskRunStatus status, std::string caveat) {
  const auto schedule = make_trial_schedule(task, options.seed);
  TaskBenchmarkResult result{
      .task = task,
      .status = status,
      .evidence_scope = EvidenceScope::canonical_smoke_test,
      .quick = options.quick,
      .planned_trials = schedule.trials.size(),
      .schedule_fully_specified_by_publication =
          schedule.fully_specified_by_publication,
      .schedule_provenance = schedule.provenance_note};
  result.caveats.push_back(std::move(caveat));
  return result;
}

TaskBenchmarkResult run_a1(const RunnerOptions& options) {
  if (!options.registered_a1.has_value() ||
      !options.registered_a1->valid()) {
    return unavailable(
        Task::image_recognition, options,
        TaskRunStatus::unavailable_requires_registration,
        "A1 is unavailable until a held-out result is registered with dataset, "
        "split, and classifier-artifact identifiers; canonical glyph recall is "
        "not a substitute for the published handwritten-image benchmark.");
  }

  const auto& registration = *options.registered_a1;
  std::vector<std::uint8_t> outcomes(registration.total, 0U);
  std::fill_n(outcomes.begin(),
              static_cast<std::ptrdiff_t>(registration.correct), 1U);
  const auto provenance =
      "registered held-out classifier result; dataset=" +
      registration.dataset_id + "; split=" + registration.split_id +
      "; artifact=" + registration.classifier_artifact_id;
  TaskBenchmarkResult result{
      .task = Task::image_recognition,
      .status = TaskRunStatus::completed,
      .evidence_scope = options.quick ? EvidenceScope::canonical_smoke_test
                                      : EvidenceScope::published_protocol_run,
      .quick = options.quick,
      .planned_trials = registration.total,
      .executed_trials = registration.total,
      .schedule_fully_specified_by_publication = false,
      .schedule_provenance = provenance};
  result.metrics.push_back(binary_metric("recognition_accuracy", outcomes,
                                         options, 0xA100U, provenance));
  result.observations.push_back(
      {.task = Task::image_recognition,
       .metric_id = "recognition_accuracy",
       .numeric = result.metrics.back().summary,
       .provenance = provenance});
  return result;
}

TaskBenchmarkResult run_a2(const RunnerOptions& options) {
  const auto schedule =
      make_trial_schedule(Task::reinforcement_learning, options.seed);
  TaskBenchmarkResult benchmark{
      .task = Task::reinforcement_learning,
      .status = TaskRunStatus::completed,
      .evidence_scope = options.quick ? EvidenceScope::canonical_smoke_test
                                      : EvidenceScope::published_protocol_run,
      .quick = options.quick,
      .planned_trials = schedule.trials.size(),
      .schedule_fully_specified_by_publication =
          schedule.fully_specified_by_publication,
      .schedule_provenance = schedule.provenance_note};

  // Model performs all 60 choices inside one call. The runner injects the three
  // published contingency blocks through the explicit environment field and
  // independently scores the emitted choices against its schedule. Reward
  // probabilities are an environment boundary, not an answer oracle or a
  // hidden visual operand.
  auto trial = make_visual_trial(schedule.trials.front());
  trial.expected.clear();
  trial.reward_probability_blocks = {
      schedule.trials.at(0).reward_probabilities,
      schedule.trials.at(20).reward_probabilities,
      schedule.trials.at(40).reward_probabilities};
  Model model(effective_config(options, options.seed));
  const auto result = model.run(trial);
  benchmark.executed_trials = 0;
  constexpr std::array<int, 3> published_best{2, 1, 0};
  std::array<std::vector<std::uint8_t>, 3> block_outcomes;
  std::vector<std::uint8_t> reward_outcomes;
  std::vector<std::uint8_t> rolling;
  rolling.reserve(5);
  for (const auto& frame : result.frames) {
    if (frame.phase != "reward learning") {
      continue;
    }
    const auto choice = parsed_choice(frame.selected_action);
    const auto index = benchmark.bandit_choices.size();
    if (index >= schedule.trials.size() || choice < 0) {
      continue;
    }
    const auto block = index / 20U;
    const auto selected_best = choice == published_best[block];
    block_outcomes[block].push_back(selected_best ? 1U : 0U);
    reward_outcomes.push_back(frame.reward > 0.0 ? 1U : 0U);
    rolling.push_back(selected_best ? 1U : 0U);
    if (rolling.size() > 5U) {
      rolling.erase(rolling.begin());
    }
    const auto moving = static_cast<double>(
                            std::accumulate(rolling.begin(), rolling.end(), 0U)) /
                        static_cast<double>(rolling.size());
    benchmark.bandit_choices.push_back(
        {.trial_index = index,
         .block = block,
         .choice = choice,
         .published_best_arm = published_best[block],
         .rewarded = frame.reward > 0.0,
         .moving_best_arm_choice_probability = moving});
  }
  benchmark.executed_trials = benchmark.bandit_choices.size();
  for (std::size_t block = 0; block < block_outcomes.size(); ++block) {
    if (!block_outcomes[block].empty()) {
      benchmark.metrics.push_back(binary_metric(
          "published_best_arm_choice_rate_block_" + std::to_string(block + 1U),
          block_outcomes[block], options,
          static_cast<std::uint32_t>(0xA200U + block),
          "choices parsed from Model reward-learning frames and scored against "
          "the published best-arm order 3,2,1"));
      const auto late_begin = block_outcomes[block].size() > 5U
                                  ? block_outcomes[block].end() - 5
                                  : block_outcomes[block].begin();
      std::vector<std::uint8_t> late(late_begin, block_outcomes[block].end());
      benchmark.metrics.push_back(binary_metric(
          "late_best_arm_choice_rate_block_" + std::to_string(block + 1U),
          late, options, static_cast<std::uint32_t>(0xA210U + block),
          "project diagnostic over the final five choices; not a published "
          "acceptance threshold"));
    }
  }
  if (!reward_outcomes.empty()) {
    benchmark.metrics.push_back(binary_metric(
        "observed_reward_rate", reward_outcomes, options, 0xA220U,
        "reward samples exposed by the current embedded Model environment"));
  }
  bool adapted = block_outcomes.size() == 3U;
  for (const auto& outcomes : block_outcomes) {
    if (outcomes.size() < 5U) {
      adapted = false;
      continue;
    }
    const auto successes = std::accumulate(outcomes.end() - 5,
                                           outcomes.end(), 0U);
    adapted = adapted && successes >= 3U;
  }
  benchmark.observations.push_back(
      {.task = Task::reinforcement_learning,
       .metric_id = "blockwise_adaptation",
       .qualitative_pass = adapted,
       .provenance =
           "project operationalization: at least three of the final five "
           "choices select the published best arm in every block; the source "
           "specifies a five-trial moving window but no pass threshold"});
  benchmark.caveats.push_back(
      "Model::run advances the complete 60-trial A2 trajectory in one call. "
      "Trial now injects all three contingency blocks, but the public interface "
      "does not yet expose one decision/reward exchange at a time.");
  return benchmark;
}

TaskBenchmarkResult run_a3(const RunnerOptions& options) {
  const auto schedule =
      make_trial_schedule(Task::serial_working_memory, options.seed);
  const auto selected = selected_trials(schedule, options.quick);
  TaskBenchmarkResult benchmark{
      .task = Task::serial_working_memory,
      .status = TaskRunStatus::completed,
      .evidence_scope = options.quick ? EvidenceScope::canonical_smoke_test
                                      : EvidenceScope::published_protocol_run,
      .quick = options.quick,
      .planned_trials = schedule.trials.size(),
      .schedule_fully_specified_by_publication =
          schedule.fully_specified_by_publication,
      .schedule_provenance = schedule.provenance_note};
  std::vector<std::uint8_t> exact;
  std::map<std::pair<std::size_t, std::size_t>, std::vector<std::uint8_t>>
      position_outcomes;
  for (const auto& planned : selected) {
    const auto trial = make_visual_trial(planned);
    Model model(effective_config(options, planned.trial_seed));
    const auto result = model.run(trial);
    benchmark.trials.push_back(record_trial(planned, trial, result));
    exact.push_back(result.output == trial.expected ? 1U : 0U);
    for (std::size_t position = 0; position < planned.digits.size(); ++position) {
      const auto correct = position < result.output.size() &&
                           result.output[position] == trial.expected[position];
      position_outcomes[{planned.digits.size(), position}].push_back(correct ? 1U
                                                                             : 0U);
    }
  }
  benchmark.executed_trials = benchmark.trials.size();
  benchmark.metrics.push_back(binary_metric(
      "exact_sequence_accuracy", exact, options, 0xA300U,
      "expected sequences were retained only by the scoring runner; Model "
      "received A3 glyph streams with empty Trial.groups"));

  std::map<std::pair<std::size_t, std::size_t>, double> profile;
  std::uint32_t salt = 0xA310U;
  for (const auto& [key, outcomes] : position_outcomes) {
    const auto id = "position_accuracy_length_" + std::to_string(key.first) +
                    "_position_" + std::to_string(key.second + 1U);
    benchmark.metrics.push_back(binary_metric(
        id, outcomes, options, salt++,
        "deterministic project digit schedule following the published 4--7 item, "
        "40-run-per-length design"));
    profile[key] = benchmark.metrics.back().summary.estimate;
  }

  bool primacy_and_recency = true;
  for (std::size_t length = 4; length <= 7; ++length) {
    double middle = 0.0;
    for (std::size_t position = 1; position + 1 < length; ++position) {
      middle += profile[{length, position}];
    }
    middle /= static_cast<double>(length - 2U);
    primacy_and_recency =
        primacy_and_recency && profile[{length, 0}] >= middle + 0.02 &&
        profile[{length, length - 1U}] >= middle + 0.02;
  }
  const auto provenance =
      "project operationalization: first and last positions must each exceed "
      "the within-length middle mean by 0.02; this is not a published numeric "
      "threshold; schedule=" + schedule.provenance_note;
  benchmark.observations.push_back(
      {.task = Task::serial_working_memory,
       .metric_id = "serial_position_profile",
       .qualitative_pass = primacy_and_recency,
       .provenance = provenance});
  if (options.quick) {
    benchmark.caveats.push_back(
        "Quick A3 uses two rather than 40 runs per list length and is smoke-test evidence.");
  }
  return benchmark;
}

TaskBenchmarkResult run_a4(const RunnerOptions& options) {
  const auto schedule = make_trial_schedule(Task::counting, options.seed);
  const auto selected = selected_trials(schedule, options.quick);
  TaskBenchmarkResult benchmark{
      .task = Task::counting,
      .status = TaskRunStatus::completed,
      .evidence_scope = options.quick ? EvidenceScope::canonical_smoke_test
                                      : EvidenceScope::published_protocol_run,
      .quick = options.quick,
      .planned_trials = schedule.trials.size(),
      .schedule_fully_specified_by_publication =
          schedule.fully_specified_by_publication,
      .schedule_provenance = schedule.provenance_note};
  std::vector<std::uint8_t> exact;
  std::map<std::size_t, std::vector<std::pair<double, double>>> participant_xy;
  for (const auto& planned : selected) {
    const auto trial = make_visual_trial(planned);
    Model model(effective_config(options, planned.trial_seed));
    const auto result = model.run(trial);
    auto record = record_trial(planned, trial, result);
    exact.push_back(record.correct ? 1U : 0U);
    participant_xy[planned.simulated_participant].push_back(
        {static_cast<double>(planned.digits.at(1)),
         record.response_onset_seconds * 1000.0});
    benchmark.trials.push_back(std::move(record));
  }
  benchmark.executed_trials = benchmark.trials.size();
  benchmark.metrics.push_back(binary_metric(
      "exact_answer_accuracy", exact, options, 0xA400U,
      "sum expected by the scorer and operands supplied to Model only through "
      "A4[start][count]? glyph streams"));

  std::vector<double> slopes;
  for (const auto& [participant, values] : participant_xy) {
    static_cast<void>(participant);
    std::vector<double> x;
    std::vector<double> y;
    for (const auto& [count, latency] : values) {
      x.push_back(count);
      y.push_back(latency);
    }
    if (x.size() >= 2U) {
      slopes.push_back(ordinary_least_squares_slope(x, y));
    }
  }
  if (!slopes.empty()) {
    benchmark.metrics.push_back(mean_metric(
        "response_time_slope_ms_per_item", slopes, options, 0xA410U,
        "milliseconds/item",
        "response onset from the final query-symbol blank to the first "
        "pen-positioning frame; each item traverses one delayed spiking "
        "semantic-successor stage"));
    benchmark.observations.push_back(
        {.task = Task::counting,
         .metric_id = "response_time_slope_ms_per_item",
         .numeric = benchmark.metrics.back().summary,
         .provenance = benchmark.metrics.back().provenance});
  }
  benchmark.caveats.push_back(
      "The project operationalizes response onset as the interval from the end "
      "of the final query blank to initial pen positioning. This landmark and "
      "the generated starting digits are recorded so the comparison to the "
      "published 419 +/- 10 ms/item slope remains auditable.");
  if (options.quick) {
    benchmark.caveats.push_back(
        "Quick A4 uses one rather than five simulated participants.");
  }
  return benchmark;
}

TaskBenchmarkResult run_a5(const RunnerOptions& options) {
  const auto schedule =
      make_trial_schedule(Task::question_answering, options.seed);
  const auto selected = selected_trials(schedule, options.quick);
  TaskBenchmarkResult benchmark{
      .task = Task::question_answering,
      .status = TaskRunStatus::completed,
      .evidence_scope = options.quick ? EvidenceScope::canonical_smoke_test
                                      : EvidenceScope::published_protocol_run,
      .quick = options.quick,
      .planned_trials = schedule.trials.size(),
      .schedule_fully_specified_by_publication =
          schedule.fully_specified_by_publication,
      .schedule_provenance = schedule.provenance_note};
  std::vector<std::uint8_t> exact;
  std::map<std::pair<char, std::size_t>, std::vector<std::uint8_t>> profiles;
  for (const auto& planned : selected) {
    const auto trial = make_visual_trial(planned);
    Model model(effective_config(options, planned.trial_seed));
    const auto result = model.run(trial);
    auto record = record_trial(planned, trial, result);
    exact.push_back(record.correct ? 1U : 0U);
    profiles[{planned.query_kind, planned.condition_index}].push_back(
        record.correct ? 1U : 0U);
    benchmark.trials.push_back(std::move(record));
  }
  benchmark.executed_trials = benchmark.trials.size();
  benchmark.metrics.push_back(binary_metric(
      "exact_answer_accuracy", exact, options, 0xA500U,
      "answer oracle held by the runner; list, P/K query kind, and query value "
      "were all presented as glyphs with empty Trial query metadata"));

  std::map<std::pair<char, std::size_t>, double> estimates;
  std::uint32_t salt = 0xA510U;
  for (const auto& [key, outcomes] : profiles) {
    const auto id = std::string("position_accuracy_") + key.first + "_position_" +
                    std::to_string(key.second + 1U);
    benchmark.metrics.push_back(binary_metric(
        id, outcomes, options, salt++,
        "published 10 participant by 7 position by 2 query-form design; digit "
        "lists come from the recorded project generator"));
    estimates[key] = benchmark.metrics.back().summary.estimate;
  }

  double mean_difference = 0.0;
  std::array<double, 7> combined{};
  for (std::size_t position = 0; position < 7; ++position) {
    const auto p = estimates[{'P', position}];
    const auto k = estimates[{'K', position}];
    mean_difference += std::abs(p - k) / 7.0;
    combined[position] = (p + k) / 2.0;
  }
  const auto middle =
      std::accumulate(combined.begin() + 1, combined.end() - 1, 0.0) / 5.0;
  const auto profile_pass = mean_difference <= 0.10 &&
                            combined.front() >= middle + 0.02 &&
                            combined.back() >= middle + 0.02;
  const auto provenance =
      "project operationalization: mean P/K profile difference <= 0.10 and "
      "both endpoints exceed the middle-position mean by 0.02; not a published "
      "numeric threshold; schedule=" + schedule.provenance_note;
  benchmark.observations.push_back(
      {.task = Task::question_answering,
       .metric_id = "question_answering_profile",
       .qualitative_pass = profile_pass,
       .provenance = provenance});
  if (options.quick) {
    benchmark.caveats.push_back(
        "Quick A5 uses one rather than ten simulated participants.");
  }
  return benchmark;
}

TaskBenchmarkResult run_task(Task task, const RunnerOptions& options) {
  switch (task) {
    case Task::copy_drawing:
      return unavailable(
          task, options, TaskRunStatus::unavailable_requires_registration,
          "A0 requires registered held-out handwritten images and blinded "
          "recognizability/style review; the built-in 5x7 glyph is only a smoke stimulus.");
    case Task::image_recognition: return run_a1(options);
    case Task::reinforcement_learning: return run_a2(options);
    case Task::serial_working_memory: return run_a3(options);
    case Task::counting: return run_a4(options);
    case Task::question_answering: return run_a5(options);
    case Task::rapid_variable_creation:
      return unavailable(
          task, options, TaskRunStatus::unresolved_stimulus_schedule,
          "A6 needs a registered source/target transformation corpus; published "
          "examples do not define an aggregate scoring schedule.");
    case Task::fluid_reasoning:
      return unavailable(
          task, options, TaskRunStatus::unresolved_stimulus_schedule,
          "A7 has a published run count but no registered item schedule in this "
          "project; generating answers from the reported aggregate would leak the oracle.");
  }
  throw std::invalid_argument("invalid benchmark task");
}

std::string json_escape(std::string_view value) {
  std::string result;
  for (const auto character : value) {
    switch (character) {
      case '"': result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default: result += character; break;
    }
  }
  return result;
}

void write_digits_json(std::ostream& output, std::span<const int> values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    output << (index == 0U ? "" : ",") << values[index];
  }
  output << ']';
}

void write_metric_json(std::ostream& output, const MetricRecord& metric) {
  output << "{\"id\":\"" << json_escape(metric.metric_id)
         << "\",\"estimate\":" << metric.summary.estimate
         << ",\"sample_size\":" << metric.summary.sample_size
         << ",\"bootstrap_samples\":" << metric.summary.bootstrap_samples;
  if (metric.summary.interval.has_value()) {
    output << ",\"interval\":{\"lower\":"
           << metric.summary.interval->lower << ",\"upper\":"
           << metric.summary.interval->upper << ",\"confidence_level\":"
           << metric.summary.interval->confidence_level << '}';
  } else {
    output << ",\"interval\":null";
  }
  output << ",\"unit\":\"" << json_escape(metric.unit)
         << "\",\"provenance\":\"" << json_escape(metric.provenance)
         << "\"}";
}

std::string_view assessment_status_name(const AssessmentStatus status) {
  switch (status) {
    case AssessmentStatus::passed: return "passed";
    case AssessmentStatus::failed: return "failed";
    case AssessmentStatus::missing: return "missing";
  }
  return "unknown";
}

std::string_view gate_rule_name(const GateRule rule) {
  switch (rule) {
    case GateRule::minimum: return "minimum";
    case GateRule::reference_band: return "reference_band";
    case GateRule::confidence_interval_overlap:
      return "confidence_interval_overlap";
    case GateRule::qualitative_review: return "qualitative_review";
  }
  return "unknown";
}

}  // namespace

bool RegisteredRecognitionRun::valid() const noexcept {
  return held_out && total > 0U && correct <= total && !dataset_id.empty() &&
         !split_id.empty() && !classifier_artifact_id.empty();
}

Trial make_visual_trial(const PlannedTrial& planned) {
  Trial trial;
  // The deliberately wrong sentinel proves routing is driven by the leading A#
  // glyph cue. Model replaces it after reading stimulus_stream.
  trial.task = Task::image_recognition;
  trial.name = "benchmark visual-stream trial";
  trial.expected = expected_answer(planned);
  trial.reward_probabilities = planned.reward_probabilities;
  switch (planned.task) {
    case Task::copy_drawing:
      if (planned.digits.size() != 1U) {
        throw std::invalid_argument("A0 visual trial requires one digit");
      }
      trial.stimulus_stream = "A0" + bracketed(planned.digits) + "?";
      break;
    case Task::reinforcement_learning:
      trial.stimulus_stream = "A2?";
      break;
    case Task::serial_working_memory:
      if (planned.digits.empty()) {
        throw std::invalid_argument("A3 visual trial requires a list");
      }
      trial.stimulus_stream = "A3" + bracketed(planned.digits) + "?";
      break;
    case Task::counting:
      if (planned.digits.size() != 2U) {
        throw std::invalid_argument("A4 visual trial requires start and count");
      }
      trial.stimulus_stream =
          "A4[" + std::to_string(planned.digits[0]) + "][" +
          std::to_string(planned.digits[1]) + "]?";
      break;
    case Task::question_answering:
      if (planned.digits.size() != 7U ||
          (planned.query_kind != 'P' && planned.query_kind != 'K')) {
        throw std::invalid_argument("A5 visual trial requires a seven-item P/K query");
      }
      trial.stimulus_stream =
          "A5" + bracketed(planned.digits) + "[" + planned.query_kind + "][" +
          std::to_string(planned.query_value) + "]?";
      break;
    case Task::image_recognition:
      throw std::logic_error("A1 requires registered held-out image input");
    case Task::rapid_variable_creation:
      throw std::logic_error("A6 has no registered aggregate schedule");
    case Task::fluid_reasoning:
      throw std::logic_error("A7 has no registered item schedule");
  }
  if (!has_clean_model_metadata(trial)) {
    throw std::logic_error("benchmark trial leaked operands through metadata");
  }
  return trial;
}

std::vector<int> expected_answer(const PlannedTrial& planned) {
  switch (planned.task) {
    case Task::copy_drawing:
    case Task::serial_working_memory:
      return planned.digits;
    case Task::reinforcement_learning: {
      const auto best = std::max_element(planned.reward_probabilities.begin(),
                                         planned.reward_probabilities.end());
      return {static_cast<int>(best - planned.reward_probabilities.begin())};
    }
    case Task::counting:
      if (planned.digits.size() != 2U) {
        throw std::invalid_argument("A4 scoring requires start and count");
      }
      return number_digits(planned.digits[0] + planned.digits[1]);
    case Task::question_answering: {
      if (planned.digits.empty()) {
        return {};
      }
      if (planned.query_kind == 'P') {
        const auto position = planned.query_value > 0
                                  ? static_cast<std::size_t>(planned.query_value - 1)
                                  : planned.digits.size();
        return position < planned.digits.size()
                   ? std::vector<int>{planned.digits[position]}
                   : std::vector<int>{};
      }
      if (planned.query_kind == 'K') {
        const auto found = std::find(planned.digits.begin(), planned.digits.end(),
                                     planned.query_value);
        return found == planned.digits.end()
                   ? std::vector<int>{}
                   : std::vector<int>{
                         static_cast<int>(found - planned.digits.begin() + 1)};
      }
      throw std::invalid_argument("A5 scoring requires P or K query kind");
    }
    case Task::image_recognition:
    case Task::rapid_variable_creation:
    case Task::fluid_reasoning:
      throw std::logic_error("task has no runner-generated scoring oracle");
  }
  return {};
}

bool has_clean_model_metadata(const Trial& trial) noexcept {
  return trial.groups.empty() && trial.query_kind == char{} &&
         trial.query_value == 0;
}

BenchmarkReport run_behavioral_benchmarks(std::span<const Task> tasks,
                                          const RunnerOptions& options) {
  if (tasks.empty()) {
    throw std::invalid_argument("select at least one Spaun benchmark task");
  }
  if (options.bootstrap_samples == 0U) {
    throw std::invalid_argument("bootstrap sample count must be positive");
  }
  std::set<Task> unique;
  BenchmarkReport report{.seed = options.seed,
                         .quick = options.quick,
                         .bootstrap_samples = options.bootstrap_samples};
  for (const auto task : tasks) {
    if (!unique.insert(task).second) {
      throw std::invalid_argument("benchmark task was selected more than once");
    }
    report.tasks.push_back(run_task(task, options));
  }

  bool complete_battery = unique.size() == task_count;
  for (const auto task : all_tasks()) {
    complete_battery = complete_battery && unique.contains(task);
  }
  bool all_protocol = complete_battery && !options.quick;
  for (const auto& task : report.tasks) {
    all_protocol = all_protocol && task.status == TaskRunStatus::completed &&
                   task.evidence_scope == EvidenceScope::published_protocol_run;
    report.evidence.observations.insert(report.evidence.observations.end(),
                                        task.observations.begin(),
                                        task.observations.end());
  }
  report.evidence.scope = all_protocol ? EvidenceScope::published_protocol_run
                                       : EvidenceScope::canonical_smoke_test;
  report.equivalence = assess_equivalence(report.evidence);
  report.caveats.push_back(
      "Expected answers are constructed in the scoring runner after the visual "
      "stream is fixed; Trial.groups/query_kind/query_value are empty for every model run.");
  if (!all_protocol) {
    report.caveats.push_back(
        "This report is not a complete published-protocol battery and cannot "
        "support a Spaun behavioral-equivalence claim.");
  }
  report.caveats.push_back(
      "No causal lesion evidence is generated by this runner, so behavior cannot "
      "yet be attributed to the spiking pathways.");
  return report;
}

std::string_view task_run_status_name(TaskRunStatus status) noexcept {
  switch (status) {
    case TaskRunStatus::completed: return "completed";
    case TaskRunStatus::completed_protocol_deviation:
      return "completed_protocol_deviation";
    case TaskRunStatus::unavailable_requires_registration:
      return "unavailable_requires_registration";
    case TaskRunStatus::unresolved_stimulus_schedule:
      return "unresolved_stimulus_schedule";
  }
  return "unknown";
}

std::string_view evidence_scope_name(EvidenceScope scope) noexcept {
  switch (scope) {
    case EvidenceScope::canonical_smoke_test: return "canonical_smoke_test";
    case EvidenceScope::published_protocol_run: return "published_protocol_run";
  }
  return "unknown";
}

void write_report_json(std::ostream& output, const BenchmarkReport& report) {
  output << std::setprecision(10)
         << "{\n  \"format\":\"snnbase-spaun-behavioral-benchmark-v1\","
         << "\n  \"seed\":" << report.seed << ",\n  \"quick\":"
         << (report.quick ? "true" : "false")
         << ",\n  \"bootstrap_samples\":" << report.bootstrap_samples
         << ",\n  \"evidence_scope\":\""
         << evidence_scope_name(report.evidence.scope) << "\",\n  \"tasks\":[\n";
  for (std::size_t task_index = 0; task_index < report.tasks.size(); ++task_index) {
    const auto& task = report.tasks[task_index];
    output << "    {\"task\":\"A" << static_cast<int>(task.task)
           << "\",\"name\":\"" << json_escape(task_name(task.task))
           << "\",\"status\":\"" << task_run_status_name(task.status)
           << "\",\"evidence_scope\":\""
           << evidence_scope_name(task.evidence_scope)
           << "\",\"planned_trials\":" << task.planned_trials
           << ",\"executed_trials\":" << task.executed_trials
           << ",\"schedule_fully_specified_by_publication\":"
           << (task.schedule_fully_specified_by_publication ? "true" : "false")
           << ",\"schedule_provenance\":\""
           << json_escape(task.schedule_provenance) << "\",\"metrics\":[";
    for (std::size_t index = 0; index < task.metrics.size(); ++index) {
      output << (index == 0U ? "" : ",");
      write_metric_json(output, task.metrics[index]);
    }
    output << "],\"observations\":[";
    for (std::size_t index = 0; index < task.observations.size(); ++index) {
      const auto& observation = task.observations[index];
      output << (index == 0U ? "" : ",") << "{\"metric_id\":\""
             << json_escape(observation.metric_id) << "\",\"numeric\":";
      if (observation.numeric.has_value()) {
        output << "{\"estimate\":" << observation.numeric->estimate
               << ",\"sample_size\":" << observation.numeric->sample_size
               << '}';
      } else {
        output << "null";
      }
      output << ",\"qualitative_pass\":";
      if (observation.qualitative_pass.has_value()) {
        output << (*observation.qualitative_pass ? "true" : "false");
      } else {
        output << "null";
      }
      output << ",\"provenance\":\"" << json_escape(observation.provenance)
             << "\"}";
    }
    output << "],\"trials\":[";
    for (std::size_t index = 0; index < task.trials.size(); ++index) {
      const auto& trial = task.trials[index];
      output << (index == 0U ? "" : ",") << "{\"schedule_index\":"
             << trial.schedule_index << ",\"participant\":"
             << trial.simulated_participant << ",\"condition\":"
             << trial.condition_index << ",\"seed\":" << trial.trial_seed
             << ",\"visual_stream\":\"" << json_escape(trial.visual_stream)
             << "\",\"expected\":";
      write_digits_json(output, trial.expected);
      output << ",\"output\":";
      write_digits_json(output, trial.output);
      output << ",\"correct\":" << (trial.correct ? "true" : "false")
             << ",\"response_onset_seconds\":"
             << trial.response_onset_seconds << ",\"spikes\":"
             << trial.spikes << ",\"simulated_seconds\":"
             << trial.simulated_seconds << '}';
    }
    output << "],\"bandit_choices\":[";
    for (std::size_t index = 0; index < task.bandit_choices.size(); ++index) {
      const auto& choice = task.bandit_choices[index];
      output << (index == 0U ? "" : ",") << "{\"trial\":"
             << choice.trial_index << ",\"block\":" << choice.block
             << ",\"choice\":" << choice.choice
             << ",\"published_best_arm\":" << choice.published_best_arm
             << ",\"rewarded\":" << (choice.rewarded ? "true" : "false")
             << ",\"moving_best_arm_choice_probability\":"
             << choice.moving_best_arm_choice_probability << '}';
    }
    output << "],\"caveats\":[";
    for (std::size_t index = 0; index < task.caveats.size(); ++index) {
      output << (index == 0U ? "" : ",") << '"'
             << json_escape(task.caveats[index]) << '"';
    }
    output << "]}" << (task_index + 1U == report.tasks.size() ? "\n" : ",\n");
  }
  output << "  ],\n  \"gate_score\":{\"passed\":"
         << report.equivalence.behavioral_score.passed << ",\"failed\":"
         << report.equivalence.behavioral_score.failed << ",\"missing\":"
         << report.equivalence.behavioral_score.missing
         << ",\"assessments\":[";
  for (std::size_t index = 0;
       index < report.equivalence.behavioral_score.assessments.size(); ++index) {
    const auto& assessment =
        report.equivalence.behavioral_score.assessments[index];
    output << (index == 0U ? "" : ",") << "{\"task\":\"A"
           << static_cast<int>(assessment.gate.task) << "\",\"metric_id\":\""
           << json_escape(assessment.gate.metric_id) << "\",\"rule\":\""
           << gate_rule_name(assessment.gate.rule) << "\",\"status\":\""
           << assessment_status_name(assessment.status)
           << "\",\"rationale\":\"" << json_escape(assessment.gate.rationale)
           << "\",\"explanation\":\""
           << json_escape(assessment.explanation) << "\"}";
  }
  output << "]},\n  \"behavioral_equivalence_supported\":"
         << (report.equivalence.behavioral_equivalence_supported ? "true" : "false")
         << ",\n  \"spiking_mechanism_supported\":"
         << (report.equivalence.spiking_mechanism_supported ? "true" : "false")
         << ",\n  \"full_equivalence_supported\":"
         << (report.equivalence.full_equivalence_supported ? "true" : "false")
         << ",\n  \"equivalence_caveats\":[";
  for (std::size_t index = 0; index < report.equivalence.caveats.size(); ++index) {
    output << (index == 0U ? "" : ",") << '"'
           << json_escape(report.equivalence.caveats[index]) << '"';
  }
  output << "],\n  \"caveats\":[";
  for (std::size_t index = 0; index < report.caveats.size(); ++index) {
    output << (index == 0U ? "" : ",") << '"'
           << json_escape(report.caveats[index]) << '"';
  }
  output << "]\n}\n";
}

}  // namespace snnbase_experiments::spaun::benchmark
