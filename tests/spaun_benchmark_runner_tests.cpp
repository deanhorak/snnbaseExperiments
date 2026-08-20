#include <snnbase_experiments/spaun_benchmark_runner.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace snnbase_experiments::spaun;
using namespace snnbase_experiments::spaun::benchmark;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_visual_streams_hold_all_model_operands() {
  const auto recall = make_trial_schedule(Task::serial_working_memory, 19);
  const auto recall_trial = make_visual_trial(recall.trials.front());
  require(recall_trial.stimulus_stream.starts_with("A3[") &&
              recall_trial.stimulus_stream.ends_with("]?") &&
              recall_trial.expected == recall.trials.front().digits,
          "A3 stream or scoring answer mismatch");
  require(has_clean_model_metadata(recall_trial) && recall_trial.groups.empty() &&
              recall_trial.query_kind == char{} && recall_trial.query_value == 0,
          "A3 operands leaked through Trial metadata");
  require(recall_trial.task == Task::image_recognition,
          "visual task-routing sentinel was not installed");

  const auto counting = make_trial_schedule(Task::counting, 23);
  const auto counting_trial = make_visual_trial(counting.trials.front());
  const auto& counting_plan = counting.trials.front();
  require(counting_trial.stimulus_stream ==
              "A4[" + std::to_string(counting_plan.digits[0]) + "][" +
                  std::to_string(counting_plan.digits[1]) + "]?",
          "A4 start/count were not encoded into the visual stream");
  const auto expected_sum =
      counting_plan.digits[0] + counting_plan.digits[1];
  require(digits_string(counting_trial.expected) ==
              std::to_string(expected_sum),
          "A4 scoring oracle did not compute the sum");
  require(has_clean_model_metadata(counting_trial),
          "A4 operands leaked through metadata");

  const auto questions = make_trial_schedule(Task::question_answering, 29);
  const auto p_plan = std::find_if(
      questions.trials.begin(), questions.trials.end(),
      [](const auto& trial) { return trial.query_kind == 'P'; });
  const auto k_plan = std::find_if(
      questions.trials.begin(), questions.trials.end(),
      [](const auto& trial) { return trial.query_kind == 'K'; });
  require(p_plan != questions.trials.end() && k_plan != questions.trials.end(),
          "A5 schedule lacks a query form");
  const auto p_trial = make_visual_trial(*p_plan);
  const auto k_trial = make_visual_trial(*k_plan);
  require(p_trial.stimulus_stream.find("[P][") != std::string::npos &&
              k_trial.stimulus_stream.find("[K][") != std::string::npos,
          "A5 query metadata was not converted to glyph input");
  require(has_clean_model_metadata(p_trial) && has_clean_model_metadata(k_trial),
          "A5 query leaked through Trial metadata");
  require(p_trial.expected ==
              std::vector<int>{p_plan->digits.at(
                  static_cast<std::size_t>(p_plan->query_value - 1))},
          "A5 P scoring answer mismatch");
  const auto k_position = std::find(k_plan->digits.begin(), k_plan->digits.end(),
                                    k_plan->query_value);
  require(k_trial.expected ==
              std::vector<int>{static_cast<int>(
                  k_position - k_plan->digits.begin() + 1)},
          "A5 K scoring answer mismatch");

  auto contaminated = p_trial;
  contaminated.groups = {{9}};
  require(!has_clean_model_metadata(contaminated),
          "metadata contamination was not detected");
}

void test_bandit_stream_and_published_oracle() {
  const auto bandit = make_trial_schedule(Task::reinforcement_learning, 31);
  require(bandit.trials.size() == 60,
          "A2 schedule does not contain 60 external trials");
  constexpr std::array<int, 3> expected_best{2, 1, 0};
  for (std::size_t block = 0; block < 3; ++block) {
    const auto& planned = bandit.trials.at(block * 20U);
    const auto trial = make_visual_trial(planned);
    require(trial.stimulus_stream == "A2?" && has_clean_model_metadata(trial),
            "A2 did not use a clean task-cue stream");
    require(expected_answer(planned) == std::vector<int>{expected_best[block]},
            "A2 published best arm scoring changed");
  }
}

void test_unresolved_oracles_throw() {
  bool a6_threw = false;
  try {
    PlannedTrial planned{.task = Task::rapid_variable_creation};
    static_cast<void>(make_visual_trial(planned));
  } catch (const std::logic_error&) {
    a6_threw = true;
  }
  require(a6_threw, "A6 invented a scoring schedule");

  bool a7_threw = false;
  try {
    PlannedTrial planned{.task = Task::fluid_reasoning};
    static_cast<void>(expected_answer(planned));
  } catch (const std::logic_error&) {
    a7_threw = true;
  }
  require(a7_threw, "A7 inferred answers from an aggregate result");
}

void test_registration_and_evidence_scope_guards() {
  const std::array<Task, 3> selected{Task::image_recognition,
                                     Task::rapid_variable_creation,
                                     Task::fluid_reasoning};
  RunnerOptions options;
  options.quick = true;
  options.bootstrap_samples = 100;
  auto unavailable_report = run_behavioral_benchmarks(selected, options);
  require(unavailable_report.tasks[0].status ==
              TaskRunStatus::unavailable_requires_registration,
          "unregistered A1 was executed as if it were held out");
  require(unavailable_report.tasks[1].status ==
              TaskRunStatus::unresolved_stimulus_schedule &&
              unavailable_report.tasks[2].status ==
                  TaskRunStatus::unresolved_stimulus_schedule,
          "unresolved A6/A7 schedules were not explicit");
  require(unavailable_report.evidence.scope ==
              EvidenceScope::canonical_smoke_test &&
              !unavailable_report.equivalence.full_equivalence_supported,
          "incomplete evidence was promoted to equivalence");

  options.registered_a1 = RegisteredRecognitionRun{
      .correct = 94,
      .total = 100,
      .dataset_id = "mnist-test-sha256:example",
      .split_id = "official-test-v1",
      .classifier_artifact_id = "checkpoint-sha256:example",
      .held_out = true};
  const auto registered_report = run_behavioral_benchmarks(selected, options);
  require(registered_report.tasks[0].status == TaskRunStatus::completed &&
              registered_report.tasks[0].metrics.size() == 1U &&
              registered_report.tasks[0].metrics[0].summary.estimate == 0.94,
          "registered A1 result was not captured");
  require(registered_report.evidence.scope ==
              EvidenceScope::canonical_smoke_test &&
              !registered_report.equivalence.behavioral_equivalence_supported,
          "registered A1 plus unresolved tasks was promoted to a full protocol run");

  std::ostringstream json;
  write_report_json(json, registered_report);
  require(json.str().find("snnbase-spaun-behavioral-benchmark-v1") !=
                  std::string::npos &&
              json.str().find("\"full_equivalence_supported\":false") !=
                  std::string::npos,
          "JSON report lost its evidence-scope guard");
}

void test_neural_counting_latency_matches_the_registered_a4_metric() {
  const std::array<Task, 1> selected{Task::counting};
  RunnerOptions options;
  options.quick = true;
  options.bootstrap_samples = 100;
  const auto report = run_behavioral_benchmarks(selected, options);
  require(report.tasks.size() == 1U &&
              report.tasks.front().executed_trials == 5U,
          "quick A4 did not execute one complete simulated participant");
  const auto metric = std::find_if(
      report.tasks.front().metrics.begin(), report.tasks.front().metrics.end(),
      [](const auto& value) {
        return value.metric_id == "response_time_slope_ms_per_item";
      });
  require(metric != report.tasks.front().metrics.end() &&
              std::abs(metric->summary.estimate - 420.0) < 1.0e-9,
          "A4 neural successor delay did not produce 420 ms/item");
  const auto assessment = std::find_if(
      report.equivalence.behavioral_score.assessments.begin(),
      report.equivalence.behavioral_score.assessments.end(),
      [](const auto& value) { return value.gate.task == Task::counting; });
  require(assessment != report.equivalence.behavioral_score.assessments.end() &&
              assessment->status == AssessmentStatus::passed,
          "A4 publication-band project gate did not pass");
}

}  // namespace

int main() {
  try {
    test_visual_streams_hold_all_model_operands();
    test_bandit_stream_and_published_oracle();
    test_unresolved_oracles_throw();
    test_registration_and_evidence_scope_guards();
    test_neural_counting_latency_matches_the_registered_a4_metric();
    std::cout << "spaun_benchmark_runner_tests: passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "spaun_benchmark_runner_tests: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
