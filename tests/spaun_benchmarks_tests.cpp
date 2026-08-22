#include <snnbase_experiments/spaun_benchmarks.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace snnbase_experiments::spaun;
using namespace snnbase_experiments::spaun::benchmark;

constexpr std::array<Task, task_count> tasks{
    Task::copy_drawing,
    Task::image_recognition,
    Task::reinforcement_learning,
    Task::serial_working_memory,
    Task::counting,
    Task::question_answering,
    Task::rapid_variable_creation,
    Task::fluid_reasoning,
};

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool near(double first, double second, double tolerance = 1.0e-12) {
  return std::abs(first - second) <= tolerance;
}

const PublishedNumericReference& reference(
    const std::vector<PublishedNumericReference>& references, Task task,
    std::string_view metric_id) {
  const auto found = std::find_if(
      references.begin(), references.end(), [&](const auto& candidate) {
        return candidate.task == task && candidate.metric_id == metric_id;
      });
  if (found == references.end()) {
    throw std::runtime_error("published reference is absent");
  }
  return *found;
}

void test_published_facts_are_separate_from_project_gates() {
  const auto shared = shared_published_protocol();
  require(shared.image_rows == 28 && shared.image_columns == 28,
          "published visual dimensions changed");
  require(near(shared.stimulus_seconds, 0.150) &&
              near(shared.blank_seconds, 0.150),
          "published presentation timing changed");
  require(shared.arm_is_only_behavioral_output,
          "published behavioral output boundary changed");

  const auto protocols = published_task_protocols();
  require(protocols.size() == task_count,
          "there must be one published protocol record per task");
  for (const auto task : tasks) {
    require(std::count_if(protocols.begin(), protocols.end(),
                          [task](const auto& protocol) {
                            return protocol.task == task;
                          }) == 1,
            "task protocol is absent or duplicated");
  }

  const auto references = published_numeric_references();
  require(near(reference(references, Task::image_recognition,
                         "recognition_accuracy")
                   .estimate,
               0.94),
          "A1 published reference changed");
  const auto& counting = reference(references, Task::counting,
                                   "response_time_slope_ms_per_item");
  require(near(counting.estimate, 419.0) &&
              counting.reported_plus_minus.has_value() &&
              near(*counting.reported_plus_minus, 10.0),
          "A4 published slope changed");
  const auto& raw =
      reference(references, Task::fluid_reasoning, "raw_accuracy");
  require(near(raw.estimate, 0.75) && raw.interval.has_value() &&
              near(raw.interval->lower, 0.60) &&
              near(raw.interval->upper, 0.88),
          "A7 raw result changed");
  require(near(reference(references, Task::fluid_reasoning,
                         "chance_adjusted_accuracy")
                   .estimate,
               0.88),
          "A7 adjusted result changed");
  require(near(reference(references, Task::fluid_reasoning, "human_accuracy")
                   .estimate,
               0.89),
          "A7 human comparator changed");

  const auto gates = project_acceptance_gates();
  require(gates.size() == task_count,
          "project must define an explicit gate for every task");
  require(std::none_of(gates.begin(), gates.end(), [](const auto& gate) {
            return gate.rule == GateRule::minimum && gate.lower.has_value() &&
                   near(*gate.lower, 0.95);
          }),
          "a universal 95 percent gate was invented");
  const auto a1_gate = std::find_if(gates.begin(), gates.end(), [](const auto& gate) {
    return gate.task == Task::image_recognition;
  });
  require(a1_gate != gates.end() && a1_gate->rule == GateRule::minimum &&
              a1_gate->lower.has_value() && near(*a1_gate->lower, 0.94),
          "A1 project gate must remain distinct and traceable to the 0.94 reference");
}

void test_deterministic_protocol_schedules() {
  const auto bandit = make_trial_schedule(Task::reinforcement_learning, 17);
  require(bandit.fully_specified_by_publication && bandit.trials.size() == 60,
          "A2 published schedule mismatch");
  require(bandit.trials[0].reward_probabilities ==
              std::array<double, 3>{0.12, 0.12, 0.72} &&
              bandit.trials[19].reward_probabilities ==
                  std::array<double, 3>{0.12, 0.12, 0.72} &&
              bandit.trials[20].reward_probabilities ==
                  std::array<double, 3>{0.12, 0.72, 0.12} &&
              bandit.trials[40].reward_probabilities ==
                  std::array<double, 3>{0.72, 0.12, 0.12},
          "A2 block contingencies changed");

  const auto recall = make_trial_schedule(Task::serial_working_memory, 91);
  const auto replay = make_trial_schedule(Task::serial_working_memory, 91);
  require(recall.trials.size() == 160 && !recall.fully_specified_by_publication,
          "A3 schedule dimensions mismatch");
  require(recall.trials.front().digits == replay.trials.front().digits &&
              recall.trials.back().trial_seed == replay.trials.back().trial_seed,
          "project trial generator is not deterministic");
  for (std::size_t condition = 0; condition < 4; ++condition) {
    const auto begin = condition * 40U;
    const auto expected_length = condition + 4U;
    require(recall.trials[begin].digits.size() == expected_length &&
                recall.trials[begin + 39U].digits.size() == expected_length,
            "A3 list-length block mismatch");
  }

  require(make_trial_schedule(Task::copy_drawing, 4).trials.size() == 20,
          "A0 example count mismatch");
  require(make_trial_schedule(Task::image_recognition, 4).trials.empty(),
          "A1 generator invented an unstated held-out split");
  require(make_trial_schedule(Task::counting, 4).trials.size() == 25,
          "A4 design size mismatch");
  require(make_trial_schedule(Task::question_answering, 4).trials.size() == 140,
          "A5 design size mismatch");
  require(make_trial_schedule(Task::rapid_variable_creation, 4).trials.empty(),
          "A6 generator invented an aggregate corpus");
  const auto reasoning = make_trial_schedule(Task::fluid_reasoning, 4);
  require(reasoning.trials.size() == 40 &&
              !reasoning.fully_specified_by_publication,
          "A7 run skeleton mismatch");
}

void test_statistics() {
  const std::array<std::uint8_t, 4> outcomes{1, 1, 1, 0};
  const auto proportion = empirical_proportion(outcomes);
  require(near(proportion.estimate, 0.75) && proportion.sample_size == 4 &&
              !proportion.interval.has_value(),
          "empirical proportion mismatch");

  const auto interval = bootstrap_proportion(outcomes, 3000, 123);
  const auto replay = bootstrap_proportion(outcomes, 3000, 123);
  require(interval.bootstrap_samples == published_bootstrap_samples &&
              interval.interval.has_value() && replay.interval.has_value() &&
              near(interval.interval->lower, replay.interval->lower) &&
              near(interval.interval->upper, replay.interval->upper) &&
              interval.interval->lower <= interval.estimate &&
              interval.interval->upper >= interval.estimate,
          "deterministic 3000-sample proportion bootstrap failed");

  const std::array<double, 5> values{1.0, 2.0, 3.0, 4.0, 5.0};
  const auto average = empirical_mean(values);
  const auto mean_interval = bootstrap_mean(values, 3000, 5);
  require(near(average.estimate, 3.0) && mean_interval.interval.has_value() &&
              mean_interval.interval->lower <= 3.0 &&
              mean_interval.interval->upper >= 3.0,
          "mean statistics mismatch");

  const std::array<double, 5> response_times{1000.0, 1419.0, 1838.0,
                                             2257.0, 2676.0};
  require(near(ordinary_least_squares_slope(values, response_times), 419.0),
          "A4 slope calculation mismatch");

  bool threw = false;
  try {
    const std::array<std::uint8_t, 2> invalid{0, 2};
    static_cast<void>(empirical_proportion(invalid));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "invalid binary outcome was accepted");
}

std::vector<Observation> passing_observations() {
  const auto qualitative = [](Task task, std::string metric) {
    return Observation{.task = task,
                       .metric_id = std::move(metric),
                       .qualitative_pass = true,
                       .provenance = "blinded registered review"};
  };
  return {
      qualitative(Task::copy_drawing, "recognizable_copy"),
      {.task = Task::image_recognition,
       .metric_id = "recognition_accuracy",
       .numeric = MetricSummary{.estimate = 0.94, .sample_size = 1000},
       .provenance = "held-out registered split"},
      qualitative(Task::reinforcement_learning, "blockwise_adaptation"),
      qualitative(Task::serial_working_memory, "serial_position_profile"),
      {.task = Task::counting,
       .metric_id = "response_time_slope_ms_per_item",
       .numeric = MetricSummary{
           .estimate = 419.0,
           .sample_size = 25,
           .interval = ConfidenceInterval{.lower = 405.0, .upper = 433.0},
           .bootstrap_samples = 3000},
       .provenance = "registered response-time analysis"},
      qualitative(Task::question_answering, "question_answering_profile"),
      qualitative(Task::rapid_variable_creation,
                  "variable_rule_generalization"),
      {.task = Task::fluid_reasoning,
       .metric_id = "raw_accuracy",
       .numeric = MetricSummary{
           .estimate = 0.75,
           .sample_size = 40,
           .interval = ConfidenceInterval{.lower = 0.60, .upper = 0.88},
           .bootstrap_samples = 3000},
       .provenance = "40 registered reasoning trials"},
  };
}

std::vector<CausalAblationResult> passing_ablations() {
  std::vector<CausalAblationResult> results;
  for (const auto task : tasks) {
    results.push_back(
        {.ablation_id = "lesion-A" +
                        std::to_string(static_cast<unsigned int>(task)),
         .task = task,
         .pathway = static_cast<CausalPathway>(
             static_cast<unsigned int>(task) % 7U),
         .intact = MetricSummary{.estimate = 0.80, .sample_size = 40},
         .ablated = MetricSummary{.estimate = 0.50, .sample_size = 40},
         .higher_is_better = true,
         .project_minimum_absolute_effect = 0.10,
         .completed = true,
         .notes = "same registered inputs and seed schedule"});
  }
  return results;
}

void test_gate_scoring_and_equivalence_claim_guard() {
  auto observations = passing_observations();
  const auto score = score_observations(observations);
  require(score.passed == task_count && score.failed == 0 && score.missing == 0 &&
              score.all_required_passed(),
          "passing task evidence did not pass all project gates");

  auto failed = observations;
  failed[1].numeric->estimate = 0.93;
  const auto failed_score = score_observations(failed);
  require(failed_score.failed == 1 && !failed_score.all_required_passed(),
          "A1 result below 0.94 passed");

  const auto smoke = assess_equivalence(
      {.scope = EvidenceScope::canonical_smoke_test,
       .observations = observations,
       .ablations = passing_ablations()});
  require(!smoke.behavioral_equivalence_supported &&
              smoke.spiking_mechanism_supported &&
              !smoke.full_equivalence_supported,
          "canonical smoke results were allowed to claim equivalence");

  const auto behavior_only = assess_equivalence(
      {.scope = EvidenceScope::published_protocol_run,
       .observations = observations,
       .ablations = {}});
  require(behavior_only.behavioral_equivalence_supported &&
              !behavior_only.spiking_mechanism_supported &&
              !behavior_only.full_equivalence_supported,
          "behavioral metrics alone were attributed to the spiking substrate");

  const auto full = assess_equivalence(
      {.scope = EvidenceScope::published_protocol_run,
       .observations = observations,
       .ablations = passing_ablations()});
  require(full.behavioral_equivalence_supported &&
              full.spiking_mechanism_supported &&
              full.full_equivalence_supported,
          "complete protocol and lesion evidence did not pass");

  const auto reverse = assess_causal_ablation(
      {.ablation_id = "lower-is-better",
       .task = Task::counting,
       .pathway = CausalPathway::working_memory_recurrence,
       .intact = MetricSummary{.estimate = 0.20, .sample_size = 25},
       .ablated = MetricSummary{.estimate = 0.55, .sample_size = 25},
       .higher_is_better = false,
       .project_minimum_absolute_effect = 0.25,
       .completed = true});
  require(reverse.status == AssessmentStatus::passed &&
              near(reverse.signed_effect, 0.35),
          "lower-is-better causal effect was scored incorrectly");
}

}  // namespace

int main() {
  try {
    test_published_facts_are_separate_from_project_gates();
    test_deterministic_protocol_schedules();
    test_statistics();
    test_gate_scoring_and_equivalence_claim_guard();
    std::cout << "spaun_benchmarks_tests: passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "spaun_benchmarks_tests: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
