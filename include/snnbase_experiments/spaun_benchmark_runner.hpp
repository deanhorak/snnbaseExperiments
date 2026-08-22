#ifndef SNNBASE_EXPERIMENTS_SPAUN_BENCHMARK_RUNNER_HPP
#define SNNBASE_EXPERIMENTS_SPAUN_BENCHMARK_RUNNER_HPP

#include <snnbase_experiments/spaun.hpp>
#include <snnbase_experiments/spaun_benchmarks.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snnbase_experiments::spaun::benchmark {

// A task can be numerically executed while still being unsuitable as evidence
// for the published protocol. Keeping that state separate from pass/fail stops
// a canonical Model::run smoke test from silently becoming an equivalence run.
enum class TaskRunStatus : std::uint8_t {
  completed,
  completed_protocol_deviation,
  unavailable_requires_registration,
  unresolved_stimulus_schedule,
};

struct RegisteredRecognitionRun {
  std::size_t correct{};
  std::size_t total{};
  std::string dataset_id{};
  std::string split_id{};
  std::string classifier_artifact_id{};
  bool held_out{};

  [[nodiscard]] bool valid() const noexcept;
};

struct RunnerOptions {
  Config model_config{};
  std::uint32_t seed{42};
  bool quick{};
  std::size_t bootstrap_samples{published_bootstrap_samples};
  std::optional<RegisteredRecognitionRun> registered_a1{};
};

struct TrialRecord {
  std::size_t schedule_index{};
  std::size_t simulated_participant{};
  std::size_t condition_index{};
  std::uint32_t trial_seed{};
  std::string visual_stream{};
  std::vector<int> expected{};
  std::vector<int> output{};
  bool correct{};
  double response_onset_seconds{};
  std::size_t spikes{};
  double simulated_seconds{};
};

struct BanditChoice {
  std::size_t trial_index{};
  std::size_t block{};
  int choice{-1};
  int published_best_arm{-1};
  bool rewarded{};
  double moving_best_arm_choice_probability{};
};

struct MetricRecord {
  std::string metric_id{};
  MetricSummary summary{};
  std::string unit{};
  std::string provenance{};
};

struct TaskBenchmarkResult {
  Task task{};
  TaskRunStatus status{TaskRunStatus::unresolved_stimulus_schedule};
  EvidenceScope evidence_scope{EvidenceScope::canonical_smoke_test};
  bool quick{};
  std::size_t planned_trials{};
  std::size_t executed_trials{};
  bool schedule_fully_specified_by_publication{};
  std::string schedule_provenance{};
  std::vector<TrialRecord> trials{};
  std::vector<BanditChoice> bandit_choices{};
  std::vector<MetricRecord> metrics{};
  std::vector<Observation> observations{};
  std::vector<std::string> caveats{};
};

struct BenchmarkReport {
  std::uint32_t seed{};
  bool quick{};
  std::size_t bootstrap_samples{};
  std::vector<TaskBenchmarkResult> tasks{};
  BenchmarkEvidence evidence{};
  EquivalenceAssessment equivalence{};
  std::vector<std::string> caveats{};
};

// These two functions deliberately separate model input construction from the
// scoring oracle. make_visual_trial leaves groups/query fields empty: the only
// operands and task cue available to Model are glyphs in stimulus_stream.
[[nodiscard]] Trial make_visual_trial(const PlannedTrial& planned);
[[nodiscard]] std::vector<int> expected_answer(const PlannedTrial& planned);
[[nodiscard]] bool has_clean_model_metadata(const Trial& trial) noexcept;

[[nodiscard]] BenchmarkReport run_behavioral_benchmarks(
    std::span<const Task> tasks, const RunnerOptions& options = {});

[[nodiscard]] std::string_view task_run_status_name(
    TaskRunStatus status) noexcept;
[[nodiscard]] std::string_view evidence_scope_name(
    EvidenceScope scope) noexcept;
void write_report_json(std::ostream& output, const BenchmarkReport& report);

}  // namespace snnbase_experiments::spaun::benchmark

#endif  // SNNBASE_EXPERIMENTS_SPAUN_BENCHMARK_RUNNER_HPP
