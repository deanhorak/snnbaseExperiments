#ifndef SNNBASE_EXPERIMENTS_SPAUN_BENCHMARKS_HPP
#define SNNBASE_EXPERIMENTS_SPAUN_BENCHMARKS_HPP

#include <snnbase_experiments/spaun.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snnbase_experiments::spaun::benchmark {

inline constexpr std::size_t published_bootstrap_samples = 3000;

struct ConfidenceInterval {
  double lower{};
  double upper{};
  double confidence_level{0.95};
};

struct MetricSummary {
  double estimate{};
  std::size_t sample_size{};
  std::optional<ConfidenceInterval> interval{};
  std::size_t bootstrap_samples{};
};

enum class ReferenceKind : std::uint8_t {
  model_result,
  human_comparator,
};

struct NamedProtocolValue {
  std::string_view id;
  double value{};
  std::string_view unit;
  std::string_view description;
};

struct SharedPublishedProtocol {
  std::size_t image_rows{28};
  std::size_t image_columns{28};
  double stimulus_seconds{0.150};
  double blank_seconds{0.150};
  bool arm_is_only_behavioral_output{true};
};

struct PublishedTaskProtocol {
  Task task{};
  std::string_view name;
  std::vector<NamedProtocolValue> parameters{};
  std::string_view protocol_note;
};

struct PublishedNumericReference {
  Task task{};
  std::string_view metric_id;
  double estimate{};
  std::optional<ConfidenceInterval> interval{};
  std::optional<double> reported_plus_minus{};
  std::string_view unit;
  ReferenceKind kind{ReferenceKind::model_result};
  std::string_view description;
};

struct PublishedQualitativeClaim {
  Task task{};
  std::string_view claim_id;
  std::string_view description;
};

[[nodiscard]] SharedPublishedProtocol shared_published_protocol() noexcept;
[[nodiscard]] std::vector<PublishedTaskProtocol> published_task_protocols();
[[nodiscard]] std::vector<PublishedNumericReference>
published_numeric_references();
[[nodiscard]] std::vector<PublishedQualitativeClaim>
published_qualitative_claims();

// A generated schedule identifies which fields are fixed by a published
// protocol and which require an explicit project choice. A false
// fully_specified_by_publication flag must be carried into result provenance.
struct PlannedTrial {
  Task task{};
  std::size_t trial_index{};
  std::size_t simulated_participant{};
  std::size_t condition_index{};
  std::vector<int> digits{};
  char query_kind{};
  int query_value{};
  std::array<double, 3> reward_probabilities{};
  std::uint32_t trial_seed{};
};

struct TrialSchedule {
  Task task{};
  std::vector<PlannedTrial> trials{};
  bool fully_specified_by_publication{};
  std::string provenance_note{};
};

[[nodiscard]] TrialSchedule make_trial_schedule(Task task,
                                                std::uint32_t seed = 42);

[[nodiscard]] MetricSummary empirical_proportion(
    std::span<const std::uint8_t> outcomes);
[[nodiscard]] MetricSummary empirical_mean(std::span<const double> values);
[[nodiscard]] MetricSummary bootstrap_proportion(
    std::span<const std::uint8_t> outcomes,
    std::size_t resamples = published_bootstrap_samples,
    std::uint32_t seed = 42,
    double confidence_level = 0.95);
[[nodiscard]] MetricSummary bootstrap_mean(
    std::span<const double> values,
    std::size_t resamples = published_bootstrap_samples,
    std::uint32_t seed = 42,
    double confidence_level = 0.95);
[[nodiscard]] double ordinary_least_squares_slope(
    std::span<const double> x, std::span<const double> y);

enum class GateRule : std::uint8_t {
  minimum,
  reference_band,
  confidence_interval_overlap,
  qualitative_review,
};

// These are project acceptance decisions, not claims that the publication
// specified pass/fail thresholds. In particular, there is no universal 95%
// accuracy gate across A0--A7.
struct ProjectAcceptanceGate {
  Task task{};
  std::string_view metric_id;
  GateRule rule{GateRule::qualitative_review};
  std::optional<double> lower{};
  std::optional<double> upper{};
  std::string_view rationale;
  bool required{true};
};

[[nodiscard]] std::vector<ProjectAcceptanceGate> project_acceptance_gates();

struct Observation {
  Task task{};
  std::string metric_id{};
  std::optional<MetricSummary> numeric{};
  std::optional<bool> qualitative_pass{};
  std::string provenance{};
};

enum class AssessmentStatus : std::uint8_t {
  passed,
  failed,
  missing,
};

struct GateAssessment {
  ProjectAcceptanceGate gate;
  AssessmentStatus status{AssessmentStatus::missing};
  std::string explanation{};
};

struct GateScore {
  std::vector<GateAssessment> assessments{};
  std::size_t passed{};
  std::size_t failed{};
  std::size_t missing{};

  [[nodiscard]] bool all_required_passed() const noexcept;
};

[[nodiscard]] GateScore score_observations(
    std::span<const Observation> observations,
    std::span<const ProjectAcceptanceGate> gates);
[[nodiscard]] GateScore score_observations(
    std::span<const Observation> observations);

enum class CausalPathway : std::uint8_t {
  visual_to_encoding,
  working_memory_recurrence,
  reward_plasticity,
  transformation,
  action_selection,
  decoding_to_motor,
  motor_to_arm,
};

struct CausalAblationResult {
  std::string ablation_id{};
  Task task{};
  CausalPathway pathway{CausalPathway::visual_to_encoding};
  MetricSummary intact;
  MetricSummary ablated;
  bool higher_is_better{true};
  double project_minimum_absolute_effect{};
  bool completed{};
  std::string notes{};
};

struct CausalAblationAssessment {
  CausalAblationResult result;
  AssessmentStatus status{AssessmentStatus::missing};
  double signed_effect{};
  std::string explanation{};
};

[[nodiscard]] CausalAblationAssessment assess_causal_ablation(
    const CausalAblationResult& result);

enum class EvidenceScope : std::uint8_t {
  canonical_smoke_test,
  published_protocol_run,
};

struct BenchmarkEvidence {
  EvidenceScope scope{EvidenceScope::canonical_smoke_test};
  std::vector<Observation> observations{};
  std::vector<CausalAblationResult> ablations{};
};

struct EquivalenceAssessment {
  GateScore behavioral_score{};
  std::vector<CausalAblationAssessment> causal_assessments{};
  bool behavioral_equivalence_supported{};
  bool spiking_mechanism_supported{};
  bool full_equivalence_supported{};
  std::vector<std::string> caveats{};
};

[[nodiscard]] EquivalenceAssessment assess_equivalence(
    const BenchmarkEvidence& evidence);

}  // namespace snnbase_experiments::spaun::benchmark

#endif  // SNNBASE_EXPERIMENTS_SPAUN_BENCHMARKS_HPP
