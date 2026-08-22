#include <snnbase_experiments/spaun_benchmarks.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace snnbase_experiments::spaun::benchmark {
namespace {

void validate_confidence_level(double confidence_level) {
  if (!(confidence_level > 0.0 && confidence_level < 1.0) ||
      !std::isfinite(confidence_level)) {
    throw std::invalid_argument("confidence level must be finite and in (0, 1)");
  }
}

void validate_finite_values(std::span<const double> values) {
  if (values.empty()) {
    throw std::invalid_argument("a metric requires at least one observation");
  }
  for (const auto value : values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("metric observations must be finite");
    }
  }
}

double mean_of(std::span<const double> values) {
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

double percentile(const std::vector<double>& sorted_values, double probability) {
  const auto position =
      probability * static_cast<double>(sorted_values.size() - 1U);
  const auto lower_index = static_cast<std::size_t>(std::floor(position));
  const auto upper_index = static_cast<std::size_t>(std::ceil(position));
  const auto fraction = position - static_cast<double>(lower_index);
  return sorted_values[lower_index] * (1.0 - fraction) +
         sorted_values[upper_index] * fraction;
}

template <typename Statistic>
MetricSummary bootstrap_impl(std::size_t sample_size, double estimate,
                             std::size_t resamples, std::uint32_t seed,
                             double confidence_level, Statistic statistic) {
  if (sample_size == 0U) {
    throw std::invalid_argument("bootstrap requires at least one observation");
  }
  if (resamples == 0U) {
    throw std::invalid_argument("bootstrap requires at least one resample");
  }
  validate_confidence_level(confidence_level);

  std::mt19937 generator(seed);
  std::vector<double> estimates;
  estimates.reserve(resamples);
  for (std::size_t resample = 0; resample < resamples; ++resample) {
    estimates.push_back(statistic(generator));
  }
  std::sort(estimates.begin(), estimates.end());
  const auto tail = (1.0 - confidence_level) / 2.0;
  return {.estimate = estimate,
          .sample_size = sample_size,
          .interval = ConfidenceInterval{.lower = percentile(estimates, tail),
                                         .upper = percentile(estimates, 1.0 - tail),
                                         .confidence_level = confidence_level},
          .bootstrap_samples = resamples};
}

std::uint32_t next_seed(std::mt19937& generator) {
  return generator();
}

std::size_t bounded(std::mt19937& generator, std::size_t bound) {
  if (bound == 0U) {
    throw std::logic_error("bounded random draw has an empty range");
  }
  return static_cast<std::size_t>(generator()) % bound;
}

std::vector<int> digit_sample(std::mt19937& generator, std::size_t length) {
  std::array<int, 10> pool{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  for (std::size_t remaining = pool.size(); remaining > 1U; --remaining) {
    const auto selected = bounded(generator, remaining);
    std::swap(pool[selected], pool[remaining - 1U]);
  }
  return {pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(length)};
}

bool intervals_overlap(const ConfidenceInterval& first,
                       const ConfidenceInterval& second) {
  return first.lower <= second.upper && second.lower <= first.upper;
}

std::string numeric_text(double value) {
  std::ostringstream stream;
  stream << value;
  return stream.str();
}

}  // namespace

SharedPublishedProtocol shared_published_protocol() noexcept { return {}; }

std::vector<PublishedTaskProtocol> published_task_protocols() {
  return {
      {.task = Task::copy_drawing,
       .name = "A0 copy drawing",
       .parameters = {{"additional_examples", 20.0, "examples",
                       "Additional handwritten examples shown in the report"}},
       .protocol_note =
           "Held-out handwritten input and reproduced style were assessed "
           "qualitatively; no numeric pass threshold was published."},
      {.task = Task::image_recognition,
       .name = "A1 image recognition",
       .parameters = {},
       .protocol_note =
           "Recognition used held-out handwritten digits, but the exact split "
           "and sample count are not stated in the primary report."},
      {.task = Task::reinforcement_learning,
       .name = "A2 three-armed bandit",
       .parameters = {{"trials", 60.0, "trials", "One 60-trial trajectory"},
                      {"block_length", 20.0, "trials", "Trials per reward block"},
                      {"moving_window", 5.0, "trials",
                       "Window used for plotted choice probability"},
                      {"high_reward_probability", 0.72, "probability",
                       "Reward probability of the best arm"},
                      {"low_reward_probability", 0.12, "probability",
                       "Reward probability of either other arm"}},
       .protocol_note =
           "Best arm order by block is arm 3, arm 2, then arm 1; no numeric "
           "adaptation pass threshold was published."},
      {.task = Task::serial_working_memory,
       .name = "A3 serial working memory",
       .parameters = {{"minimum_list_length", 4.0, "items", "Shortest list"},
                      {"maximum_list_length", 7.0, "items", "Longest list"},
                      {"runs_per_length", 40.0, "runs", "Runs for each list length"},
                      {"position_means", 22.0, "means",
                       "Total serial-position means over lengths four through seven"},
                      {"bootstrap_resamples", 3000.0, "resamples",
                       "Resamples used for reported confidence intervals"}},
       .protocol_note =
           "Comparison emphasizes the serial-position profile, including "
           "primacy and recency, rather than a single accuracy cutoff."},
      {.task = Task::counting,
       .name = "A4 counting",
       .parameters = {{"minimum_count", 1.0, "items", "Smallest count"},
                      {"maximum_count", 5.0, "items", "Largest count"},
                      {"simulated_individuals", 5.0, "individuals",
                       "Reported simulated individuals"}},
       .protocol_note =
           "The published numeric comparison is response-time slope, not a "
           "universal task accuracy."},
      {.task = Task::question_answering,
       .name = "A5 question answering",
       .parameters = {{"list_length", 7.0, "items", "Items in each list"},
                      {"simulated_individuals", 10.0, "individuals",
                       "Reported simulated individuals"},
                      {"positions", 7.0, "positions", "Queried serial positions"},
                      {"query_types", 2.0, "types", "P and K query forms"},
                      {"total_queries", 140.0, "queries",
                       "Ten individuals times seven positions times two forms"}},
       .protocol_note =
           "The P/K profiles and their primacy/recency structure were compared; "
           "no numeric pass threshold was published."},
      {.task = Task::rapid_variable_creation,
       .name = "A6 rapid variable creation",
       .parameters = {{"response_delay", 0.150, "seconds",
                       "Answer period begins after the final example"}},
       .protocol_note =
           "The report presents transformation examples and timing, but no "
           "aggregate accuracy threshold."},
      {.task = Task::fluid_reasoning,
       .name = "A7 fluid reasoning",
       .parameters = {{"runs", 40.0, "runs", "Reported reasoning runs"},
                      {"bootstrap_resamples", 3000.0, "resamples",
                       "Resamples used for the reported confidence interval"}},
       .protocol_note =
           "Raw exact accuracy and its interval must remain distinct from the "
           "chance-adjusted value."},
  };
}

std::vector<PublishedNumericReference> published_numeric_references() {
  return {
      {.task = Task::image_recognition,
       .metric_id = "recognition_accuracy",
       .estimate = 0.94,
       .interval = std::nullopt,
       .reported_plus_minus = std::nullopt,
       .unit = "proportion correct",
       .kind = ReferenceKind::model_result,
       .description = "Reported Spaun held-out digit-recognition accuracy"},
      {.task = Task::image_recognition,
       .metric_id = "human_recognition_accuracy",
       .estimate = 0.98,
       .interval = std::nullopt,
       .reported_plus_minus = std::nullopt,
       .unit = "proportion correct",
       .kind = ReferenceKind::human_comparator,
       .description = "Approximate human comparison reported with A1"},
      {.task = Task::serial_working_memory,
       .metric_id = "human_means_inside_model_ci",
       .estimate = 17.0 / 22.0,
       .interval = std::nullopt,
       .reported_plus_minus = std::nullopt,
       .unit = "fraction of position means",
       .kind = ReferenceKind::model_result,
       .description = "Seventeen of 22 human means fell inside Spaun intervals"},
      {.task = Task::counting,
       .metric_id = "response_time_slope_ms_per_item",
       .estimate = 419.0,
       .interval = std::nullopt,
       .reported_plus_minus = 10.0,
       .unit = "milliseconds per item",
       .kind = ReferenceKind::model_result,
       .description = "Reported Spaun counting response-time slope"},
      {.task = Task::counting,
       .metric_id = "human_response_time_slope_ms_per_item",
       .estimate = 344.0,
       .interval = std::nullopt,
       .reported_plus_minus = 135.0,
       .unit = "milliseconds per item",
       .kind = ReferenceKind::human_comparator,
       .description = "Reported human counting slope comparison"},
      {.task = Task::fluid_reasoning,
       .metric_id = "raw_accuracy",
       .estimate = 0.75,
       .interval = ConfidenceInterval{.lower = 0.60,
                                      .upper = 0.88,
                                      .confidence_level = 0.95},
       .reported_plus_minus = std::nullopt,
       .unit = "proportion correct",
       .kind = ReferenceKind::model_result,
       .description = "Thirty exact answers in 40 Spaun reasoning runs"},
      {.task = Task::fluid_reasoning,
       .metric_id = "chance_adjusted_accuracy",
       .estimate = 0.88,
       .interval = std::nullopt,
       .reported_plus_minus = std::nullopt,
       .unit = "adjusted proportion",
       .kind = ReferenceKind::model_result,
       .description = "Published chance-adjusted A7 value; not raw accuracy"},
      {.task = Task::fluid_reasoning,
       .metric_id = "human_accuracy",
       .estimate = 0.89,
       .interval = std::nullopt,
       .reported_plus_minus = std::nullopt,
       .unit = "proportion correct",
       .kind = ReferenceKind::human_comparator,
       .description = "Reported human A7 comparison"},
  };
}

std::vector<PublishedQualitativeClaim> published_qualitative_claims() {
  return {
      {Task::copy_drawing, "recognizable_copy",
       "The arm reproduces recognizable handwritten digits in the input style."},
      {Task::reinforcement_learning, "blockwise_adaptation",
       "Choice preference adapts when the high-reward arm changes by block."},
      {Task::serial_working_memory, "serial_position_profile",
       "Recall exhibits human-like primacy and recency over list position."},
      {Task::question_answering, "question_answering_profile",
       "P and K queries have similar serial-position profiles with primacy and recency."},
      {Task::rapid_variable_creation, "variable_rule_generalization",
       "A newly inferred transformation is applied to a novel operand."},
  };
}

TrialSchedule make_trial_schedule(Task task, std::uint32_t seed) {
  TrialSchedule schedule{.task = task};
  std::mt19937 generator(seed);

  switch (task) {
    case Task::copy_drawing:
      schedule.trials.reserve(20);
      for (std::size_t trial = 0; trial < 20; ++trial) {
        schedule.trials.push_back(
            {.task = task,
             .trial_index = trial,
             .digits = {static_cast<int>(bounded(generator, 10))},
             .trial_seed = next_seed(generator)});
      }
      schedule.fully_specified_by_publication = false;
      schedule.provenance_note =
          "The publication gives 20 additional examples but not their exact "
          "MNIST indices; digit identities and image selection remain project choices.";
      break;

    case Task::image_recognition:
      schedule.fully_specified_by_publication = false;
      schedule.provenance_note =
          "A fixed held-out dataset split must be registered by the project; the "
          "publication does not provide enough information to generate it.";
      break;

    case Task::reinforcement_learning: {
      constexpr std::array<std::array<double, 3>, 3> blocks{{
          {0.12, 0.12, 0.72},
          {0.12, 0.72, 0.12},
          {0.72, 0.12, 0.12},
      }};
      schedule.trials.reserve(60);
      for (std::size_t trial = 0; trial < 60; ++trial) {
        schedule.trials.push_back({.task = task,
                                   .trial_index = trial,
                                   .condition_index = trial / 20U,
                                   .reward_probabilities = blocks[trial / 20U],
                                   .trial_seed = next_seed(generator)});
      }
      schedule.fully_specified_by_publication = true;
      schedule.provenance_note =
          "Trial count, block boundaries, and reward probabilities follow the "
          "published protocol; the seed only makes reward sampling reproducible.";
      break;
    }

    case Task::serial_working_memory:
      schedule.trials.reserve(160);
      for (std::size_t length = 4; length <= 7; ++length) {
        for (std::size_t repetition = 0; repetition < 40; ++repetition) {
          schedule.trials.push_back(
              {.task = task,
               .trial_index = schedule.trials.size(),
               .condition_index = length - 4U,
               .digits = digit_sample(generator, length),
               .trial_seed = next_seed(generator)});
        }
      }
      schedule.fully_specified_by_publication = false;
      schedule.provenance_note =
          "List lengths and repetitions are published; this deterministic "
          "project generator supplies the otherwise unstated digit lists.";
      break;

    case Task::counting:
      schedule.trials.reserve(25);
      for (std::size_t participant = 0; participant < 5; ++participant) {
        for (std::size_t count = 1; count <= 5; ++count) {
          const auto start = bounded(generator, 10U - count);
          schedule.trials.push_back(
              {.task = task,
               .trial_index = schedule.trials.size(),
               .simulated_participant = participant,
               .condition_index = count - 1U,
               .digits = {static_cast<int>(start), static_cast<int>(count)},
               .trial_seed = next_seed(generator)});
        }
      }
      schedule.fully_specified_by_publication = false;
      schedule.provenance_note =
          "Count lengths and participant count are published; starting digits "
          "are generated here and must be recorded with benchmark results.";
      break;

    case Task::question_answering:
      schedule.trials.reserve(140);
      for (std::size_t participant = 0; participant < 10; ++participant) {
        for (std::size_t position = 0; position < 7; ++position) {
          for (std::size_t query = 0; query < 2; ++query) {
            auto digits = digit_sample(generator, 7);
            const auto query_kind = query == 0U ? 'P' : 'K';
            const auto query_value = query_kind == 'P'
                                         ? static_cast<int>(position + 1U)
                                         : digits[position];
            schedule.trials.push_back(
                {.task = task,
                 .trial_index = schedule.trials.size(),
                 .simulated_participant = participant,
                 .condition_index = position,
                 .digits = std::move(digits),
                 .query_kind = query_kind,
                 .query_value = query_value,
                 .trial_seed = next_seed(generator)});
          }
        }
      }
      schedule.fully_specified_by_publication = false;
      schedule.provenance_note =
          "The 10 by 7 by 2 design is published; digit lists are supplied by "
          "this deterministic project generator.";
      break;

    case Task::rapid_variable_creation:
      schedule.fully_specified_by_publication = false;
      schedule.provenance_note =
          "Register an explicit source/target transformation corpus before "
          "scoring; published examples alone do not define an aggregate trial set.";
      break;

    case Task::fluid_reasoning:
      schedule.trials.reserve(40);
      for (std::size_t trial = 0; trial < 40; ++trial) {
        schedule.trials.push_back({.task = task,
                                   .trial_index = trial,
                                   .trial_seed = next_seed(generator)});
      }
      schedule.fully_specified_by_publication = false;
      schedule.provenance_note =
          "The run count is published; exact reasoning items must be registered "
          "rather than inferred from the reported aggregate.";
      break;
  }
  return schedule;
}

MetricSummary empirical_proportion(std::span<const std::uint8_t> outcomes) {
  if (outcomes.empty()) {
    throw std::invalid_argument("a proportion requires at least one outcome");
  }
  std::size_t successes = 0;
  for (const auto outcome : outcomes) {
    if (outcome > 1U) {
      throw std::invalid_argument("proportion outcomes must be zero or one");
    }
    successes += outcome;
  }
  return {.estimate = static_cast<double>(successes) /
                      static_cast<double>(outcomes.size()),
          .sample_size = outcomes.size()};
}

MetricSummary empirical_mean(std::span<const double> values) {
  validate_finite_values(values);
  return {.estimate = mean_of(values), .sample_size = values.size()};
}

MetricSummary bootstrap_proportion(std::span<const std::uint8_t> outcomes,
                                   std::size_t resamples, std::uint32_t seed,
                                   double confidence_level) {
  const auto empirical = empirical_proportion(outcomes);
  return bootstrap_impl(
      outcomes.size(), empirical.estimate, resamples, seed, confidence_level,
      [&](std::mt19937& generator) {
        std::size_t successes = 0;
        for (std::size_t draw = 0; draw < outcomes.size(); ++draw) {
          successes += outcomes[bounded(generator, outcomes.size())];
        }
        return static_cast<double>(successes) /
               static_cast<double>(outcomes.size());
      });
}

MetricSummary bootstrap_mean(std::span<const double> values,
                             std::size_t resamples, std::uint32_t seed,
                             double confidence_level) {
  const auto empirical = empirical_mean(values);
  return bootstrap_impl(
      values.size(), empirical.estimate, resamples, seed, confidence_level,
      [&](std::mt19937& generator) {
        double sum = 0.0;
        for (std::size_t draw = 0; draw < values.size(); ++draw) {
          sum += values[bounded(generator, values.size())];
        }
        return sum / static_cast<double>(values.size());
      });
}

double ordinary_least_squares_slope(std::span<const double> x,
                                    std::span<const double> y) {
  validate_finite_values(x);
  validate_finite_values(y);
  if (x.size() != y.size()) {
    throw std::invalid_argument("regression vectors must have equal length");
  }
  if (x.size() < 2U) {
    throw std::invalid_argument("regression requires at least two observations");
  }
  const auto x_mean = mean_of(x);
  const auto y_mean = mean_of(y);
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t index = 0; index < x.size(); ++index) {
    const auto centered_x = x[index] - x_mean;
    numerator += centered_x * (y[index] - y_mean);
    denominator += centered_x * centered_x;
  }
  if (denominator <= std::numeric_limits<double>::epsilon()) {
    throw std::invalid_argument("regression predictor has zero variance");
  }
  return numerator / denominator;
}

std::vector<ProjectAcceptanceGate> project_acceptance_gates() {
  return {
      {Task::copy_drawing, "recognizable_copy", GateRule::qualitative_review,
       std::nullopt, std::nullopt,
       "Project review must confirm recognizable held-out copies; the source has no numeric cutoff."},
      {Task::image_recognition, "recognition_accuracy", GateRule::minimum, 0.94,
       std::nullopt,
       "Project gate matches or exceeds Spaun's reported 0.94 A1 result."},
      {Task::reinforcement_learning, "blockwise_adaptation",
       GateRule::qualitative_review, std::nullopt, std::nullopt,
       "Project review checks adaptation after each contingency switch without inventing a source threshold."},
      {Task::serial_working_memory, "serial_position_profile",
       GateRule::qualitative_review, std::nullopt, std::nullopt,
       "Project review checks primacy/recency and position-wise agreement."},
      {Task::counting, "response_time_slope_ms_per_item",
       GateRule::confidence_interval_overlap, 409.0, 429.0,
       "Project confidence interval and estimate must overlap the reported 419 +/- 10 band."},
      {Task::question_answering, "question_answering_profile",
       GateRule::qualitative_review, std::nullopt, std::nullopt,
       "Project review checks P/K similarity and serial-position structure."},
      {Task::rapid_variable_creation, "variable_rule_generalization",
       GateRule::qualitative_review, std::nullopt, std::nullopt,
       "Project review requires transfer of an inferred rule to held-out operands."},
      {Task::fluid_reasoning, "raw_accuracy",
       GateRule::confidence_interval_overlap, 0.60, 0.88,
       "Project raw estimate and interval must agree with Spaun's reported raw 0.75 [0.60, 0.88], not its adjusted value."},
  };
}

bool GateScore::all_required_passed() const noexcept {
  return std::all_of(assessments.begin(), assessments.end(),
                     [](const auto& assessment) {
                       return !assessment.gate.required ||
                              assessment.status == AssessmentStatus::passed;
                     });
}

GateScore score_observations(std::span<const Observation> observations,
                             std::span<const ProjectAcceptanceGate> gates) {
  GateScore score;
  score.assessments.reserve(gates.size());
  for (const auto& gate : gates) {
    std::vector<const Observation*> matches;
    for (const auto& observation : observations) {
      if (observation.task == gate.task && observation.metric_id == gate.metric_id) {
        matches.push_back(&observation);
      }
    }

    GateAssessment assessment{.gate = gate};
    if (matches.empty()) {
      assessment.explanation = "required observation is absent";
    } else if (matches.size() > 1U) {
      assessment.explanation = "observation is ambiguous because the metric appears more than once";
    } else {
      const auto& observation = *matches.front();
      switch (gate.rule) {
        case GateRule::qualitative_review:
          if (!observation.qualitative_pass.has_value()) {
            assessment.explanation = "qualitative review has not been recorded";
          } else {
            assessment.status = *observation.qualitative_pass
                                    ? AssessmentStatus::passed
                                    : AssessmentStatus::failed;
            assessment.explanation = *observation.qualitative_pass
                                         ? "qualitative project gate passed"
                                         : "qualitative project gate failed";
          }
          break;

        case GateRule::minimum:
          if (!observation.numeric.has_value() || !gate.lower.has_value()) {
            assessment.explanation = "numeric estimate or project minimum is absent";
          } else {
            assessment.status = observation.numeric->estimate >= *gate.lower
                                    ? AssessmentStatus::passed
                                    : AssessmentStatus::failed;
            assessment.explanation =
                "estimate " + numeric_text(observation.numeric->estimate) +
                (assessment.status == AssessmentStatus::passed ? " meets " : " is below ") +
                numeric_text(*gate.lower);
          }
          break;

        case GateRule::reference_band:
          if (!observation.numeric.has_value() || !gate.lower.has_value() ||
              !gate.upper.has_value()) {
            assessment.explanation = "numeric estimate or project reference band is absent";
          } else {
            const auto estimate = observation.numeric->estimate;
            assessment.status = estimate >= *gate.lower && estimate <= *gate.upper
                                    ? AssessmentStatus::passed
                                    : AssessmentStatus::failed;
            assessment.explanation = "estimate " + numeric_text(estimate) +
                                     " compared with project band [" +
                                     numeric_text(*gate.lower) + ", " +
                                     numeric_text(*gate.upper) + "]";
          }
          break;

        case GateRule::confidence_interval_overlap:
          if (!observation.numeric.has_value() ||
              !observation.numeric->interval.has_value() ||
              !gate.lower.has_value() || !gate.upper.has_value()) {
            assessment.explanation = "confidence interval or project reference band is absent";
          } else {
            const ConfidenceInterval reference{.lower = *gate.lower,
                                               .upper = *gate.upper,
                                               .confidence_level =
                                                   observation.numeric->interval->confidence_level};
            const auto estimate_in_band =
                observation.numeric->estimate >= reference.lower &&
                observation.numeric->estimate <= reference.upper;
            const auto overlaps =
                intervals_overlap(*observation.numeric->interval, reference);
            assessment.status = estimate_in_band && overlaps
                                    ? AssessmentStatus::passed
                                    : AssessmentStatus::failed;
            assessment.explanation =
                estimate_in_band && overlaps
                    ? "estimate lies in and confidence interval overlaps the project reference band"
                    : "estimate and confidence interval do not jointly agree with the project reference band";
          }
          break;
      }
    }

    switch (assessment.status) {
      case AssessmentStatus::passed: ++score.passed; break;
      case AssessmentStatus::failed: ++score.failed; break;
      case AssessmentStatus::missing: ++score.missing; break;
    }
    score.assessments.push_back(std::move(assessment));
  }
  return score;
}

GateScore score_observations(std::span<const Observation> observations) {
  const auto gates = project_acceptance_gates();
  return score_observations(observations, gates);
}

CausalAblationAssessment assess_causal_ablation(
    const CausalAblationResult& result) {
  CausalAblationAssessment assessment{.result = result};
  if (!result.completed) {
    assessment.explanation = "ablation has not been completed";
    return assessment;
  }
  if (!std::isfinite(result.intact.estimate) ||
      !std::isfinite(result.ablated.estimate) ||
      !std::isfinite(result.project_minimum_absolute_effect) ||
      result.project_minimum_absolute_effect < 0.0) {
    assessment.status = AssessmentStatus::failed;
    assessment.explanation = "ablation contains invalid numeric evidence";
    return assessment;
  }
  assessment.signed_effect = result.higher_is_better
                                 ? result.intact.estimate - result.ablated.estimate
                                 : result.ablated.estimate - result.intact.estimate;
  assessment.status =
      assessment.signed_effect >= result.project_minimum_absolute_effect
          ? AssessmentStatus::passed
          : AssessmentStatus::failed;
  assessment.explanation =
      "signed lesion effect " + numeric_text(assessment.signed_effect) +
      " compared with project minimum " +
      numeric_text(result.project_minimum_absolute_effect);
  return assessment;
}

EquivalenceAssessment assess_equivalence(const BenchmarkEvidence& evidence) {
  EquivalenceAssessment assessment;
  assessment.behavioral_score = score_observations(evidence.observations);
  assessment.behavioral_equivalence_supported =
      evidence.scope == EvidenceScope::published_protocol_run &&
      assessment.behavioral_score.all_required_passed();

  if (evidence.scope != EvidenceScope::published_protocol_run) {
    assessment.caveats.emplace_back(
        "Canonical smoke tests are implementation checks and cannot establish behavioral equivalence.");
  }
  if (!assessment.behavioral_score.all_required_passed()) {
    assessment.caveats.emplace_back(
        "One or more project behavioral gates failed or lack protocol evidence.");
  }

  std::set<Task> causally_supported_tasks;
  bool all_completed_ablations_pass = !evidence.ablations.empty();
  assessment.causal_assessments.reserve(evidence.ablations.size());
  for (const auto& ablation : evidence.ablations) {
    auto ablation_assessment = assess_causal_ablation(ablation);
    if (ablation_assessment.status == AssessmentStatus::passed) {
      causally_supported_tasks.insert(ablation.task);
    } else {
      all_completed_ablations_pass = false;
    }
    assessment.causal_assessments.push_back(std::move(ablation_assessment));
  }
  assessment.spiking_mechanism_supported =
      all_completed_ablations_pass && causally_supported_tasks.size() == task_count;
  if (!assessment.spiking_mechanism_supported) {
    assessment.caveats.emplace_back(
        "Passing causal lesions are required for every A0--A7 task before attributing behavior to the spiking substrate.");
  }
  assessment.full_equivalence_supported =
      assessment.behavioral_equivalence_supported &&
      assessment.spiking_mechanism_supported;
  return assessment;
}

}  // namespace snnbase_experiments::spaun::benchmark
