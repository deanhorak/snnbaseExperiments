#include <snnbase_experiments/spaun_benchmark_runner.hpp>

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using snnbase_experiments::spaun::Task;
using namespace snnbase_experiments::spaun::benchmark;

struct RegistrationBuilder {
  std::optional<std::size_t> correct;
  std::optional<std::size_t> total;
  std::string dataset;
  std::string split;
  std::string artifact;
  bool held_out{};

  [[nodiscard]] bool any() const noexcept {
    return correct.has_value() || total.has_value() || !dataset.empty() ||
           !split.empty() || !artifact.empty() || held_out;
  }
};

struct Options {
  RunnerOptions runner;
  std::vector<Task> tasks{Task::reinforcement_learning,
                          Task::serial_working_memory,
                          Task::counting,
                          Task::question_answering};
  std::filesystem::path json_path;
  bool json_stdout{};
  bool require_equivalence{};
  RegistrationBuilder a1;
};

void usage(std::ostream& output, std::string_view program) {
  output
      << "Usage: " << program << " [options]\n"
      << "  --task LIST          all, A0-A7, names, or comma-separated tasks\n"
      << "                       (default: A2,A3,A4,A5)\n"
      << "  --seed N             Deterministic schedule seed (default: 42)\n"
      << "  --quick              Subsample A3-A5 and use a small/fast Model\n"
      << "  --bootstrap N        Bootstrap resamples (default: 3000)\n"
      << "  --memory-noise F     Recurrent-memory noise standard deviation\n"
      << "  --memory-recurrence F  Middle-position recurrent gain\n"
      << "  --primacy-recurrence F First-position recurrent gain\n"
      << "  --recency-recurrence F Last-position recurrent gain\n"
      << "  --counting-delay N   Neural successor delay per counted item\n"
      << "  --json PATH          Write machine-readable report; '-' means stdout\n"
      << "  --require-equivalence  Return failure unless full equivalence is supported\n"
      << "\nRegistered A1 result (all fields are required together):\n"
      << "  --a1-correct N       Correct held-out classifications\n"
      << "  --a1-total N         Total held-out classifications\n"
      << "  --a1-dataset ID      Immutable dataset identifier\n"
      << "  --a1-split ID        Held-out split identifier\n"
      << "  --a1-artifact ID     Classifier/checkpoint identifier\n"
      << "  --a1-held-out        Assert that no reported item was used for fitting\n"
      << "  --help               Show this help\n";
}

template <typename Integer>
Integer parse_integer(std::string_view value, std::string_view option) {
  Integer parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size()) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return parsed;
}

double parse_finite(std::string_view value, std::string_view option) {
  double parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      !std::isfinite(parsed)) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return parsed;
}

std::vector<Task> parse_tasks(std::string_view value) {
  if (value == "all") {
    const auto tasks = snnbase_experiments::spaun::all_tasks();
    return {tasks.begin(), tasks.end()};
  }
  std::vector<Task> tasks;
  for (std::size_t begin = 0; begin <= value.size();) {
    const auto separator = value.find(',', begin);
    const auto end = separator == std::string_view::npos ? value.size() : separator;
    if (end == begin) {
      throw std::invalid_argument("--task contains an empty task name");
    }
    tasks.push_back(
        snnbase_experiments::spaun::parse_task(value.substr(begin, end - begin)));
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
  }
  return tasks;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help") {
      usage(std::cout, argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (argument == "--quick") {
      options.runner.quick = true;
      continue;
    }
    if (argument == "--require-equivalence") {
      options.require_equivalence = true;
      continue;
    }
    if (argument == "--a1-held-out") {
      options.a1.held_out = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(argument));
    }
    const std::string_view value(argv[++index]);
    if (argument == "--task") {
      options.tasks = parse_tasks(value);
    } else if (argument == "--seed") {
      options.runner.seed = parse_integer<std::uint32_t>(value, argument);
      options.runner.model_config.seed = options.runner.seed;
    } else if (argument == "--bootstrap") {
      options.runner.bootstrap_samples = parse_integer<std::size_t>(value, argument);
      if (options.runner.bootstrap_samples == 0U) {
        throw std::invalid_argument("--bootstrap must be positive");
      }
    } else if (argument == "--memory-noise") {
      options.runner.model_config.memory_noise_standard_deviation =
          parse_finite(value, argument);
    } else if (argument == "--memory-recurrence") {
      options.runner.model_config.memory_recurrent_weight_gain =
          parse_finite(value, argument);
    } else if (argument == "--primacy-recurrence") {
      options.runner.model_config.primacy_recurrent_weight_gain =
          parse_finite(value, argument);
    } else if (argument == "--recency-recurrence") {
      options.runner.model_config.recency_recurrent_weight_gain =
          parse_finite(value, argument);
    } else if (argument == "--counting-delay") {
      options.runner.model_config.counting_delay_ticks =
          parse_integer<std::size_t>(value, argument);
      if (options.runner.model_config.counting_delay_ticks == 0U) {
        throw std::invalid_argument("--counting-delay must be positive");
      }
    } else if (argument == "--json") {
      options.json_stdout = value == "-";
      if (!options.json_stdout) {
        options.json_path = value;
      }
    } else if (argument == "--a1-correct") {
      options.a1.correct = parse_integer<std::size_t>(value, argument);
    } else if (argument == "--a1-total") {
      options.a1.total = parse_integer<std::size_t>(value, argument);
    } else if (argument == "--a1-dataset") {
      options.a1.dataset = value;
    } else if (argument == "--a1-split") {
      options.a1.split = value;
    } else if (argument == "--a1-artifact") {
      options.a1.artifact = value;
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }

  if (options.a1.any()) {
    if (!options.a1.correct.has_value() || !options.a1.total.has_value() ||
        options.a1.dataset.empty() || options.a1.split.empty() ||
        options.a1.artifact.empty() || !options.a1.held_out) {
      throw std::invalid_argument(
          "a registered A1 result requires correct, total, dataset, split, "
          "artifact, and --a1-held-out");
    }
    RegisteredRecognitionRun registration{
        .correct = *options.a1.correct,
        .total = *options.a1.total,
        .dataset_id = options.a1.dataset,
        .split_id = options.a1.split,
        .classifier_artifact_id = options.a1.artifact,
        .held_out = true};
    if (!registration.valid()) {
      throw std::invalid_argument("registered A1 values are inconsistent");
    }
    options.runner.registered_a1 = std::move(registration);
  }
  return options;
}

void print_summary(const BenchmarkReport& report) {
  std::cout << "Spaun behavioral benchmark runner\n"
            << "seed=" << report.seed << " quick=" << std::boolalpha
            << report.quick << " bootstrap=" << report.bootstrap_samples
            << " evidence_scope=" << evidence_scope_name(report.evidence.scope)
            << "\n\n";
  for (const auto& task : report.tasks) {
    std::cout << 'A' << static_cast<int>(task.task) << ' '
              << snnbase_experiments::spaun::task_name(task.task)
              << " status=" << task_run_status_name(task.status)
              << " executed=" << task.executed_trials << '/'
              << task.planned_trials << " scope="
              << evidence_scope_name(task.evidence_scope) << '\n';
    for (const auto& metric : task.metrics) {
      std::cout << "  " << metric.metric_id << '=' << std::fixed
                << std::setprecision(4) << metric.summary.estimate
                << " n=" << metric.summary.sample_size;
      if (metric.summary.interval.has_value()) {
        std::cout << " ci95=[" << metric.summary.interval->lower << ','
                  << metric.summary.interval->upper << ']';
      }
      std::cout << ' ' << metric.unit << '\n';
    }
    for (const auto& caveat : task.caveats) {
      std::cout << "  caveat: " << caveat << '\n';
    }
  }
  std::cout << "\ngates passed=" << report.equivalence.behavioral_score.passed
            << " failed=" << report.equivalence.behavioral_score.failed
            << " missing=" << report.equivalence.behavioral_score.missing
            << "\nbehavioral_equivalence_supported="
            << report.equivalence.behavioral_equivalence_supported
            << " spiking_mechanism_supported="
            << report.equivalence.spiking_mechanism_supported
            << " full_equivalence_supported="
            << report.equivalence.full_equivalence_supported << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    const auto report = run_behavioral_benchmarks(options.tasks, options.runner);
    if (options.json_stdout) {
      write_report_json(std::cout, report);
    } else {
      print_summary(report);
      if (!options.json_path.empty()) {
        std::ofstream output(options.json_path);
        if (!output) {
          throw std::runtime_error("could not open JSON report: " +
                                   options.json_path.string());
        }
        write_report_json(output, report);
        std::cout << "json=" << options.json_path.string() << '\n';
      }
    }
    return options.require_equivalence &&
                   !report.equivalence.full_equivalence_supported
               ? EXIT_FAILURE
               : EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "spaun_benchmark: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
