#include <snnbase_experiments/spaun.hpp>

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  bool all{true};
  snnbase_experiments::spaun::Task task{
      snnbase_experiments::spaun::Task::serial_working_memory};
  snnbase_experiments::spaun::Config config;
  std::filesystem::path learned_digit_checkpoint;
  std::filesystem::path trace_json;
  double minimum_accuracy{};
};

void usage(std::ostream& output, std::string_view program) {
  output << "Usage: " << program << " [options]\n"
         << "  --task NAME              all, A0-A7, 0-7, or task short name\n"
         << "  --seed N                 Deterministic participant seed (default: 42)\n"
         << "  --neurons-per-module N   LIF neurons per functional module (default: 64)\n"
         << "  --stimulus-ticks N       Ticks per visible symbol (default: 15 = 150 ms)\n"
         << "  --blank-ticks N          Blank ticks between symbols (default: 15 = 150 ms)\n"
         << "  --motor-ticks N          Arm integration ticks per target (default: 5)\n"
         << "  --memory-noise F         Recurrent-memory noise standard deviation\n"
         << "  --memory-recurrence F    Middle-position recurrent gain\n"
         << "  --primacy-recurrence F   First-position recurrent gain\n"
         << "  --recency-recurrence F   Last-position recurrent gain\n"
         << "  --counting-delay N       Neural successor delay per counted item\n"
         << "  --learned-digit-checkpoint PATH\n"
         << "                           Route visible digits through registered 28x28 A1 model\n"
         << "  --trace-json PATH        Write complete probe frames as JSON\n"
         << "  --minimum-accuracy F     Fail below task-battery accuracy [0,1]\n"
         << "  --help                    Show this help\n";
}

std::size_t parse_size(std::string_view value, std::string_view option) {
  std::size_t parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() || parsed == 0) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return parsed;
}

std::uint32_t parse_seed(std::string_view value) {
  std::uint32_t parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size()) {
    throw std::invalid_argument("invalid value for --seed");
  }
  return parsed;
}

double parse_probability(std::string_view value, std::string_view option) {
  double parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() || parsed < 0.0 ||
      parsed > 1.0) {
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

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help") {
      usage(std::cout, argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(argument));
    }
    const std::string_view value(argv[++index]);
    if (argument == "--task") {
      options.all = value == "all";
      if (!options.all) {
        options.task = snnbase_experiments::spaun::parse_task(value);
      }
    } else if (argument == "--seed") {
      options.config.seed = parse_seed(value);
    } else if (argument == "--neurons-per-module") {
      options.config.neurons_per_module = parse_size(value, argument);
    } else if (argument == "--stimulus-ticks") {
      options.config.stimulus_ticks = parse_size(value, argument);
    } else if (argument == "--blank-ticks") {
      options.config.blank_ticks = parse_size(value, argument);
    } else if (argument == "--motor-ticks") {
      options.config.motor_ticks_per_target = parse_size(value, argument);
    } else if (argument == "--memory-noise") {
      options.config.memory_noise_standard_deviation =
          parse_finite(value, argument);
    } else if (argument == "--memory-recurrence") {
      options.config.memory_recurrent_weight_gain =
          parse_finite(value, argument);
    } else if (argument == "--primacy-recurrence") {
      options.config.primacy_recurrent_weight_gain =
          parse_finite(value, argument);
    } else if (argument == "--recency-recurrence") {
      options.config.recency_recurrent_weight_gain =
          parse_finite(value, argument);
    } else if (argument == "--counting-delay") {
      options.config.counting_delay_ticks = parse_size(value, argument);
    } else if (argument == "--learned-digit-checkpoint") {
      options.learned_digit_checkpoint = value;
    } else if (argument == "--trace-json") {
      options.trace_json = value;
    } else if (argument == "--minimum-accuracy") {
      options.minimum_accuracy = parse_probability(value, argument);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  return options;
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

void write_digits(std::ostream& output, const std::vector<int>& digits) {
  output << '[';
  for (std::size_t index = 0; index < digits.size(); ++index) {
    output << (index == 0 ? "" : ",") << digits[index];
  }
  output << ']';
}

void write_trace(const std::filesystem::path& path,
                 const std::vector<snnbase_experiments::spaun::Result>& results) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("could not open trace output: " + path.string());
  }
  output << "{\n  \"format\": \"snnbase-spaun-trace-v1\",\n  \"results\": [\n";
  for (std::size_t result_index = 0; result_index < results.size();
       ++result_index) {
    const auto& result = results[result_index];
    output << "    {\"task\":" << static_cast<int>(result.task)
           << ",\"name\":\"" << json_escape(result.task_name)
           << "\",\"stimuli\":\"" << json_escape(result.stimulus_stream)
           << "\",\"expected\":";
    write_digits(output, result.expected);
    output << ",\"output\":";
    write_digits(output, result.output);
    output << ",\"correct\":" << (result.correct ? "true" : "false")
           << ",\"frames\":[\n";
    for (std::size_t frame_index = 0; frame_index < result.frames.size();
         ++frame_index) {
      const auto& frame = result.frames[frame_index];
      output << "      {\"tick\":" << frame.tick << ",\"time\":"
             << std::fixed << std::setprecision(4) << frame.time_seconds
             << ",\"stimulus\":\""
             << json_escape(std::string(1, frame.stimulus))
             << "\",\"recognized\":\""
             << json_escape(std::string(1, frame.recognized))
             << "\",\"phase\":\"" << json_escape(frame.phase)
             << "\",\"memory\":\"" << json_escape(frame.working_memory)
             << "\",\"output\":\"" << json_escape(frame.output)
             << "\",\"action\":\"" << json_escape(frame.selected_action)
             << "\",\"reward\":" << frame.reward
             << ",\"ink_points\":" << frame.pen_trace.size()
             << ",\"activity\":[";
      for (std::size_t module = 0; module < frame.activity.size(); ++module) {
        output << (module == 0 ? "" : ",") << frame.activity[module];
      }
      output << "],\"spikes\":[";
      for (std::size_t module = 0; module < frame.spikes.size(); ++module) {
        output << (module == 0 ? "" : ",") << frame.spikes[module];
      }
      output << "],\"arm\":{\"shoulder\":" << frame.arm.shoulder
             << ",\"elbow\":" << frame.arm.elbow << ",\"elbow_x\":"
             << frame.arm.elbow_x << ",\"elbow_y\":" << frame.arm.elbow_y
             << ",\"pen_x\":" << frame.arm.pen_x << ",\"pen_y\":"
             << frame.arm.pen_y << ",\"pen_down\":"
             << (frame.arm.pen_down ? "true" : "false") << "}}"
             << (frame_index + 1 == result.frames.size() ? "\n" : ",\n");
    }
    output << "    ]}" << (result_index + 1 == results.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    snnbase_experiments::spaun::Model model(options.config);
    if (!options.learned_digit_checkpoint.empty()) {
      model.set_learned_digit_checkpoint(options.learned_digit_checkpoint);
    }
    std::vector<snnbase_experiments::spaun::Result> results;
    if (options.all) {
      for (const auto task : snnbase_experiments::spaun::all_tasks()) {
        results.push_back(
            model.run(snnbase_experiments::spaun::canonical_trial(task)));
      }
    } else {
      results.push_back(
          model.run(snnbase_experiments::spaun::canonical_trial(options.task)));
    }

    std::size_t correct = 0;
    std::cout << "architecture=snnbase-scaled-spaun"
              << " neurons=" << model.neuron_count()
              << " synapses=" << model.synapse_count()
              << " neurons_per_module=" << options.config.neurons_per_module
              << " seed=" << options.config.seed << '\n'
              << "task,name,stimuli,expected,output,correct,spikes,simulated_seconds,frames\n";
    for (const auto& result : results) {
      correct += result.correct ? 1U : 0U;
      std::cout << 'A' << static_cast<int>(result.task) << ',' << result.task_name
                << ',' << result.stimulus_stream << ','
                << snnbase_experiments::spaun::digits_string(result.expected)
                << ',' << snnbase_experiments::spaun::digits_string(result.output)
                << ',' << std::boolalpha << result.correct << ','
                << result.total_spikes << ',' << std::fixed
                << std::setprecision(3) << result.simulated_seconds << ','
                << result.frames.size() << '\n';
    }
    const auto accuracy = static_cast<double>(correct) /
                          static_cast<double>(results.size());
    std::cout << "summary_correct=" << correct << '/' << results.size()
              << " accuracy=" << std::fixed << std::setprecision(2)
              << accuracy * 100.0 << "%\n";
    if (!options.trace_json.empty()) {
      write_trace(options.trace_json, results);
      std::cout << "trace_json=" << options.trace_json.string() << '\n';
    }
    return accuracy >= options.minimum_accuracy ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "spaun_experiment: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
