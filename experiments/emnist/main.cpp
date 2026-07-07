#include <snnbase_experiments/emnist.hpp>
#include <snnbase_experiments/project_config.hpp>

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  std::filesystem::path data_dir{snnbase_experiments::config::emnist_data_dir};
  std::vector<const snnbase_experiments::emnist::Split*> splits;
  std::size_t train_limit{};
  std::size_t test_limit{};
  std::size_t epochs{1};
  float threshold{1.30F};
  float novelty{0.78F};
  float learning_rate{0.35F};
  float minimum_accuracy{};
  bool structured{true};
};

void usage(std::ostream& output, std::string_view program) {
  output
      << "Usage: " << program << " [options]\n"
      << "  --data-dir PATH       Directory containing uncompressed EMNIST IDX files\n"
      << "  --splits LIST         Comma-separated splits (default: all six)\n"
      << "  --architecture NAME   structured or baseline (default: structured)\n"
      << "  --train-limit N       Train samples per split (0 means all)\n"
      << "  --test-limit N        Test samples per split (0 means all)\n"
      << "  --epochs N            Training passes (default: 1)\n"
      << "  --threshold FLOAT     Pixel threshold multiplier (default: 1.30)\n"
      << "  --novelty FLOAT       Prototype novelty threshold (default: 0.78)\n"
      << "  --learning-rate FLOAT Reward learning rate (default: 0.35)\n"
      << "  --minimum-accuracy F  Fail below accuracy F, in the range 0-1\n"
      << "  --help                Show this help\n";
}

std::size_t parse_size(std::string_view value, std::string_view option,
                       bool allow_zero = true) {
  std::size_t parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      (!allow_zero && parsed == 0)) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return parsed;
}

float parse_float(std::string_view value, std::string_view option,
                  bool probability = false) {
  float parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      !(parsed > 0.0F) || (probability && parsed > 1.0F)) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return parsed;
}

std::vector<const snnbase_experiments::emnist::Split*> parse_splits(
    std::string_view value) {
  std::vector<const snnbase_experiments::emnist::Split*> result;
  while (!value.empty()) {
    const auto separator = value.find(',');
    const auto name = value.substr(0, separator);
    if (name.empty()) {
      throw std::invalid_argument("empty EMNIST split name");
    }
    result.push_back(&snnbase_experiments::emnist::find_split(name));
    if (separator == std::string_view::npos) {
      break;
    }
    value.remove_prefix(separator + 1);
  }
  return result;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (const auto& split : snnbase_experiments::emnist::standard_splits()) {
    options.splits.push_back(&split);
  }
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
    if (argument == "--data-dir") {
      options.data_dir = value;
    } else if (argument == "--splits") {
      options.splits = parse_splits(value);
    } else if (argument == "--architecture") {
      if (value == "structured") {
        options.structured = true;
      } else if (value == "baseline") {
        options.structured = false;
      } else {
        throw std::invalid_argument("invalid architecture: " +
                                    std::string(value));
      }
    } else if (argument == "--train-limit") {
      options.train_limit = parse_size(value, argument);
    } else if (argument == "--test-limit") {
      options.test_limit = parse_size(value, argument);
    } else if (argument == "--epochs") {
      options.epochs = parse_size(value, argument, false);
    } else if (argument == "--threshold") {
      options.threshold = parse_float(value, argument);
    } else if (argument == "--novelty") {
      options.novelty = parse_float(value, argument, true);
    } else if (argument == "--learning-rate") {
      options.learning_rate = parse_float(value, argument);
    } else if (argument == "--minimum-accuracy") {
      options.minimum_accuracy = parse_float(value, argument, true);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  return options;
}

double elapsed_seconds(std::chrono::steady_clock::time_point start,
                       std::chrono::steady_clock::time_point finish) {
  return std::chrono::duration<double>(finish - start).count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    std::cout << "architecture="
              << (options.structured ? "structured" : "baseline")
              << " event_bits=" << snnbase::spike_event_bits
              << " payload_bits=" << snnbase::SpikeEvent::payload_width()
              << " queue_depth=" << snnbase::history_length
              << " threshold=" << options.threshold
              << " novelty=" << options.novelty
              << " learning_rate=" << options.learning_rate
              << " epochs=" << options.epochs << "\n\n"
              << "split,classes,train_samples,test_samples,train_seconds,"
                 "test_seconds,correct,accuracy_percent\n";

    for (const auto* split : options.splits) {
      auto training = snnbase_experiments::emnist::load_split(
          options.data_dir, *split, true, options.train_limit);
      auto test = snnbase_experiments::emnist::load_split(
          options.data_dir, *split, false, options.test_limit);
      const auto train_start = std::chrono::steady_clock::now();
      snnbase_experiments::mnist::Evaluation evaluation;
      std::chrono::steady_clock::time_point train_finish;
      if (options.structured) {
        snnbase_experiments::emnist::StructuredClassifier classifier{
            *split, options.novelty, options.learning_rate};
        classifier.train(training, options.epochs);
        train_finish = std::chrono::steady_clock::now();
        evaluation = classifier.evaluate(test);
      } else {
        snnbase_experiments::emnist::Classifier classifier{
            *split, options.novelty, options.learning_rate};
        classifier.train(training, options.epochs, options.threshold);
        train_finish = std::chrono::steady_clock::now();
        evaluation = classifier.evaluate(test, options.threshold);
      }
      const auto test_finish = std::chrono::steady_clock::now();

      std::cout << split->name << ',' << split->class_count << ','
                << training.size() << ',' << test.size() << ',' << std::fixed
                << std::setprecision(3)
                << elapsed_seconds(train_start, train_finish) << ','
                << elapsed_seconds(train_finish, test_finish) << ','
                << evaluation.correct << ',' << std::setprecision(2)
                << evaluation.accuracy() * 100.0 << '\n';
      if (evaluation.accuracy() < options.minimum_accuracy) {
        std::cerr << split->name << " accuracy " << evaluation.accuracy()
                  << " is below required minimum "
                  << options.minimum_accuracy << '\n';
        return EXIT_FAILURE;
      }
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "emnist_battery: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
