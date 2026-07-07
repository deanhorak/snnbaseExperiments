#include <snnbase_experiments/mnist.hpp>
#include <snnbase_experiments/project_config.hpp>

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

struct Options {
  std::filesystem::path data_dir{snnbase_experiments::config::mnist_data_dir};
  std::size_t train_limit{0};
  std::size_t test_limit{0};
  std::size_t epochs{1};
  float threshold{1.30F};
  float novelty_threshold{0.78F};
  float learning_rate{0.35F};
  float minimum_accuracy{0.0F};
};

void usage(std::ostream& output, std::string_view program) {
  output << "Usage: " << program << " [options]\n"
         << "  --data-dir PATH       Directory containing uncompressed IDX files\n"
         << "  --train-limit N       Train on at most N samples (0 means all)\n"
         << "  --test-limit N        Evaluate at most N samples (0 means all)\n"
         << "  --epochs N            Number of training passes (default: 1)\n"
         << "  --threshold FLOAT     Pooled-pixel threshold multiplier (default: 1.3)\n"
         << "  --novelty FLOAT       Prototype novelty threshold, 0-1 (default: 0.78)\n"
         << "  --learning-rate FLOAT Reward learning rate (default: 0.35)\n"
         << "  --minimum-accuracy F  Fail if accuracy is below F, in the range 0-1\n"
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

float parse_float(std::string_view value, std::string_view option) {
  float parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      !(parsed > 0.0F)) {
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
    if (argument == "--data-dir") {
      options.data_dir = value;
    } else if (argument == "--train-limit") {
      options.train_limit = parse_size(value, argument);
    } else if (argument == "--test-limit") {
      options.test_limit = parse_size(value, argument);
    } else if (argument == "--epochs") {
      options.epochs = parse_size(value, argument, false);
    } else if (argument == "--threshold") {
      options.threshold = parse_float(value, argument);
    } else if (argument == "--novelty") {
      options.novelty_threshold = parse_float(value, argument);
      if (options.novelty_threshold > 1.0F) {
        throw std::invalid_argument("invalid value for " + std::string(argument));
      }
    } else if (argument == "--learning-rate") {
      options.learning_rate = parse_float(value, argument);
    } else if (argument == "--minimum-accuracy") {
      options.minimum_accuracy = parse_float(value, argument);
      if (options.minimum_accuracy > 1.0F) {
        throw std::invalid_argument("invalid value for " + std::string(argument));
      }
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  return options;
}

void print_confusion(const snnbase_experiments::mnist::Evaluation& result) {
  std::cout << "\nconfusion matrix (rows=actual, columns=predicted)\n    ";
  for (std::size_t label = 0; label < snnbase_experiments::mnist::class_count;
       ++label) {
    std::cout << std::setw(6) << label;
  }
  std::cout << '\n';
  for (std::size_t actual = 0;
       actual < snnbase_experiments::mnist::class_count; ++actual) {
    std::cout << std::setw(3) << actual << ' ';
    for (const auto count : result.confusion[actual]) {
      std::cout << std::setw(6) << count;
    }
    std::cout << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    const auto train = snnbase_experiments::mnist::load_idx_dataset(
        options.data_dir / "train-images-idx3-ubyte",
        options.data_dir / "train-labels-idx1-ubyte", options.train_limit);
    const auto test = snnbase_experiments::mnist::load_idx_dataset(
        options.data_dir / "t10k-images-idx3-ubyte",
        options.data_dir / "t10k-labels-idx1-ubyte", options.test_limit);

    std::cout << "training samples: " << train.size()
              << "\ntest samples: " << test.size()
              << "\nspike event bits: " << snnbase::spike_event_bits
              << "\npayload bits: " << snnbase::SpikeEvent::payload_width()
              << "\nprototype capacity/class: " << snnbase::history_length
              << "\nnovelty threshold: " << options.novelty_threshold
              << "\nlearning rate: " << options.learning_rate << '\n';

    snnbase_experiments::mnist::Classifier classifier{
        options.novelty_threshold, options.learning_rate};
    classifier.train(train, options.epochs, options.threshold);
    const auto evaluation = classifier.evaluate(test, options.threshold);

    std::cout << "\ncorrect: " << evaluation.correct << '/' << evaluation.total
              << "\naccuracy: " << std::fixed << std::setprecision(2)
              << evaluation.accuracy() * 100.0 << "%\n";
    print_confusion(evaluation);
    if (evaluation.accuracy() < options.minimum_accuracy) {
      std::cerr << "accuracy " << evaluation.accuracy()
                << " is below required minimum " << options.minimum_accuracy
                << '\n';
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "mnist_experiment: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
