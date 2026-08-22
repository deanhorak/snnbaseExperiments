#include <snnbase_experiments/nmnist.hpp>

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

struct Options {
  std::filesystem::path data_dir{"data/nmnist"};
  std::size_t train_limit{};
  std::size_t test_limit{};
  std::size_t epochs{1};
  std::size_t time_bins{10};
  float novelty{0.78F};
  float learning_rate{0.35F};
  float minimum_accuracy{};
};

void usage(std::ostream& output, std::string_view program) {
  output << "Usage: " << program << " [options]\n"
         << "  --data-dir PATH       N-MNIST root containing Train/ and Test/\n"
         << "  --train-limit N       Train on at most N recordings (0 means all)\n"
         << "  --test-limit N        Test at most N recordings (0 means all)\n"
         << "  --epochs N            Training passes (default: 1)\n"
         << "  --time-bins N         Timestamp-preserving event bins (default: 10)\n"
         << "  --novelty F           Prototype novelty threshold in [0,1]\n"
         << "  --learning-rate F     Reward learning rate\n"
         << "  --minimum-accuracy F  Fail below accuracy F in [0,1]\n"
         << "  --help                Show this help\n";
}

std::size_t parse_size(std::string_view value, std::string_view option,
                       bool allow_zero = true) {
  std::size_t result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() || (!allow_zero && result == 0)) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return result;
}

float parse_float(std::string_view value, std::string_view option) {
  float result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() || !(result >= 0.0F)) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return result;
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
    if (argument == "--data-dir") options.data_dir = value;
    else if (argument == "--train-limit") options.train_limit = parse_size(value, argument);
    else if (argument == "--test-limit") options.test_limit = parse_size(value, argument);
    else if (argument == "--epochs") options.epochs = parse_size(value, argument, false);
    else if (argument == "--time-bins") options.time_bins = parse_size(value, argument, false);
    else if (argument == "--novelty") {
      options.novelty = parse_float(value, argument);
      if (options.novelty > 1.0F) throw std::invalid_argument("invalid value for --novelty");
    } else if (argument == "--learning-rate") {
      options.learning_rate = parse_float(value, argument);
      if (!(options.learning_rate > 0.0F)) throw std::invalid_argument("invalid value for --learning-rate");
    } else if (argument == "--minimum-accuracy") {
      options.minimum_accuracy = parse_float(value, argument);
      if (options.minimum_accuracy > 1.0F) throw std::invalid_argument("invalid value for --minimum-accuracy");
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    const auto train = snnbase_experiments::nmnist::load_dataset(options.data_dir, true, options.train_limit);
    const auto test = snnbase_experiments::nmnist::load_dataset(options.data_dir, false, options.test_limit);
    snnbase_experiments::nmnist::Classifier classifier{
        options.time_bins, options.novelty, options.learning_rate};
    classifier.train(train, options.epochs);
    const auto evaluation = classifier.evaluate(test);
    std::cout << "dataset=n-mnist architecture=timestamp-binned-prototype-snn"
              << " train_samples=" << train.size() << " test_samples=" << test.size()
              << " time_bins=" << classifier.time_bins()
              << " event_bits=" << snnbase::spike_event_bits
              << " payload_bits=" << snnbase::SpikeEvent::payload_width()
              << " accuracy=" << std::fixed << std::setprecision(2)
              << evaluation.accuracy() * 100.0 << "%\n";
    if (evaluation.accuracy() < options.minimum_accuracy) {
      std::cerr << "accuracy is below required minimum " << options.minimum_accuracy << '\n';
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "nmnist_experiment: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
