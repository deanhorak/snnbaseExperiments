#include <snnbase_experiments/project_config.hpp>
#include <snnbase_experiments/spiking_conv.hpp>

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Options {
  std::filesystem::path data_dir{snnbase_experiments::config::emnist_data_dir};
  const snnbase_experiments::emnist::Split* split{
      &snnbase_experiments::emnist::mnist};
  std::size_t train_limit{};
  std::size_t test_limit{};
  float minimum_accuracy{};
  snnbase_experiments::spiking_conv::TrainingConfig training;
};

void usage(std::ostream& output, std::string_view program) {
  output << "Usage: " << program << " [options]\n"
         << "  --data-dir PATH       Uncompressed EMNIST IDX directory\n"
         << "  --split NAME          One EMNIST split (default: mnist)\n"
         << "  --epochs N            Training epochs (default: 5)\n"
         << "  --batch-size N        Optimizer batch size (default: 64)\n"
         << "  --time-steps N        Rate-code timesteps (default: 8)\n"
         << "  --learning-rate F     Adam learning rate (default: 0.001)\n"
         << "  --train-limit N       Training sample limit (default: all)\n"
         << "  --test-limit N        Test sample limit (default: all)\n"
         << "  --minimum-accuracy F  Fail below accuracy F, in the range 0-1\n"
         << "  --no-augment          Disable translation augmentation\n"
         << "  --help                Show this help\n";
}

std::size_t parse_size(std::string_view value, std::string_view option,
                       bool allow_zero = false) {
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
    if (argument == "--no-augment") {
      options.training.augment = false;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(argument));
    }
    const std::string_view value(argv[++index]);
    if (argument == "--data-dir") {
      options.data_dir = value;
    } else if (argument == "--split") {
      options.split = &snnbase_experiments::emnist::find_split(value);
    } else if (argument == "--epochs") {
      options.training.epochs = parse_size(value, argument);
    } else if (argument == "--batch-size") {
      options.training.batch_size = parse_size(value, argument);
    } else if (argument == "--time-steps") {
      options.training.time_steps = parse_size(value, argument);
    } else if (argument == "--learning-rate") {
      options.training.learning_rate = parse_float(value, argument);
    } else if (argument == "--train-limit") {
      options.train_limit = parse_size(value, argument, true);
    } else if (argument == "--test-limit") {
      options.test_limit = parse_size(value, argument, true);
    } else if (argument == "--minimum-accuracy") {
      options.minimum_accuracy = parse_float(value, argument);
      if (options.minimum_accuracy > 1.0F) {
        throw std::invalid_argument("invalid value for " +
                                    std::string(argument));
      }
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
    const auto training_data = snnbase_experiments::emnist::load_split(
        options.data_dir, *options.split, true, options.train_limit);
    const auto test_data = snnbase_experiments::emnist::load_split(
        options.data_dir, *options.split, false, options.test_limit);
    snnbase_experiments::spiking_conv::Classifier classifier{
        *options.split, options.training};

    std::cout << "split=" << options.split->name
              << " classes=" << options.split->class_count
              << " train_samples=" << training_data.size()
              << " test_samples=" << test_data.size()
              << " timesteps=" << options.training.time_steps
              << " parameters=" << classifier.parameter_count()
              << " snnbase_neurons=" << classifier.neuron_count() << '\n';

    const auto train_start = std::chrono::steady_clock::now();
    const auto epochs = classifier.train(training_data);
    const auto train_finish = std::chrono::steady_clock::now();
    for (const auto& epoch : epochs) {
      std::cout << "epoch=" << epoch.epoch << " loss=" << std::fixed
                << std::setprecision(5) << epoch.loss
                << " train_accuracy=" << std::setprecision(2)
                << epoch.accuracy * 100.0 << "%"
                << " seconds=" << std::setprecision(3) << epoch.seconds
                << '\n';
    }

    const auto evaluation = classifier.evaluate(test_data);
    const auto test_finish = std::chrono::steady_clock::now();
    std::cout << "test_correct=" << evaluation.correct << '/'
              << evaluation.total << " test_accuracy=" << std::fixed
              << std::setprecision(2) << evaluation.accuracy() * 100.0
              << "% test_loss=" << std::setprecision(5) << evaluation.loss
              << "\ntrain_seconds=" << std::setprecision(3)
              << std::chrono::duration<double>(train_finish - train_start)
                     .count()
              << " test_seconds="
              << std::chrono::duration<double>(test_finish - train_finish)
                     .count()
              << '\n';
    if (evaluation.accuracy() < options.minimum_accuracy) {
      std::cerr << "accuracy is below required minimum "
                << options.minimum_accuracy << '\n';
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "emnist_deep: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
