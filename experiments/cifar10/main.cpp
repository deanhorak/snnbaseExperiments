#include <snnbase_experiments/cifar10.hpp>
#include <snnbase_experiments/project_config.hpp>

#include <snnbase/spiking_conv.hpp>

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
  std::filesystem::path data_dir{snnbase_experiments::config::cifar10_data_dir};
  std::filesystem::path load_checkpoint;
  std::filesystem::path save_checkpoint;
  std::size_t train_limit{};
  std::size_t test_limit{};
  std::size_t epochs{10};
  std::size_t batch_size{64};
  std::size_t time_steps{8};
  std::size_t channels{16};
  float learning_rate{0.001F};
  float neuron_threshold{0.5F};
  float minimum_accuracy{};
  bool augment{true};
  bool normalize{true};
  snnbase::spiking_conv::ResidualMerge residual{
      snnbase::spiking_conv::ResidualMerge::sew_add};
};

void usage(std::ostream& output, std::string_view program) {
  output << "Usage: " << program << " [options]\n"
         << "  --data-dir PATH       Directory containing CIFAR-10 .bin files\n"
         << "  --epochs N            Training epochs (default: 10)\n"
         << "  --batch-size N        Optimizer batch size (default: 64)\n"
         << "  --time-steps N        Rate-code timesteps (default: 8)\n"
         << "  --channels N          Convolution channels per stage (default: 16)\n"
         << "  --learning-rate F     Adam learning rate (default: 0.001)\n"
         << "  --neuron-threshold F  Spiking activation threshold (default: 0.5)\n"
         << "  --residual NAME       none, add, sew-add, or sew-multiply (default: sew-add)\n"
         << "  --train-limit N       Train on at most N samples (0 means all)\n"
         << "  --test-limit N        Evaluate at most N samples (0 means all)\n"
         << "  --minimum-accuracy F  Fail if accuracy is below F, in the range 0-1\n"
         << "  --load-checkpoint P   Load model/optimizer state before training\n"
         << "  --save-checkpoint P   Save model/optimizer state after training\n"
         << "  --no-augment          Disable deterministic one-pixel translation augmentation\n"
         << "  --no-normalize        Disable per-channel activation normalization\n"
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

snnbase::spiking_conv::ResidualMerge parse_residual(std::string_view value) {
  if (value == "none") {
    return snnbase::spiking_conv::ResidualMerge::none;
  }
  if (value == "add") {
    return snnbase::spiking_conv::ResidualMerge::add;
  }
  if (value == "sew-add") {
    return snnbase::spiking_conv::ResidualMerge::sew_add;
  }
  if (value == "sew-multiply") {
    return snnbase::spiking_conv::ResidualMerge::sew_multiply;
  }
  throw std::invalid_argument("invalid residual mode: " + std::string(value));
}

std::string_view residual_name(snnbase::spiking_conv::ResidualMerge value) {
  switch (value) {
    case snnbase::spiking_conv::ResidualMerge::none:
      return "none";
    case snnbase::spiking_conv::ResidualMerge::add:
      return "add";
    case snnbase::spiking_conv::ResidualMerge::sew_add:
      return "sew-add";
    case snnbase::spiking_conv::ResidualMerge::sew_multiply:
      return "sew-multiply";
  }
  return "unknown";
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
      options.augment = false;
      continue;
    }
    if (argument == "--no-normalize") {
      options.normalize = false;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(argument));
    }
    const std::string_view value(argv[++index]);
    if (argument == "--data-dir") {
      options.data_dir = value;
    } else if (argument == "--load-checkpoint") {
      options.load_checkpoint = value;
    } else if (argument == "--save-checkpoint") {
      options.save_checkpoint = value;
    } else if (argument == "--train-limit") {
      options.train_limit = parse_size(value, argument, true);
    } else if (argument == "--test-limit") {
      options.test_limit = parse_size(value, argument, true);
    } else if (argument == "--epochs") {
      options.epochs = parse_size(value, argument);
    } else if (argument == "--batch-size") {
      options.batch_size = parse_size(value, argument);
    } else if (argument == "--time-steps") {
      options.time_steps = parse_size(value, argument);
    } else if (argument == "--channels") {
      options.channels = parse_size(value, argument);
    } else if (argument == "--learning-rate") {
      options.learning_rate = parse_float(value, argument);
    } else if (argument == "--neuron-threshold") {
      options.neuron_threshold = parse_float(value, argument);
    } else if (argument == "--residual") {
      options.residual = parse_residual(value);
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

std::vector<snnbase::spiking_conv::SampleView> samples(
    const snnbase_experiments::cifar10::Dataset& dataset) {
  if (dataset.images.size() != dataset.labels.size()) {
    throw std::invalid_argument("image and label counts differ");
  }
  std::vector<snnbase::spiking_conv::SampleView> result;
  result.reserve(dataset.size());
  for (std::size_t index = 0; index < dataset.size(); ++index) {
    result.push_back(
        {{snnbase_experiments::cifar10::rows,
          snnbase_experiments::cifar10::columns,
          snnbase_experiments::cifar10::channels,
          dataset.images[index].pixels},
         dataset.labels[index]});
  }
  return result;
}

void print_confusion(const snnbase_experiments::mnist::Evaluation& result) {
  std::cout << "\nconfusion matrix (rows=actual, columns=predicted)\n    ";
  for (std::size_t label = 0; label < snnbase_experiments::cifar10::class_count;
       ++label) {
    std::cout << std::setw(6) << label;
  }
  std::cout << '\n';
  for (std::size_t actual = 0;
       actual < snnbase_experiments::cifar10::class_count; ++actual) {
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
    const auto start = std::chrono::steady_clock::now();
    const auto train = snnbase_experiments::cifar10::load_binary_dataset(
        options.data_dir, true, options.train_limit);
    const auto test = snnbase_experiments::cifar10::load_binary_dataset(
        options.data_dir, false, options.test_limit);
    const auto train_samples = samples(train);
    const auto test_samples = samples(test);
    const auto load_finish = std::chrono::steady_clock::now();

    snnbase::spiking_conv::ModelConfig model{
        .input_rows = snnbase_experiments::cifar10::rows,
        .input_columns = snnbase_experiments::cifar10::columns,
        .input_channels = snnbase_experiments::cifar10::channels,
        .class_count = snnbase_experiments::cifar10::class_count,
        .first_convolution = {options.channels, 3, 1, true},
        .second_convolution = {options.channels, 3, 1, true},
        .neuron_threshold = options.neuron_threshold,
        .channel_normalization = options.normalize,
        .second_residual = options.residual};
    snnbase::spiking_conv::TrainingConfig training{
        .epochs = options.epochs,
        .batch_size = options.batch_size,
        .time_steps = options.time_steps,
        .learning_rate = options.learning_rate,
        .augment = options.augment};
    snnbase::spiking_conv::Classifier classifier{model, training};
    if (!options.load_checkpoint.empty()) {
      classifier.load_checkpoint(options.load_checkpoint);
    }

    std::cout << "dataset=cifar-10"
              << " architecture=snnbase-spiking-conv"
              << " train_samples=" << train.size()
              << " test_samples=" << test.size()
              << " event_bits=" << snnbase::spike_event_bits
              << " payload_bits=" << snnbase::SpikeEvent::payload_width()
              << " queue_depth=" << snnbase::history_length
              << " channels=" << options.channels
              << " residual=" << residual_name(options.residual)
              << " normalize=" << (options.normalize ? "true" : "false")
              << " augment=" << (options.augment ? "true" : "false")
              << " neuron_threshold=" << options.neuron_threshold
              << " learning_rate=" << options.learning_rate
              << " batch_size=" << options.batch_size
              << " time_steps=" << options.time_steps
              << " epochs=" << options.epochs
              << " parameters=" << classifier.parameter_count()
              << " snnbase_neurons=" << classifier.neurons().size() << '\n';

    const auto train_begin = std::chrono::steady_clock::now();
    const auto metrics = classifier.train(train_samples);
    const auto train_finish = std::chrono::steady_clock::now();
    for (const auto& epoch : metrics) {
      std::cout << "epoch=" << epoch.epoch << " loss=" << std::fixed
                << std::setprecision(5) << epoch.loss
                << " train_accuracy=" << std::setprecision(2)
                << epoch.accuracy * 100.0 << "%"
                << " seconds=" << std::setprecision(3) << epoch.seconds
                << '\n';
    }

    const auto deep_evaluation = classifier.evaluate(test_samples);
    snnbase_experiments::mnist::Evaluation evaluation;
    evaluation.correct = deep_evaluation.correct;
    evaluation.total = deep_evaluation.total;
    for (std::size_t index = 0; index < test.size(); ++index) {
      const auto predicted = classifier.predict(test_samples[index].image);
      ++evaluation.confusion[test.labels[index]][predicted];
    }
    const auto test_finish = std::chrono::steady_clock::now();
    if (!options.save_checkpoint.empty()) {
      classifier.save_checkpoint(options.save_checkpoint);
    }

    std::cout << "load_seconds=" << std::fixed << std::setprecision(3)
              << elapsed_seconds(start, load_finish)
              << " train_seconds=" << elapsed_seconds(train_begin, train_finish)
              << " test_seconds=" << elapsed_seconds(train_finish, test_finish)
              << "\ncorrect=" << evaluation.correct << '/' << evaluation.total
              << "\naccuracy=" << std::setprecision(2)
              << evaluation.accuracy() * 100.0 << "%"
              << "\ntest_loss=" << std::setprecision(5)
              << deep_evaluation.loss << '\n';
    print_confusion(evaluation);
    if (evaluation.accuracy() < options.minimum_accuracy) {
      std::cerr << "accuracy " << evaluation.accuracy()
                << " is below required minimum " << options.minimum_accuracy
                << '\n';
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "cifar10_experiment: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
