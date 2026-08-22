#include <snnbase/temporal_resnet.hpp>
#include <snnbase_experiments/cifar10.hpp>
#include <snnbase_experiments/project_config.hpp>

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  std::filesystem::path data_dir{snnbase_experiments::config::cifar10_data_dir};
  std::filesystem::path load_checkpoint;
  std::filesystem::path save_checkpoint;
  std::filesystem::path save_best_checkpoint;
  std::size_t train_limit{};
  std::size_t test_limit{};
  std::size_t width{32};
  std::size_t blocks{2};
  std::size_t readout_population{2};
  float threshold{1.0F};
  float leak{0.90F};
  float surrogate_slope{4.0F};
  float minimum_accuracy{};
  snnbase::temporal::TrainingConfig training{
      .epochs = 100,
      .batch_size = 128,
      .time_steps = 4,
      .learning_rate = 0.001F,
      .minimum_learning_rate = 1.0e-5F,
      .weight_decay = 5.0e-4F,
      .warmup_epochs = 5,
      .label_smoothing = 0.1F,
      .spike_rate_target = 0.15F,
      .spike_rate_penalty = 1.0e-4F,
      .validation_fraction = 0.1F,
      .seed = 42,
      .augment = true,
      .crop_padding = 4,
      .horizontal_flip = true,
      .cutout_size = 8,
      .device = "auto"};
  bool normalization{true};
  bool bntt{false};
  bool evaluate_only{false};
  bool validation_only{false};
};

void usage(std::ostream& output, std::string_view program) {
  output << "Usage: " << program << " [options]\n"
         << "  --data-dir PATH          Directory containing CIFAR-10 .bin files\n"
         << "  --epochs N               Training epochs (default: 100)\n"
         << "  --batch-size N           Optimizer batch size (default: 128)\n"
         << "  --time-steps N           LIF timesteps (default: 4)\n"
         << "  --learning-rate F        Adam learning rate (default: 0.001)\n"
         << "  --width N                Residual stage channel width (default: 32)\n"
         << "  --blocks N               Blocks per residual stage (default: 2)\n"
         << "  --threshold F            Initial LIF threshold (default: 1.0)\n"
         << "  --leak F                 Initial LIF leak in (0,1) (default: 0.9)\n"
         << "  --surrogate-slope F      Sigmoid surrogate slope (default: 4.0)\n"
         << "  --readout-population N   Readout neurons per class (default: 2)\n"
         << "  --weight-decay F         AdamW weight decay (default: 0.0005)\n"
         << "  --label-smoothing F      Cross-entropy smoothing [0,1) (default: 0.1)\n"
         << "  --warmup-epochs N        Linear warmup epochs (default: 5)\n"
         << "  --cutout-size N          Cutout side length (default: 8)\n"
         << "  --encoding NAME          direct or rate (default: direct)\n"
         << "  --train-limit N          Training sample limit (default: all)\n"
         << "  --test-limit N           Test sample limit (default: all)\n"
         << "  --load-checkpoint PATH   Resume model/AdamW/epoch state\n"
         << "  --save-checkpoint PATH   Save model/AdamW/epoch state\n"
         << "  --save-best-checkpoint PATH\n"
         << "                           Save whenever validation accuracy improves\n"
         << "  --evaluate-only          Skip training; requires a checkpoint\n"
         << "  --validation-only        Train and report validation; skip test set\n"
         << "  --minimum-accuracy F     Fail below accuracy F, in the range 0-1\n"
         << "  --seed N                 Training/data-split seed (default: 42)\n"
         << "  --device NAME            auto, cpu, or cuda (default: auto)\n"
         << "  --no-augment             Disable crop/flip/Cutout augmentation\n"
         << "  --no-normalization       Disable temporal batch normalization\n"
         << "  --bntt                   Use independent batch normalization per timestep\n"
         << "  --black-padding          Use black rather than zero-centered crop padding\n"
         << "  --help                   Show this help\n";
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

float parse_nonnegative_float(std::string_view value,
                              std::string_view option) {
  float parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      parsed < 0.0F) {
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
    if (argument == "--no-normalization") {
      options.normalization = false;
      continue;
    }
    if (argument == "--bntt") {
      options.bntt = true;
      continue;
    }
    if (argument == "--black-padding") {
      options.training.mean_padding = false;
      continue;
    }
    if (argument == "--evaluate-only") {
      options.evaluate_only = true;
      continue;
    }
    if (argument == "--validation-only") {
      options.validation_only = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(argument));
    }
    const std::string_view value(argv[++index]);
    if (argument == "--data-dir") {
      options.data_dir = value;
    } else if (argument == "--epochs") {
      options.training.epochs = parse_size(value, argument);
    } else if (argument == "--batch-size") {
      options.training.batch_size = parse_size(value, argument);
    } else if (argument == "--time-steps") {
      options.training.time_steps = parse_size(value, argument);
    } else if (argument == "--learning-rate") {
      options.training.learning_rate = parse_float(value, argument);
    } else if (argument == "--width") {
      options.width = parse_size(value, argument);
    } else if (argument == "--blocks") {
      options.blocks = parse_size(value, argument);
    } else if (argument == "--threshold") {
      options.threshold = parse_float(value, argument);
    } else if (argument == "--leak") {
      options.leak = parse_float(value, argument);
      if (options.leak >= 1.0F) {
        throw std::invalid_argument("invalid value for " +
                                    std::string(argument));
      }
    } else if (argument == "--surrogate-slope") {
      options.surrogate_slope = parse_float(value, argument);
    } else if (argument == "--readout-population") {
      options.readout_population = parse_size(value, argument);
    } else if (argument == "--weight-decay") {
      options.training.weight_decay = parse_nonnegative_float(value, argument);
    } else if (argument == "--label-smoothing") {
      options.training.label_smoothing = parse_nonnegative_float(value, argument);
      if (options.training.label_smoothing >= 1.0F) {
        throw std::invalid_argument("invalid value for " +
                                    std::string(argument));
      }
    } else if (argument == "--warmup-epochs") {
      options.training.warmup_epochs = parse_size(value, argument, true);
    } else if (argument == "--cutout-size") {
      options.training.cutout_size = parse_size(value, argument, true);
    } else if (argument == "--encoding") {
      if (value == "direct") {
        options.training.input_encoding =
            snnbase::temporal::InputEncoding::direct_current;
      } else if (value == "rate") {
        options.training.input_encoding =
            snnbase::temporal::InputEncoding::deterministic_rate;
      } else {
        throw std::invalid_argument("invalid value for " +
                                    std::string(argument));
      }
    } else if (argument == "--train-limit") {
      options.train_limit = parse_size(value, argument, true);
    } else if (argument == "--test-limit") {
      options.test_limit = parse_size(value, argument, true);
    } else if (argument == "--load-checkpoint") {
      options.load_checkpoint = value;
    } else if (argument == "--save-checkpoint") {
      options.save_checkpoint = value;
    } else if (argument == "--save-best-checkpoint") {
      options.save_best_checkpoint = value;
    } else if (argument == "--seed") {
      options.training.seed = parse_size(value, argument, true);
    } else if (argument == "--device") {
      options.training.device = value;
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
  if (options.evaluate_only && options.load_checkpoint.empty()) {
    throw std::invalid_argument("--evaluate-only requires --load-checkpoint");
  }
  if (options.evaluate_only && options.validation_only) {
    throw std::invalid_argument(
        "--evaluate-only and --validation-only are mutually exclusive");
  }
  return options;
}

snnbase::spiking_conv::ImageView view(
    const snnbase_experiments::cifar10::Image& image) {
  return {snnbase_experiments::cifar10::rows,
          snnbase_experiments::cifar10::columns,
          snnbase_experiments::cifar10::channels,
          image.pixels};
}

std::vector<snnbase::spiking_conv::SampleView> samples(
    const snnbase_experiments::cifar10::Dataset& dataset) {
  if (dataset.images.size() != dataset.labels.size()) {
    throw std::invalid_argument("image and label counts differ");
  }
  std::vector<snnbase::spiking_conv::SampleView> result;
  result.reserve(dataset.size());
  for (std::size_t index = 0; index < dataset.size(); ++index) {
    result.push_back({view(dataset.images[index]), dataset.labels[index]});
  }
  return result;
}

void print_confusion(const snnbase::temporal::Evaluation& evaluation,
                     std::size_t class_count) {
  std::cout << "confusion_matrix_rows_actual_cols_predicted\n";
  for (std::size_t actual = 0; actual < class_count; ++actual) {
    std::cout << actual;
    for (std::size_t predicted = 0; predicted < class_count; ++predicted) {
      std::cout << ','
                << evaluation.confusion[actual * class_count + predicted];
    }
    std::cout << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    const auto training_data = snnbase_experiments::cifar10::load_binary_dataset(
        options.data_dir, true, options.train_limit);
    snnbase_experiments::cifar10::Dataset test_data;
    if (!options.validation_only) {
      test_data = snnbase_experiments::cifar10::load_binary_dataset(
          options.data_dir, false, options.test_limit);
    }
    const auto training_samples = samples(training_data);
    const auto test_samples = samples(test_data);

    snnbase::temporal::Classifier classifier{
        {.input_rows = snnbase_experiments::cifar10::rows,
         .input_columns = snnbase_experiments::cifar10::columns,
         .input_channels = snnbase_experiments::cifar10::channels,
         .class_count = snnbase_experiments::cifar10::class_count,
         .stem_channels = options.width,
         .stages = {{options.width, options.blocks, 1},
                    {options.width * 2, options.blocks, 2},
                    {options.width * 4, options.blocks, 2}},
         .initial_threshold = options.threshold,
         .initial_leak = options.leak,
         .surrogate_slope = options.surrogate_slope,
         .learn_thresholds = true,
         .learn_leaks = true,
         .temporal_batch_normalization = options.normalization,
         .batch_normalization_through_time = options.bntt,
         .readout_population = options.readout_population,
         .channel_mean = {0.4914F, 0.4822F, 0.4465F},
         .channel_stddev = {0.2470F, 0.2435F, 0.2616F}},
        options.training};

    classifier.set_best_checkpoint_path(options.save_best_checkpoint);

    if (!options.load_checkpoint.empty()) {
      classifier.load_checkpoint(options.load_checkpoint);
    }

    std::cout << "dataset=cifar-10"
              << " architecture=temporal-sew-resnet"
              << " train_samples=" << training_samples.size()
              << " test_samples=" << test_samples.size()
              << " width=" << options.width
              << " blocks_per_stage=" << options.blocks
              << " normalization=" << std::boolalpha << options.normalization
              << " bntt=" << options.bntt
              << " timesteps=" << options.training.time_steps
              << " threshold=" << options.threshold
              << " leak=" << options.leak
              << " surrogate_slope=" << options.surrogate_slope
              << " readout_population=" << options.readout_population
              << " encoding="
              << (options.training.input_encoding ==
                          snnbase::temporal::InputEncoding::direct_current
                      ? "direct"
                      : "rate")
              << " learning_rate=" << options.training.learning_rate
              << " weight_decay=" << options.training.weight_decay
              << " label_smoothing=" << options.training.label_smoothing
              << " cutout_size=" << options.training.cutout_size
              << " mean_padding=" << options.training.mean_padding
              << " seed=" << options.training.seed
              << " best_checkpoint="
              << options.save_best_checkpoint.string()
              << " device=" << classifier.device()
              << " parameters=" << classifier.parameter_count()
              << '\n';

    const auto train_start = std::chrono::steady_clock::now();
    const auto epochs = options.evaluate_only
                            ? std::vector<snnbase::temporal::EpochMetrics>{}
                            : classifier.train(training_samples);
    const auto train_finish = std::chrono::steady_clock::now();
    for (const auto& epoch : epochs) {
      std::cout << "epoch=" << epoch.epoch << " lr=" << std::scientific
                << epoch.learning_rate << " train_loss=" << std::fixed
                << std::setprecision(5) << epoch.training_loss
                << " train_accuracy=" << std::setprecision(2)
                << epoch.training_accuracy * 100.0 << "%"
                << " validation_loss=" << std::setprecision(5)
                << epoch.validation_loss
                << " validation_accuracy=" << std::setprecision(2)
                << epoch.validation_accuracy * 100.0 << "%"
                << " spike_rate=" << std::setprecision(4) << epoch.spike_rate
                << " seconds=" << std::setprecision(3) << epoch.seconds
                << '\n';
    }

    double best_validation_accuracy = -std::numeric_limits<double>::infinity();
    std::size_t best_validation_epoch = 0;
    for (const auto& epoch : epochs) {
      if (epoch.validation_accuracy > best_validation_accuracy) {
        best_validation_accuracy = epoch.validation_accuracy;
        best_validation_epoch = epoch.epoch;
      }
    }
    if (!epochs.empty()) {
      std::cout << "best_validation_epoch=" << best_validation_epoch
                << " best_validation_accuracy=" << std::fixed
                << std::setprecision(2) << best_validation_accuracy * 100.0
                << "%\n";
    }

    if (!options.save_checkpoint.empty()) {
      classifier.save_checkpoint(options.save_checkpoint);
    }

    if (options.validation_only) {
      std::cout << "train_seconds=" << std::fixed << std::setprecision(3)
                << std::chrono::duration<double>(train_finish - train_start)
                       .count()
                << '\n';
      if (best_validation_accuracy < options.minimum_accuracy) {
        std::cerr << "validation accuracy is below required minimum "
                  << options.minimum_accuracy << '\n';
        return EXIT_FAILURE;
      }
      return EXIT_SUCCESS;
    }

    const auto evaluation = classifier.evaluate(test_samples);
    const auto test_finish = std::chrono::steady_clock::now();
    std::cout << "test_correct=" << evaluation.correct << '/'
              << evaluation.total << " test_accuracy=" << std::fixed
              << std::setprecision(2) << evaluation.accuracy() * 100.0
              << "% test_loss=" << std::setprecision(5) << evaluation.loss
              << " test_spike_rate=" << std::setprecision(4)
              << evaluation.spike_rate
              << "\ntrain_seconds=" << std::setprecision(3)
              << std::chrono::duration<double>(train_finish - train_start)
                     .count()
              << " test_seconds="
              << std::chrono::duration<double>(test_finish - train_finish)
                     .count()
              << '\n';
    print_confusion(evaluation, snnbase_experiments::cifar10::class_count);
    if (evaluation.accuracy() < options.minimum_accuracy) {
      std::cerr << "accuracy is below required minimum "
                << options.minimum_accuracy << '\n';
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "cifar10_deep: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
