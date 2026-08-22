#include <snnbase_experiments/project_config.hpp>
#include <snnbase_experiments/spaun_vision_benchmark.hpp>

#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using Config = snnbase_experiments::spaun::vision_benchmark::Config;

std::size_t size_value(const std::string_view value,
                       const std::string_view option, const bool zero = false) {
  std::size_t parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      (!zero && parsed == 0)) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return parsed;
}

float float_value(const std::string_view value,
                  const std::string_view option) {
  float parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      !(parsed > 0.0F)) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return parsed;
}

void usage(const std::string_view program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --data-dir PATH          Registered MNIST IDX directory\n"
      << "  --epochs N               Training epochs (default 5)\n"
      << "  --batch-size N            Mini-batch size (default 64)\n"
      << "  --time-steps N            Spike-rate timesteps (default 8)\n"
      << "  --learning-rate F         Adam learning rate (default .001)\n"
      << "  --train-limit N           Training limit; 0 means all\n"
      << "  --test-limit N            Test limit; 0 means all\n"
      << "  --seed N                  Deterministic seed (default 42)\n"
      << "  --load-checkpoint PATH    Load registered model state\n"
      << "  --save-checkpoint PATH    Save model/optimizer state\n"
      << "  --evaluate-only           Do not train; checkpoint required\n"
      << "  --no-augment              Disable +/-1 pixel translations\n"
      << "  --minimum-accuracy F      Exit nonzero below this proportion\n"
      << "  --help                    Show this text\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Config config;
    config.data_directory = snnbase_experiments::config::mnist_data_dir;
    double minimum_accuracy = 0.94;
    for (int index = 1; index < argc; ++index) {
      const std::string_view option(argv[index]);
      if (option == "--help") {
        usage(argv[0]);
        return EXIT_SUCCESS;
      }
      if (option == "--evaluate-only") {
        config.evaluate_only = true;
        continue;
      }
      if (option == "--no-augment") {
        config.augment = false;
        continue;
      }
      if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " +
                                    std::string(option));
      }
      const std::string_view value(argv[++index]);
      if (option == "--data-dir") {
        config.data_directory = value;
      } else if (option == "--epochs") {
        config.epochs = size_value(value, option);
      } else if (option == "--batch-size") {
        config.batch_size = size_value(value, option);
      } else if (option == "--time-steps") {
        config.time_steps = size_value(value, option);
      } else if (option == "--learning-rate") {
        config.learning_rate = float_value(value, option);
      } else if (option == "--train-limit") {
        config.train_limit = size_value(value, option, true);
      } else if (option == "--test-limit") {
        config.test_limit = size_value(value, option, true);
      } else if (option == "--seed") {
        config.seed = static_cast<std::uint32_t>(size_value(value, option, true));
      } else if (option == "--load-checkpoint") {
        config.load_checkpoint = value;
      } else if (option == "--save-checkpoint") {
        config.save_checkpoint = value;
      } else if (option == "--minimum-accuracy") {
        minimum_accuracy = float_value(value, option);
        if (minimum_accuracy > 1.0) {
          throw std::invalid_argument("minimum accuracy must be <= 1");
        }
      } else {
        throw std::invalid_argument("unknown option: " + std::string(option));
      }
    }

    const auto result =
        snnbase_experiments::spaun::vision_benchmark::run(config);
    std::cout << "benchmark=Spaun-A1-registered-MNIST"
              << " train_samples=" << result.training_samples
              << " test_samples=" << result.test_samples
              << " parameters=" << result.parameters
              << " conv1=12x5x5/2"
              << " conv2=24x3x3/2"
              << " time_steps=" << result.training.time_steps << '\n';
    for (const auto& epoch : result.epochs) {
      std::cout << "epoch=" << epoch.epoch << " loss=" << std::fixed
                << std::setprecision(5) << epoch.loss
                << " accuracy=" << std::setprecision(4) << epoch.accuracy
                << " seconds=" << std::setprecision(3) << epoch.seconds
                << '\n';
    }
    std::cout << "test_correct=" << result.evaluation.correct << '/'
              << result.evaluation.total << " test_accuracy=" << std::fixed
              << std::setprecision(4) << result.evaluation.accuracy()
              << " test_loss=" << std::setprecision(5)
              << result.evaluation.loss
              << " train_seconds=" << std::setprecision(3)
              << result.training_seconds
              << " evaluation_seconds=" << result.evaluation_seconds << '\n';
    if (result.evaluation.accuracy() < minimum_accuracy) {
      std::cerr << "A1 accuracy is below registered gate "
                << minimum_accuracy << '\n';
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "spaun_vision_benchmark: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
