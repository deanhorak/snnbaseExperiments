#include <snnbase_experiments/spaun_vision_benchmark.hpp>

#include <chrono>
#include <stdexcept>
#include <utility>

namespace snnbase_experiments::spaun::vision_benchmark {
namespace {

std::vector<snnbase::spiking_conv::SampleView> views(
    const mnist::Dataset& dataset) {
  if (dataset.images.size() != dataset.labels.size()) {
    throw std::invalid_argument("MNIST images and labels differ in length");
  }
  std::vector<snnbase::spiking_conv::SampleView> result;
  result.reserve(dataset.size());
  for (std::size_t index = 0; index < dataset.size(); ++index) {
    const auto& image = dataset.images[index];
    result.push_back(
        {{image.rows, image.columns, image.pixels}, dataset.labels[index]});
  }
  return result;
}

void validate(const Config& config) {
  if (config.data_directory.empty()) {
    throw std::invalid_argument("A1 benchmark requires an MNIST directory");
  }
  if (config.epochs == 0 || config.batch_size == 0 ||
      config.time_steps == 0 || !(config.learning_rate > 0.0F)) {
    throw std::invalid_argument("invalid A1 training configuration");
  }
  if (config.evaluate_only && config.load_checkpoint.empty()) {
    throw std::invalid_argument(
        "A1 evaluate-only mode requires a registered checkpoint");
  }
}

}  // namespace

snnbase::spiking_conv::ModelConfig spaun_a1_model() {
  return {.input_rows = 28,
          .input_columns = 28,
          .input_channels = 1,
          .class_count = 10,
          .label_offset = 0,
          .first_convolution = {12, 5, 2, false},
          .second_convolution = {24, 3, 2, false},
          .neuron_threshold = 0.5F,
          .channel_normalization = false,
          .second_residual = snnbase::spiking_conv::ResidualMerge::none};
}

snnbase::spiking_conv::TrainingConfig training_config(const Config& config) {
  return {.epochs = config.epochs,
          .batch_size = config.batch_size,
          .time_steps = config.time_steps,
          .learning_rate = config.learning_rate,
          .seed = config.seed,
          .augment = config.augment};
}

Result run(const Config& config) {
  validate(config);
  const auto training_data = mnist::load_idx_dataset(
      config.data_directory / "train-images-idx3-ubyte",
      config.data_directory / "train-labels-idx1-ubyte", config.train_limit);
  const auto test_data = mnist::load_idx_dataset(
      config.data_directory / "t10k-images-idx3-ubyte",
      config.data_directory / "t10k-labels-idx1-ubyte", config.test_limit);
  const auto training_views = views(training_data);
  const auto test_views = views(test_data);
  const auto model = spaun_a1_model();
  const auto training = training_config(config);
  snnbase::spiking_conv::Classifier classifier(model, training);
  if (!config.load_checkpoint.empty()) {
    classifier.load_checkpoint(config.load_checkpoint);
  }

  Result result{.model = model,
                .training = training,
                .training_samples = training_data.size(),
                .test_samples = test_data.size(),
                .parameters = classifier.parameter_count(),
                .epochs = {},
                .evaluation = {},
                .training_seconds = 0.0,
                .evaluation_seconds = 0.0};
  const auto train_begin = std::chrono::steady_clock::now();
  if (!config.evaluate_only) {
    result.epochs = classifier.train(training_views);
  }
  const auto train_end = std::chrono::steady_clock::now();
  if (!config.save_checkpoint.empty()) {
    classifier.save_checkpoint(config.save_checkpoint);
  }
  result.evaluation = classifier.evaluate(test_views);
  const auto evaluation_end = std::chrono::steady_clock::now();
  result.training_seconds =
      std::chrono::duration<double>(train_end - train_begin).count();
  result.evaluation_seconds =
      std::chrono::duration<double>(evaluation_end - train_end).count();
  return result;
}

}  // namespace snnbase_experiments::spaun::vision_benchmark
