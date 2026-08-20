#include <snnbase_experiments/spiking_conv.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

namespace snnbase_experiments::spiking_conv {
namespace {

snnbase::spiking_conv::ImageView view(const mnist::Image& image) {
  return {image.rows, image.columns, image.pixels};
}

std::vector<snnbase::spiking_conv::SampleView> samples(
    const mnist::Dataset& dataset) {
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

}  // namespace

struct Classifier::Impl {
  explicit Impl(const emnist::Split& split, TrainingConfig training)
      : classifier(make_model(split), make_training(std::move(training))) {}

  static snnbase::temporal::ModelConfig make_model(const emnist::Split& split) {
    return {.input_rows = 28,
            .input_columns = 28,
            .input_channels = 1,
            .class_count = split.class_count,
            .label_offset = split.label_offset,
            .stem_channels = 16,
            .stages = {{16, 2, 1}, {32, 2, 2}, {64, 2, 2}},
            .initial_threshold = 1.0F,
            .initial_leak = 0.90F,
            .surrogate_slope = 4.0F,
            .readout_population = 2,
            .channel_mean = {0.1736F},
            .channel_stddev = {0.3317F}};
  }

  static TrainingConfig make_training(TrainingConfig training) {
    training.crop_padding = 2;
    training.horizontal_flip = false;
    training.cutout_size = 0;
    return training;
  }

  snnbase::temporal::Classifier classifier;
};

Classifier::Classifier(const emnist::Split& split, TrainingConfig config)
    : impl_(std::make_unique<Impl>(split, config)) {}

Classifier::~Classifier() = default;
Classifier::Classifier(Classifier&&) noexcept = default;
Classifier& Classifier::operator=(Classifier&&) noexcept = default;

std::vector<EpochMetrics> Classifier::train(const mnist::Dataset& dataset) {
  const auto dataset_samples = samples(dataset);
  return impl_->classifier.train(dataset_samples);
}

Evaluation Classifier::evaluate(const mnist::Dataset& dataset) const {
  const auto dataset_samples = samples(dataset);
  return impl_->classifier.evaluate(dataset_samples);
}

std::uint8_t Classifier::predict(const mnist::Image& image) const {
  return static_cast<std::uint8_t>(impl_->classifier.predict(view(image)));
}

std::size_t Classifier::parameter_count() const noexcept {
  return impl_->classifier.parameter_count();
}

std::string Classifier::device() const {
  return impl_->classifier.device();
}

}  // namespace snnbase_experiments::spiking_conv
