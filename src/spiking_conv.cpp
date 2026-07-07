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
      : classifier(
            {.class_count = split.class_count,
             .label_offset = split.label_offset},
            training) {}

  snnbase::spiking_conv::Classifier classifier;
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

std::size_t Classifier::neuron_count() const noexcept {
  return impl_->classifier.neurons().size();
}

}  // namespace snnbase_experiments::spiking_conv
