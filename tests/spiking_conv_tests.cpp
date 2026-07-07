#include <snnbase_experiments/spiking_conv.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

snnbase_experiments::mnist::Image horizontal() {
  snnbase_experiments::mnist::Image image{
      28, 28, std::vector<std::uint8_t>(28 * 28)};
  for (std::size_t row = 12; row < 16; ++row) {
    for (std::size_t column = 5; column < 23; ++column) {
      image.pixels[row * 28 + column] = 255;
    }
  }
  return image;
}

snnbase_experiments::mnist::Image vertical() {
  snnbase_experiments::mnist::Image image{
      28, 28, std::vector<std::uint8_t>(28 * 28)};
  for (std::size_t row = 5; row < 23; ++row) {
    for (std::size_t column = 12; column < 16; ++column) {
      image.pixels[row * 28 + column] = 255;
    }
  }
  return image;
}

}  // namespace

int main() {
  const snnbase_experiments::emnist::Split binary{"binary", 2, 0};
  snnbase_experiments::mnist::Dataset dataset;
  for (std::size_t repetition = 0; repetition < 20; ++repetition) {
    dataset.images.push_back(horizontal());
    dataset.labels.push_back(0);
    dataset.images.push_back(vertical());
    dataset.labels.push_back(1);
  }

  snnbase_experiments::spiking_conv::Classifier classifier{
      binary,
      {.epochs = 3,
       .batch_size = 8,
       .time_steps = 4,
       .learning_rate = 0.003F,
       .seed = 7,
       .augment = false}};
  require(classifier.parameter_count() == 4130,
          "unexpected trainable synapse count");
  require(classifier.neuron_count() == 2330,
          "unexpected snnbase neuron count");
  const auto epochs = classifier.train(dataset);
  require(epochs.size() == 3, "wrong epoch metric count");
  const auto evaluation = classifier.evaluate(dataset);
  require(evaluation.accuracy() >= 0.95,
          "deep spiking classifier did not learn separable shapes");
}
