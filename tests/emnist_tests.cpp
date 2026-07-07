#include <snnbase_experiments/emnist.hpp>

#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  using snnbase_experiments::mnist::Image;
  const Image upper_left{2, 2, {255, 0, 0, 0}};
  const Image lower_right{2, 2, {0, 0, 0, 255}};
  const snnbase_experiments::mnist::Dataset training{
      {upper_left, lower_right}, {1, 26}};

  snnbase_experiments::emnist::Classifier classifier{
      snnbase_experiments::emnist::letters};
  classifier.train(training);
  const auto result = classifier.evaluate(training);
  require(result.correct == 2, "EMNIST classifier failed label-offset test");

  snnbase_experiments::mnist::Image top_left{
      28, 28, std::vector<std::uint8_t>(28 * 28)};
  snnbase_experiments::mnist::Image bottom_right{
      28, 28, std::vector<std::uint8_t>(28 * 28)};
  for (std::size_t position = 3; position < 11; ++position) {
    top_left.pixels[4 * 28 + position] = 255;
    bottom_right.pixels[position * 28 + 23] = 255;
  }
  snnbase_experiments::mnist::Dataset structured_training;
  for (std::size_t repetition = 0; repetition < 20; ++repetition) {
    structured_training.images.push_back(top_left);
    structured_training.labels.push_back(1);
    structured_training.images.push_back(bottom_right);
    structured_training.labels.push_back(26);
  }
  snnbase_experiments::emnist::StructuredClassifier structured{
      snnbase_experiments::emnist::letters};
  structured.train(structured_training);
  const auto structured_result = structured.evaluate(structured_training);
  require(structured_result.correct == structured_result.total,
          "structured classifier failed its training examples");

  require(snnbase_experiments::emnist::find_split("balanced").class_count == 47,
          "balanced split metadata is incorrect");

  bool threw = false;
  try {
    static_cast<void>(
        snnbase_experiments::emnist::find_split("not-a-split"));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "unknown split was accepted");
}
