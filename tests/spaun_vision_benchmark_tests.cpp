#include <snnbase_experiments/spaun_vision_benchmark.hpp>

#include <stdexcept>

namespace {

void require(const bool condition) {
  if (!condition) {
    throw std::runtime_error("Spaun A1 vision benchmark invariant failed");
  }
}

}  // namespace

int main() {
  using namespace snnbase_experiments::spaun::vision_benchmark;
  const auto model = spaun_a1_model();
  require(model.input_rows == 28);
  require(model.input_columns == 28);
  require(model.class_count == 10);
  require(model.first_convolution.output_channels == 12);
  require(model.second_convolution.output_channels == 24);
  Config selected;
  selected.data_directory = "unused";
  selected.epochs = 5;
  selected.batch_size = 64;
  selected.time_steps = 8;
  selected.learning_rate = 0.001F;
  selected.seed = 42;
  const auto training = training_config(selected);
  require(training.epochs == 5);
  require(training.time_steps == 8);
  require(training.augment);

  bool threw = false;
  try {
    Config invalid;
    invalid.data_directory = "unused";
    invalid.evaluate_only = true;
    static_cast<void>(run(invalid));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw);
}
