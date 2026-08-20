#include <snnbase_experiments/cifar10.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

snnbase_experiments::cifar10::Image solid(std::uint8_t red,
                                          std::uint8_t green,
                                          std::uint8_t blue) {
  snnbase_experiments::cifar10::Image image;
  constexpr auto plane =
      snnbase_experiments::cifar10::rows * snnbase_experiments::cifar10::columns;
  for (std::size_t index = 0; index < plane; ++index) {
    image.pixels[index] = red;
    image.pixels[plane + index] = green;
    image.pixels[2 * plane + index] = blue;
  }
  return image;
}

void write_record(std::ostream& output, std::uint8_t label,
                  const snnbase_experiments::cifar10::Image& image) {
  output.put(static_cast<char>(label));
  output.write(reinterpret_cast<const char*>(image.pixels.data()),
               static_cast<std::streamsize>(image.pixels.size()));
}

void test_loader_and_limit(const std::filesystem::path& directory) {
  const auto batch = directory / "data_batch_1.bin";
  std::ofstream output(batch, std::ios::binary);
  write_record(output, 2, solid(255, 0, 0));
  write_record(output, 7, solid(0, 0, 255));
  output.close();

  const auto all = snnbase_experiments::cifar10::load_binary_batch(batch);
  require(all.size() == 2, "loader returned the wrong sample count");
  require(all.labels[1] == 7, "loader returned the wrong label");

  const auto limited = snnbase_experiments::cifar10::load_binary_batch(batch, 1);
  require(limited.size() == 1, "loader did not apply its sample limit");
}

void test_encoding_and_classifier() {
  auto red_left = solid(0, 0, 0);
  auto blue_right = solid(0, 0, 0);
  constexpr auto plane =
      snnbase_experiments::cifar10::rows * snnbase_experiments::cifar10::columns;
  for (std::size_t row = 0; row < snnbase_experiments::cifar10::rows; ++row) {
    for (std::size_t column = 0; column < snnbase_experiments::cifar10::columns;
         ++column) {
      if (column < snnbase_experiments::cifar10::columns / 2) {
        red_left.pixels[row * snnbase_experiments::cifar10::columns + column] =
            255;
      } else {
        blue_right.pixels[2 * plane +
                          row * snnbase_experiments::cifar10::columns +
                          column] = 255;
      }
    }
  }

  const auto first = snnbase_experiments::cifar10::encode_image(red_left);
  const auto second = snnbase_experiments::cifar10::encode_image(blue_right);
  require(!first.empty(), "encoder produced an empty first event");
  require(!second.empty(), "encoder produced an empty second event");
  require(first.bits() != second.bits(), "encoder lost color or spatial data");

  snnbase_experiments::cifar10::Dataset training{
      {red_left, blue_right}, {1, 8}};
  snnbase_experiments::cifar10::Classifier classifier;
  classifier.train(training);
  const auto result = classifier.evaluate(training);
  require(result.correct == 2, "classifier did not learn its training examples");
  require(result.total == 2, "classifier reported the wrong evaluation count");
}

void test_invalid_label(const std::filesystem::path& directory) {
  const auto batch = directory / "bad_batch.bin";
  std::ofstream output(batch, std::ios::binary);
  write_record(output, 10, solid(0, 0, 0));
  output.close();

  bool threw = false;
  try {
    static_cast<void>(snnbase_experiments::cifar10::load_binary_batch(batch));
  } catch (const std::runtime_error&) {
    threw = true;
  }
  require(threw, "loader accepted an invalid label");
}

}  // namespace

int main() {
  const auto directory =
      std::filesystem::temp_directory_path() / "snnbase-cifar10-tests";
  std::filesystem::create_directories(directory);
  test_loader_and_limit(directory);
  test_encoding_and_classifier();
  test_invalid_label(directory);
  std::filesystem::remove_all(directory);
}
