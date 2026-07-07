#include <snnbase_experiments/mnist.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void write_u32(std::ostream& output, std::uint32_t value) {
  const std::array<char, 4> bytes{
      static_cast<char>((value >> 24U) & 0xffU),
      static_cast<char>((value >> 16U) & 0xffU),
      static_cast<char>((value >> 8U) & 0xffU),
      static_cast<char>(value & 0xffU)};
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_fixture(const std::filesystem::path& images_path,
                   const std::filesystem::path& labels_path) {
  std::ofstream images(images_path, std::ios::binary);
  write_u32(images, 2051);
  write_u32(images, 2);
  write_u32(images, 2);
  write_u32(images, 2);
  const std::array<std::uint8_t, 8> pixels{255, 0, 0, 0, 0, 0, 0, 255};
  images.write(reinterpret_cast<const char*>(pixels.data()),
               static_cast<std::streamsize>(pixels.size()));

  std::ofstream labels(labels_path, std::ios::binary);
  write_u32(labels, 2049);
  write_u32(labels, 2);
  const std::array<std::uint8_t, 2> values{1, 7};
  labels.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size()));
}

void test_loader_and_limit(const std::filesystem::path& directory) {
  const auto image_path = directory / "images";
  const auto label_path = directory / "labels";
  write_fixture(image_path, label_path);

  const auto all =
      snnbase_experiments::mnist::load_idx_dataset(image_path, label_path);
  require(all.size() == 2, "loader returned the wrong sample count");
  require(all.images[0].rows == 2, "loader returned the wrong row count");
  require(all.images[0].columns == 2, "loader returned the wrong column count");
  require(all.labels[1] == 7, "loader returned the wrong label");

  const auto limited =
      snnbase_experiments::mnist::load_idx_dataset(image_path, label_path, 1);
  require(limited.size() == 1, "loader did not apply its sample limit");
}

void test_encoding_and_classifier() {
  using snnbase_experiments::mnist::Image;
  const Image upper_left{2, 2, {255, 0, 0, 0}};
  const Image lower_right{2, 2, {0, 0, 0, 255}};
  const auto first = snnbase_experiments::mnist::encode_image(upper_left);
  const auto second = snnbase_experiments::mnist::encode_image(lower_right);
  require(!first.empty(), "encoder produced an empty first event");
  require(!second.empty(), "encoder produced an empty second event");
  require(first.bits() != second.bits(), "encoder lost spatial information");

  snnbase_experiments::mnist::Dataset training{
      {upper_left, lower_right}, {1, 7}};
  snnbase_experiments::mnist::Classifier classifier;
  classifier.train(training);
  const auto result = classifier.evaluate(training);
  require(result.correct == 2, "classifier did not learn its training examples");
  require(result.total == 2, "classifier reported the wrong evaluation count");
  require(result.accuracy() == 1.0, "classifier reported the wrong accuracy");
}

void test_invalid_image() {
  bool threw = false;
  try {
    static_cast<void>(snnbase_experiments::mnist::encode_image({2, 2, {1}}));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "encoder accepted malformed image data");
}

void test_invalid_idx(const std::filesystem::path& directory) {
  const auto image_path = directory / "bad-images";
  const auto label_path = directory / "bad-labels";
  {
    std::ofstream images(image_path, std::ios::binary);
    write_u32(images, 0);
  }
  {
    std::ofstream labels(label_path, std::ios::binary);
    write_u32(labels, 2049);
    write_u32(labels, 0);
  }

  bool threw = false;
  try {
    static_cast<void>(
        snnbase_experiments::mnist::load_idx_dataset(image_path, label_path));
  } catch (const std::runtime_error&) {
    threw = true;
  }
  require(threw, "loader accepted an invalid IDX header");
}

}  // namespace

int main() {
  const auto directory =
      std::filesystem::temp_directory_path() / "snnbase-experiments-tests";
  std::filesystem::create_directories(directory);
  test_loader_and_limit(directory);
  test_encoding_and_classifier();
  test_invalid_image();
  test_invalid_idx(directory);
  std::filesystem::remove_all(directory);
}
