#include <snnbase_experiments/mnist.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace snnbase_experiments::mnist {
namespace {

constexpr std::uint32_t image_magic = 2051;
constexpr std::uint32_t label_magic = 2049;

std::uint32_t read_u32_be(std::istream& input, const std::filesystem::path& path) {
  std::array<unsigned char, 4> bytes{};
  if (!input.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("truncated IDX header: " + path.string());
  }
  return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
         (static_cast<std::uint32_t>(bytes[1]) << 16U) |
         (static_cast<std::uint32_t>(bytes[2]) << 8U) |
         static_cast<std::uint32_t>(bytes[3]);
}

std::ifstream open_binary(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open " + path.string());
  }
  return input;
}

std::size_t checked_image_size(std::uint32_t rows, std::uint32_t columns) {
  if (rows == 0 || columns == 0 ||
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::runtime_error("invalid IDX image dimensions");
  }
  return static_cast<std::size_t>(rows) * columns;
}

}  // namespace

std::size_t Dataset::size() const noexcept {
  return images.size();
}

double Evaluation::accuracy() const noexcept {
  return total == 0 ? 0.0
                    : static_cast<double>(correct) / static_cast<double>(total);
}

Dataset load_idx_dataset(const std::filesystem::path& image_file,
                         const std::filesystem::path& label_file,
                         std::size_t limit, std::uint8_t maximum_label) {
  auto images = open_binary(image_file);
  auto labels = open_binary(label_file);

  if (read_u32_be(images, image_file) != image_magic) {
    throw std::runtime_error("invalid IDX image magic number: " + image_file.string());
  }
  if (read_u32_be(labels, label_file) != label_magic) {
    throw std::runtime_error("invalid IDX label magic number: " + label_file.string());
  }

  const auto image_count = read_u32_be(images, image_file);
  const auto label_count = read_u32_be(labels, label_file);
  const auto rows = read_u32_be(images, image_file);
  const auto columns = read_u32_be(images, image_file);
  if (image_count != label_count) {
    throw std::runtime_error("IDX image and label counts differ");
  }

  const auto image_size = checked_image_size(rows, columns);
  const auto count = limit == 0 ? static_cast<std::size_t>(image_count)
                                : std::min<std::size_t>(image_count, limit);
  Dataset dataset;
  dataset.images.reserve(count);
  dataset.labels.resize(count);

  for (std::size_t index = 0; index < count; ++index) {
    Image image{rows, columns, std::vector<std::uint8_t>(image_size)};
    if (!images.read(reinterpret_cast<char*>(image.pixels.data()),
                     static_cast<std::streamsize>(image_size))) {
      throw std::runtime_error("truncated IDX image data: " + image_file.string());
    }
    dataset.images.push_back(std::move(image));
  }
  if (!labels.read(reinterpret_cast<char*>(dataset.labels.data()),
                   static_cast<std::streamsize>(count))) {
    throw std::runtime_error("truncated IDX label data: " + label_file.string());
  }
  if (std::any_of(dataset.labels.begin(), dataset.labels.end(),
                  [maximum_label](std::uint8_t label) {
                    return label > maximum_label;
                  })) {
    throw std::runtime_error("IDX label exceeds the configured maximum");
  }
  return dataset;
}

snnbase::SpikeEvent encode_image(const Image& image, float threshold_multiplier) {
  if (image.rows == 0 || image.columns == 0 ||
      image.pixels.size() != image.rows * image.columns) {
    throw std::invalid_argument("image dimensions do not match its pixel data");
  }
  if (!(threshold_multiplier > 0.0F) || !std::isfinite(threshold_multiplier)) {
    throw std::invalid_argument("threshold multiplier must be finite and positive");
  }

  constexpr auto payload_width = snnbase::SpikeEvent::payload_width();
  static_assert(payload_width > 0, "MNIST encoding requires spike payload bits");
  const auto aspect = static_cast<double>(image.columns) /
                      static_cast<double>(image.rows);
  auto grid_columns = static_cast<std::size_t>(
      std::sqrt(static_cast<double>(payload_width) * aspect));
  grid_columns = std::clamp<std::size_t>(grid_columns, 1, payload_width);
  const auto grid_rows = std::max<std::size_t>(1, payload_width / grid_columns);
  const auto bins = grid_rows * grid_columns;

  std::vector<std::uint64_t> sums(bins);
  std::vector<std::size_t> counts(bins);
  std::uint64_t total = 0;
  for (std::size_t row = 0; row < image.rows; ++row) {
    for (std::size_t column = 0; column < image.columns; ++column) {
      const auto pixel = image.pixels[row * image.columns + column];
      const auto grid_row = std::min(grid_rows - 1, row * grid_rows / image.rows);
      const auto grid_column =
          std::min(grid_columns - 1, column * grid_columns / image.columns);
      const auto bin = grid_row * grid_columns + grid_column;
      sums[bin] += pixel;
      ++counts[bin];
      total += pixel;
    }
  }

  const auto global_mean = static_cast<double>(total) /
                           static_cast<double>(image.pixels.size());
  const auto threshold = global_mean * threshold_multiplier;
  snnbase::SpikeEvent::Storage pattern = 0;
  for (std::size_t bin = 0; bin < bins; ++bin) {
    if (counts[bin] == 0) {
      continue;
    }
    const auto cell_mean = static_cast<double>(sums[bin]) /
                           static_cast<double>(counts[bin]);
    if (cell_mean >= threshold && sums[bin] != 0) {
      pattern |= snnbase::SpikeEvent::payload_bit(bin);
    }
  }
  return snnbase::SpikeEvent(pattern);
}

Classifier::Classifier(float novelty_threshold, float reward_learning_rate) {
  if (!(novelty_threshold >= 0.0F && novelty_threshold <= 1.0F)) {
    throw std::invalid_argument("novelty threshold must be in the range 0-1");
  }
  if (!(reward_learning_rate > 0.0F) ||
      !std::isfinite(reward_learning_rate)) {
    throw std::invalid_argument("reward learning rate must be finite and positive");
  }
  for (std::size_t label = 0; label < neurons_.size(); ++label) {
    neurons_[label] = snnbase::Neuron{
        label,
        {.initial_stdp_relevance = 1.0F,
         .reward_learning_rate = reward_learning_rate,
         .novelty_threshold = novelty_threshold,
         .prototype_aggregation = snnbase::PrototypeAggregation::maximum,
         .replay_interval = 1000,
         .replay_batch_size = 4,
         .replay_reward = 0.05F}};
  }
}

void Classifier::train(const Dataset& dataset, std::size_t epochs,
                       float threshold_multiplier) {
  if (dataset.images.size() != dataset.labels.size()) {
    throw std::invalid_argument("training dataset image and label counts differ");
  }
  for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
    for (std::size_t index = 0; index < dataset.size(); ++index) {
      static_cast<void>(neurons_[dataset.labels[index]].reinforce(
          encode_image(dataset.images[index], threshold_multiplier), 1.0F));
    }
  }
}

std::pair<std::uint8_t, float> Classifier::predict(
    const Image& image, float threshold_multiplier) const {
  const auto event = encode_image(image, threshold_multiplier);
  std::uint8_t best_label = 0;
  float best_score = -1.0F;
  for (std::size_t label = 0; label < neurons_.size(); ++label) {
    const auto score = neurons_[label].evaluate(event);
    if (score > best_score) {
      best_label = static_cast<std::uint8_t>(label);
      best_score = score;
    }
  }
  return {best_label, best_score};
}

Evaluation Classifier::evaluate(const Dataset& dataset,
                                float threshold_multiplier) const {
  if (dataset.images.size() != dataset.labels.size()) {
    throw std::invalid_argument("evaluation dataset image and label counts differ");
  }
  Evaluation result;
  result.total = dataset.size();
  for (std::size_t index = 0; index < dataset.size(); ++index) {
    const auto actual = dataset.labels[index];
    const auto [predicted, score] =
        predict(dataset.images[index], threshold_multiplier);
    static_cast<void>(score);
    ++result.confusion[actual][predicted];
    result.correct += predicted == actual ? 1U : 0U;
  }
  return result;
}

std::span<const snnbase::Neuron, class_count> Classifier::neurons() const noexcept {
  return neurons_;
}

}  // namespace snnbase_experiments::mnist
