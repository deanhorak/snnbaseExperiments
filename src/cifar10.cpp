#include <snnbase_experiments/cifar10.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace snnbase_experiments::cifar10 {
namespace {

constexpr std::size_t record_bytes = image_bytes + 1;

std::ifstream open_binary(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open " + path.string());
  }
  return input;
}

void append_batch(const std::filesystem::path& batch_file, Dataset& dataset,
                  std::size_t remaining) {
  auto input = open_binary(batch_file);
  const auto before = input.tellg();
  input.seekg(0, std::ios::end);
  const auto after = input.tellg();
  input.seekg(before);
  if (before < 0 || after < 0 ||
      (static_cast<std::size_t>(after - before) % record_bytes) != 0) {
    throw std::runtime_error("invalid CIFAR-10 binary batch size: " +
                             batch_file.string());
  }

  const auto available =
      static_cast<std::size_t>(after - before) / record_bytes;
  const auto count = remaining == 0 ? available : std::min(available, remaining);
  dataset.images.reserve(dataset.images.size() + count);
  dataset.labels.reserve(dataset.labels.size() + count);
  for (std::size_t index = 0; index < count; ++index) {
    char label{};
    Image image;
    if (!input.read(&label, 1) ||
        !input.read(reinterpret_cast<char*>(image.pixels.data()),
                    static_cast<std::streamsize>(image.pixels.size()))) {
      throw std::runtime_error("truncated CIFAR-10 record: " +
                               batch_file.string());
    }
    const auto value = static_cast<std::uint8_t>(label);
    if (value >= class_count) {
      throw std::runtime_error("CIFAR-10 label exceeds 9: " +
                               batch_file.string());
    }
    dataset.labels.push_back(value);
    dataset.images.push_back(image);
  }
}

std::uint8_t pixel(const Image& image, std::size_t channel, std::size_t row,
                   std::size_t column) {
  return image.pixels[channel * rows * columns + row * columns + column];
}

double luminance(const Image& image, std::size_t row, std::size_t column) {
  return 0.299 * pixel(image, 0, row, column) +
         0.587 * pixel(image, 1, row, column) +
         0.114 * pixel(image, 2, row, column);
}

template <std::size_t Planes, std::size_t GridRows, std::size_t GridColumns>
snnbase::SpikeEvent encode_planes(
    const std::array<std::array<double, rows * columns>, Planes>& planes,
    float threshold_multiplier) {
  static_assert(Planes * GridRows * GridColumns <=
                snnbase::SpikeEvent::payload_width());
  std::array<std::array<double, GridRows * GridColumns>, Planes> sums{};
  std::array<std::array<std::uint16_t, GridRows * GridColumns>, Planes> counts{};
  std::array<double, Planes> totals{};
  for (std::size_t plane = 0; plane < Planes; ++plane) {
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t column = 0; column < columns; ++column) {
        const auto grid_row = std::min(GridRows - 1, row * GridRows / rows);
        const auto grid_column =
            std::min(GridColumns - 1, column * GridColumns / columns);
        const auto bin = grid_row * GridColumns + grid_column;
        const auto value = planes[plane][row * columns + column];
        sums[plane][bin] += value;
        totals[plane] += value;
        ++counts[plane][bin];
      }
    }
  }

  snnbase::SpikeEvent::Storage bits = 0;
  for (std::size_t plane = 0; plane < Planes; ++plane) {
    const auto mean = totals[plane] / static_cast<double>(rows * columns);
    const auto threshold = mean * threshold_multiplier;
    for (std::size_t bin = 0; bin < GridRows * GridColumns; ++bin) {
      const auto cell_mean =
          sums[plane][bin] / static_cast<double>(counts[plane][bin]);
      if (cell_mean >= threshold && sums[plane][bin] > 0.0) {
        bits |= snnbase::SpikeEvent::payload_bit(
            plane * GridRows * GridColumns + bin);
      }
    }
  }
  return snnbase::SpikeEvent(bits);
}

snnbase::SpikeEvent encode_color_intensity(const Image& image,
                                           float threshold_multiplier) {
  std::array<std::array<double, rows * columns>, channels> planes{};
  for (std::size_t channel = 0; channel < channels; ++channel) {
    for (std::size_t index = 0; index < rows * columns; ++index) {
      planes[channel][index] = image.pixels[channel * rows * columns + index];
    }
  }
  return encode_planes<3, 4, 8>(planes, threshold_multiplier);
}

snnbase::SpikeEvent encode_opponent(const Image& image,
                                    float threshold_multiplier) {
  std::array<std::array<double, rows * columns>, 3> planes{};
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      const auto red = static_cast<double>(pixel(image, 0, row, column));
      const auto green = static_cast<double>(pixel(image, 1, row, column));
      const auto blue = static_cast<double>(pixel(image, 2, row, column));
      const auto index = row * columns + column;
      planes[0][index] = 0.299 * red + 0.587 * green + 0.114 * blue;
      planes[1][index] = std::max(0.0, red - green + 128.0);
      planes[2][index] = std::max(0.0, (red + green) * 0.5 - blue + 128.0);
    }
  }
  return encode_planes<3, 4, 8>(planes, threshold_multiplier);
}

snnbase::SpikeEvent encode_sobel_orientation(const Image& image,
                                             float threshold_multiplier) {
  constexpr std::size_t orientation_count = 4;
  std::array<std::array<double, rows * columns>, orientation_count> planes{};
  for (std::size_t row = 1; row + 1 < rows; ++row) {
    for (std::size_t column = 1; column + 1 < columns; ++column) {
      const auto gx =
          -luminance(image, row - 1, column - 1) +
          luminance(image, row - 1, column + 1) -
          2.0 * luminance(image, row, column - 1) +
          2.0 * luminance(image, row, column + 1) -
          luminance(image, row + 1, column - 1) +
          luminance(image, row + 1, column + 1);
      const auto gy =
          -luminance(image, row - 1, column - 1) -
          2.0 * luminance(image, row - 1, column) -
          luminance(image, row - 1, column + 1) +
          luminance(image, row + 1, column - 1) +
          2.0 * luminance(image, row + 1, column) +
          luminance(image, row + 1, column + 1);
      const auto magnitude = std::hypot(gx, gy);
      auto angle = std::atan2(gy, gx);
      if (angle < 0.0) {
        angle += std::numbers::pi;
      }
      const auto orientation = std::min<std::size_t>(
          orientation_count - 1,
          static_cast<std::size_t>(angle * orientation_count /
                                   std::numbers::pi));
      planes[orientation][row * columns + column] = magnitude;
    }
  }
  return encode_planes<4, 4, 6>(planes, threshold_multiplier);
}

snnbase::SpikeEvent encode_center_surround(const Image& image,
                                           float threshold_multiplier) {
  std::array<std::array<double, rows * columns>, 2> planes{};
  for (std::size_t row = 1; row + 1 < rows; ++row) {
    for (std::size_t column = 1; column + 1 < columns; ++column) {
      double surround = 0.0;
      for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
          if (dr != 0 || dc != 0) {
            surround += luminance(image, row + dr, column + dc);
          }
        }
      }
      surround /= 8.0;
      const auto center = luminance(image, row, column);
      const auto index = row * columns + column;
      planes[0][index] = std::max(0.0, center - surround);
      planes[1][index] = std::max(0.0, surround - center);
    }
  }
  return encode_planes<2, 6, 8>(planes, threshold_multiplier);
}

snnbase::SpikeEvent encode_color_categories(const Image& image,
                                            float threshold_multiplier) {
  constexpr std::size_t category_count = 6;
  std::array<std::array<double, rows * columns>, category_count> planes{};
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      const auto red = static_cast<double>(pixel(image, 0, row, column));
      const auto green = static_cast<double>(pixel(image, 1, row, column));
      const auto blue = static_cast<double>(pixel(image, 2, row, column));
      const auto index = row * columns + column;
      planes[0][index] = red;
      planes[1][index] = green;
      planes[2][index] = blue;
      planes[3][index] = std::max(0.0, red - std::max(green, blue));
      planes[4][index] = std::max(0.0, green - std::max(red, blue));
      planes[5][index] = std::max(0.0, blue - std::max(red, green));
    }
  }
  return encode_planes<6, 4, 4>(planes, threshold_multiplier);
}

snnbase::SpikeEvent encode_quadrant_texture(const Image& image,
                                            float threshold_multiplier) {
  constexpr std::size_t plane_count = 4;
  std::array<std::array<double, rows * columns>, plane_count> planes{};
  for (std::size_t row = 1; row + 1 < rows; ++row) {
    for (std::size_t column = 1; column + 1 < columns; ++column) {
      const auto lum = luminance(image, row, column);
      const auto horizontal =
          std::abs(luminance(image, row, column - 1) -
                   luminance(image, row, column + 1));
      const auto vertical =
          std::abs(luminance(image, row - 1, column) -
                   luminance(image, row + 1, column));
      const auto chroma =
          std::max({std::abs(static_cast<double>(pixel(image, 0, row, column)) -
                             pixel(image, 1, row, column)),
                    std::abs(static_cast<double>(pixel(image, 0, row, column)) -
                             pixel(image, 2, row, column)),
                    std::abs(static_cast<double>(pixel(image, 1, row, column)) -
                             pixel(image, 2, row, column))});
      const auto index = row * columns + column;
      planes[0][index] = lum;
      planes[1][index] = horizontal;
      planes[2][index] = vertical;
      planes[3][index] = chroma;
    }
  }
  return encode_planes<4, 4, 6>(planes, threshold_multiplier);
}

}  // namespace

std::size_t Dataset::size() const noexcept {
  return images.size();
}

Dataset load_binary_batch(const std::filesystem::path& batch_file,
                          std::size_t limit) {
  Dataset dataset;
  append_batch(batch_file, dataset, limit);
  return dataset;
}

Dataset load_binary_dataset(const std::filesystem::path& data_dir,
                            bool training, std::size_t limit) {
  Dataset dataset;
  if (!training) {
    append_batch(data_dir / "test_batch.bin", dataset, limit);
    return dataset;
  }

  std::size_t remaining = limit;
  for (std::size_t batch = 1; batch <= 5; ++batch) {
    if (limit != 0 && remaining == 0) {
      break;
    }
    const auto before = dataset.size();
    append_batch(data_dir / ("data_batch_" + std::to_string(batch) + ".bin"),
                 dataset, remaining);
    if (limit != 0) {
      remaining -= dataset.size() - before;
    }
  }
  return dataset;
}

snnbase::SpikeEvent encode_image(const Image& image, float threshold_multiplier) {
  if (!(threshold_multiplier > 0.0F) || !std::isfinite(threshold_multiplier)) {
    throw std::invalid_argument("threshold multiplier must be finite and positive");
  }

  constexpr auto payload_width = snnbase::SpikeEvent::payload_width();
  static_assert(payload_width >= channels,
                "CIFAR-10 encoding requires at least one payload bit/channel");
  constexpr auto bins_per_channel = payload_width / channels;
  constexpr std::size_t grid_columns = 8;
  constexpr auto grid_rows = std::max<std::size_t>(1, bins_per_channel / grid_columns);
  constexpr auto bins = grid_rows * grid_columns;
  static_assert(bins * channels <= payload_width);

  std::array<std::array<std::uint64_t, bins>, channels> sums{};
  std::array<std::array<std::uint16_t, bins>, channels> counts{};
  std::array<std::uint64_t, channels> totals{};
  for (std::size_t channel = 0; channel < channels; ++channel) {
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t column = 0; column < columns; ++column) {
        const auto grid_row = std::min(grid_rows - 1, row * grid_rows / rows);
        const auto grid_column =
            std::min(grid_columns - 1, column * grid_columns / columns);
        const auto bin = grid_row * grid_columns + grid_column;
        const auto pixel =
            image.pixels[channel * rows * columns + row * columns + column];
        sums[channel][bin] += pixel;
        ++counts[channel][bin];
        totals[channel] += pixel;
      }
    }
  }

  snnbase::SpikeEvent::Storage pattern = 0;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto mean = static_cast<double>(totals[channel]) /
                      static_cast<double>(rows * columns);
    const auto threshold = mean * threshold_multiplier;
    for (std::size_t bin = 0; bin < bins; ++bin) {
      const auto cell_mean = static_cast<double>(sums[channel][bin]) /
                             static_cast<double>(counts[channel][bin]);
      if (cell_mean >= threshold && sums[channel][bin] != 0) {
        pattern |= snnbase::SpikeEvent::payload_bit(channel * bins + bin);
      }
    }
  }
  return snnbase::SpikeEvent(pattern);
}

std::array<snnbase::SpikeEvent, feature_bank_count> encode_feature_banks(
    const Image& image, float threshold_multiplier) {
  if (!(threshold_multiplier > 0.0F) || !std::isfinite(threshold_multiplier)) {
    throw std::invalid_argument("threshold multiplier must be finite and positive");
  }
  return {encode_color_intensity(image, threshold_multiplier),
          encode_opponent(image, threshold_multiplier),
          encode_sobel_orientation(image, threshold_multiplier),
          encode_center_surround(image, threshold_multiplier),
          encode_color_categories(image, threshold_multiplier),
          encode_quadrant_texture(image, threshold_multiplier)};
}

Classifier::Classifier(float novelty_threshold, float reward_learning_rate) {
  if (!(novelty_threshold >= 0.0F && novelty_threshold <= 1.0F)) {
    throw std::invalid_argument("novelty threshold must be in the range 0-1");
  }
  if (!(reward_learning_rate > 0.0F) ||
      !std::isfinite(reward_learning_rate)) {
    throw std::invalid_argument("reward learning rate must be finite and positive");
  }
  smoothing_ = 1.0F + reward_learning_rate + novelty_threshold;
}

void Classifier::train(const Dataset& dataset, std::size_t epochs,
                       float threshold_multiplier) {
  if (dataset.images.size() != dataset.labels.size()) {
    throw std::invalid_argument("training dataset image and label counts differ");
  }
  class_counts_ = {};
  active_counts_ = {};
  for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
    for (std::size_t index = 0; index < dataset.size(); ++index) {
      const auto features =
          encode_feature_banks(dataset.images[index], threshold_multiplier);
      const auto label = dataset.labels[index];
      ++class_counts_[label];
      for (std::size_t bank = 0; bank < feature_bank_count; ++bank) {
        for (std::size_t bit = 0; bit < snnbase::SpikeEvent::payload_width();
             ++bit) {
          if ((features[bank].bits() & snnbase::SpikeEvent::payload_bit(bit)) !=
              0) {
            ++active_counts_[label]
                            [bank * snnbase::SpikeEvent::payload_width() + bit];
          }
        }
      }
    }
  }
}

std::pair<std::uint8_t, float> Classifier::predict(
    const Image& image, float threshold_multiplier) const {
  const auto features = encode_feature_banks(image, threshold_multiplier);
  std::uint8_t best_label = 0;
  double best_score = -std::numeric_limits<double>::infinity();
  std::size_t total_count = 0;
  for (const auto count : class_counts_) {
    total_count += count;
  }
  for (std::size_t label = 0; label < class_count; ++label) {
    const auto class_count_value = class_counts_[label];
    auto score = std::log(
        (static_cast<double>(class_count_value) + smoothing_) /
        (static_cast<double>(total_count) + smoothing_ * class_count));
    for (std::size_t bank = 0; bank < feature_bank_count; ++bank) {
      for (std::size_t bit = 0; bit < snnbase::SpikeEvent::payload_width();
           ++bit) {
        const auto active =
            (features[bank].bits() & snnbase::SpikeEvent::payload_bit(bit)) != 0;
        const auto feature_index =
            bank * snnbase::SpikeEvent::payload_width() + bit;
        const auto probability =
            (static_cast<double>(active_counts_[label][feature_index]) +
             smoothing_) /
            (static_cast<double>(class_count_value) + 2.0 * smoothing_);
        score += active ? std::log(probability)
                        : std::log(std::max(1.0e-12, 1.0 - probability));
      }
    }
    if (score > best_score) {
      best_label = static_cast<std::uint8_t>(label);
      best_score = score;
    }
  }
  return {best_label, static_cast<float>(best_score)};
}

mnist::Evaluation Classifier::evaluate(const Dataset& dataset,
                                       float threshold_multiplier) const {
  if (dataset.images.size() != dataset.labels.size()) {
    throw std::invalid_argument("evaluation dataset image and label counts differ");
  }
  mnist::Evaluation result;
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

}  // namespace snnbase_experiments::cifar10
