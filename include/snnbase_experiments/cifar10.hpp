#ifndef SNNBASE_EXPERIMENTS_CIFAR10_HPP
#define SNNBASE_EXPERIMENTS_CIFAR10_HPP

#include <snnbase/snnbase.hpp>
#include <snnbase_experiments/mnist.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace snnbase_experiments::cifar10 {

inline constexpr std::size_t class_count = 10;
inline constexpr std::size_t rows = 32;
inline constexpr std::size_t columns = 32;
inline constexpr std::size_t channels = 3;
inline constexpr std::size_t image_bytes = rows * columns * channels;
inline constexpr std::size_t feature_bank_count = 6;

struct Image {
  std::array<std::uint8_t, image_bytes> pixels{};
};

struct Dataset {
  std::vector<Image> images;
  std::vector<std::uint8_t> labels;

  [[nodiscard]] std::size_t size() const noexcept;
};

[[nodiscard]] Dataset load_binary_batch(const std::filesystem::path& batch_file,
                                        std::size_t limit = 0);
[[nodiscard]] Dataset load_binary_dataset(const std::filesystem::path& data_dir,
                                          bool training,
                                          std::size_t limit = 0);

[[nodiscard]] snnbase::SpikeEvent encode_image(
    const Image& image,
    float threshold_multiplier = 1.0F);
[[nodiscard]] std::array<snnbase::SpikeEvent, feature_bank_count>
encode_feature_banks(const Image& image, float threshold_multiplier = 1.0F);

class Classifier {
 public:
  explicit Classifier(float novelty_threshold = 0.72F,
                      float reward_learning_rate = 0.60F);

  void train(const Dataset& dataset, std::size_t epochs = 1,
             float threshold_multiplier = 1.0F);
  [[nodiscard]] std::pair<std::uint8_t, float> predict(
      const Image& image, float threshold_multiplier = 1.0F) const;
  [[nodiscard]] mnist::Evaluation evaluate(
      const Dataset& dataset, float threshold_multiplier = 1.0F) const;

 private:
  static constexpr std::size_t feature_bits =
      feature_bank_count * snnbase::SpikeEvent::payload_width();
  std::array<std::size_t, class_count> class_counts_{};
  std::array<std::array<std::uint32_t, feature_bits>, class_count>
      active_counts_{};
  float smoothing_{1.0F};
};

}  // namespace snnbase_experiments::cifar10

#endif
