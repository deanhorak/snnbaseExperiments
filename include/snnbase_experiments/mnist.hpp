#ifndef SNNBASE_EXPERIMENTS_MNIST_HPP
#define SNNBASE_EXPERIMENTS_MNIST_HPP

#include <snnbase/snnbase.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

namespace snnbase_experiments::mnist {

inline constexpr std::size_t class_count = 10;

struct Image {
  std::size_t rows{};
  std::size_t columns{};
  std::vector<std::uint8_t> pixels;
};

struct Dataset {
  std::vector<Image> images;
  std::vector<std::uint8_t> labels;

  [[nodiscard]] std::size_t size() const noexcept;
};

struct Evaluation {
  std::size_t correct{};
  std::size_t total{};
  std::array<std::array<std::size_t, class_count>, class_count> confusion{};

  [[nodiscard]] double accuracy() const noexcept;
};

[[nodiscard]] Dataset load_idx_dataset(
    const std::filesystem::path& image_file,
    const std::filesystem::path& label_file,
    std::size_t limit = 0,
    std::uint8_t maximum_label = class_count - 1);

[[nodiscard]] snnbase::SpikeEvent encode_image(
    const Image& image,
    float threshold_multiplier = 1.0F);

class Classifier {
 public:
  explicit Classifier(float novelty_threshold = 0.78F,
                      float reward_learning_rate = 0.35F);

  void train(const Dataset& dataset, std::size_t epochs = 1,
             float threshold_multiplier = 1.0F);
  [[nodiscard]] std::pair<std::uint8_t, float> predict(
      const Image& image, float threshold_multiplier = 1.0F) const;
  [[nodiscard]] Evaluation evaluate(
      const Dataset& dataset, float threshold_multiplier = 1.0F) const;
  [[nodiscard]] std::span<const snnbase::Neuron, class_count> neurons() const noexcept;

 private:
  std::array<snnbase::Neuron, class_count> neurons_;
};

}  // namespace snnbase_experiments::mnist

#endif
