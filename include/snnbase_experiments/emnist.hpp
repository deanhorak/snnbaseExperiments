#ifndef SNNBASE_EXPERIMENTS_EMNIST_HPP
#define SNNBASE_EXPERIMENTS_EMNIST_HPP

#include <snnbase/neuron.hpp>
#include <snnbase_experiments/mnist.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace snnbase_experiments::emnist {

struct Split {
  std::string_view name;
  std::size_t class_count;
  std::uint8_t label_offset;
};

inline constexpr Split byclass{"byclass", 62, 0};
inline constexpr Split bymerge{"bymerge", 47, 0};
inline constexpr Split balanced{"balanced", 47, 0};
inline constexpr Split letters{"letters", 26, 1};
inline constexpr Split digits{"digits", 10, 0};
inline constexpr Split mnist{"mnist", 10, 0};

[[nodiscard]] std::span<const Split> standard_splits() noexcept;
[[nodiscard]] const Split& find_split(std::string_view name);

[[nodiscard]] mnist::Dataset load_split(
    const std::filesystem::path& data_dir, const Split& split, bool training,
    std::size_t limit = 0);

class Classifier {
 public:
  explicit Classifier(const Split& split, float novelty_threshold = 0.78F,
                      float reward_learning_rate = 0.35F);

  void train(const mnist::Dataset& dataset, std::size_t epochs = 1,
             float threshold_multiplier = 1.30F);
  [[nodiscard]] std::uint8_t predict(
      const mnist::Image& image,
      float threshold_multiplier = 1.30F) const;
  [[nodiscard]] mnist::Evaluation evaluate(
      const mnist::Dataset& dataset,
      float threshold_multiplier = 1.30F) const;

 private:
  Split split_;
  std::vector<snnbase::Neuron> neurons_;
};

class StructuredClassifier {
 public:
  explicit StructuredClassifier(const Split& split,
                                float novelty_threshold = 0.78F,
                                float reward_learning_rate = 0.35F);
  ~StructuredClassifier();
  StructuredClassifier(StructuredClassifier&&) noexcept;
  StructuredClassifier& operator=(StructuredClassifier&&) noexcept;
  StructuredClassifier(const StructuredClassifier&) = delete;
  StructuredClassifier& operator=(const StructuredClassifier&) = delete;

  void train(const mnist::Dataset& dataset, std::size_t epochs = 1);
  [[nodiscard]] std::uint8_t predict(const mnist::Image& image) const;
  [[nodiscard]] mnist::Evaluation evaluate(
      const mnist::Dataset& dataset) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace snnbase_experiments::emnist

#endif
