#ifndef SNNBASE_EXPERIMENTS_NMNIST_HPP
#define SNNBASE_EXPERIMENTS_NMNIST_HPP

#include <snnbase/snnbase.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace snnbase_experiments::nmnist {

inline constexpr std::size_t rows = 34;
inline constexpr std::size_t columns = 34;
inline constexpr std::size_t class_count = 10;

struct Event {
  std::uint8_t x{};
  std::uint8_t y{};
  bool polarity{};
  std::uint32_t timestamp{};
};

struct Sample {
  std::uint8_t label{};
  std::vector<Event> events;
};

struct Dataset {
  std::vector<Sample> samples;
  [[nodiscard]] std::size_t size() const noexcept;
};

struct Evaluation {
  std::size_t correct{};
  std::size_t total{};
  std::array<std::array<std::size_t, class_count>, class_count> confusion{};
  [[nodiscard]] double accuracy() const noexcept;
};

// Reads the N-MNIST five-byte event format. Overflow marker events (y == 240)
// extend subsequent timestamps and are not returned as sensor events.
[[nodiscard]] std::vector<Event> load_events(const std::filesystem::path& path);
[[nodiscard]] Dataset load_dataset(const std::filesystem::path& root,
                                   bool training, std::size_t limit = 0);

// Encodes an ordered event sequence into fixed temporal bins. Each payload bit
// represents a downsampled spatial cell and ON/OFF polarity, retaining timing
// as a sequence rather than collapsing a recording to one static image.
[[nodiscard]] std::vector<snnbase::SpikeEvent> encode_events(
    const std::vector<Event>& events, std::size_t time_bins = 10);

class Classifier {
 public:
  explicit Classifier(std::size_t time_bins = 10, float novelty_threshold = 0.78F,
                      float reward_learning_rate = 0.35F);

  void train(const Dataset& dataset, std::size_t epochs = 1);
  [[nodiscard]] std::pair<std::uint8_t, float> predict(const Sample& sample) const;
  [[nodiscard]] Evaluation evaluate(const Dataset& dataset) const;
  [[nodiscard]] std::size_t time_bins() const noexcept;

 private:
  std::size_t time_bins_{};
  std::vector<std::vector<snnbase::Neuron>> neurons_;
};

}  // namespace snnbase_experiments::nmnist

#endif  // SNNBASE_EXPERIMENTS_NMNIST_HPP
