#include <snnbase_experiments/nmnist.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace snnbase_experiments::nmnist {
namespace {

std::vector<std::filesystem::path> sample_paths(const std::filesystem::path& root,
                                                 bool training) {
  const auto split = root / (training ? "Train" : "Test");
  if (!std::filesystem::is_directory(split)) {
    throw std::runtime_error("N-MNIST split directory is missing: " + split.string());
  }
  std::vector<std::filesystem::path> paths;
  for (std::size_t label = 0; label < class_count; ++label) {
    const auto directory = split / std::to_string(label);
    if (!std::filesystem::is_directory(directory)) {
      throw std::runtime_error("N-MNIST class directory is missing: " + directory.string());
    }
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
      if (entry.is_regular_file() && entry.path().extension() == ".bin") {
        paths.push_back(entry.path());
      }
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

std::uint8_t label_from_path(const std::filesystem::path& path) {
  const auto text = path.parent_path().filename().string();
  if (text.size() != 1 || text.front() < '0' || text.front() > '9') {
    throw std::runtime_error("invalid N-MNIST class directory: " + path.string());
  }
  return static_cast<std::uint8_t>(text.front() - '0');
}

}  // namespace

std::size_t Dataset::size() const noexcept { return samples.size(); }

double Evaluation::accuracy() const noexcept {
  return total == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(total);
}

std::vector<Event> load_events(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open N-MNIST event file: " + path.string());
  }
  std::vector<Event> result;
  std::uint32_t timestamp_offset{};
  while (true) {
    std::array<unsigned char, 5> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() == 0 && input.eof()) {
      break;
    }
    if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
      throw std::runtime_error("truncated N-MNIST event record: " + path.string());
    }
    const auto timestamp = (static_cast<std::uint32_t>(bytes[2] & 0x7FU) << 16U) |
                           (static_cast<std::uint32_t>(bytes[3]) << 8U) |
                           static_cast<std::uint32_t>(bytes[4]);
    if (bytes[1] == 240U) {
      if (timestamp_offset > std::numeric_limits<std::uint32_t>::max() - (1U << 23U)) {
        throw std::runtime_error("N-MNIST timestamp overflow: " + path.string());
      }
      timestamp_offset += 1U << 23U;
      continue;
    }
    if (bytes[0] >= columns || bytes[1] >= rows ||
        timestamp > std::numeric_limits<std::uint32_t>::max() - timestamp_offset) {
      throw std::runtime_error("invalid N-MNIST event record: " + path.string());
    }
    result.push_back({bytes[0], bytes[1], (bytes[2] & 0x80U) != 0U,
                      timestamp_offset + timestamp});
  }
  return result;
}

Dataset load_dataset(const std::filesystem::path& root, const bool training,
                     const std::size_t limit) {
  const auto paths = sample_paths(root, training);
  const auto count = limit == 0 ? paths.size() : std::min(limit, paths.size());
  Dataset dataset;
  dataset.samples.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    dataset.samples.push_back({label_from_path(paths[index]), load_events(paths[index])});
  }
  return dataset;
}

std::vector<snnbase::SpikeEvent> encode_events(const std::vector<Event>& events,
                                                const std::size_t time_bins) {
  if (time_bins == 0) {
    throw std::invalid_argument("N-MNIST time bins must be positive");
  }
  std::vector<snnbase::SpikeEvent::Storage> patterns(time_bins);
  if (events.empty()) {
    return std::vector<snnbase::SpikeEvent>(time_bins);
  }
  constexpr auto payload = snnbase::SpikeEvent::payload_width();
  static_assert(payload >= 2, "N-MNIST encoding requires at least two payload bits");
  const auto cells = payload / 2;
  const auto grid_columns = std::max<std::size_t>(1, static_cast<std::size_t>(
      std::sqrt(static_cast<double>(cells) * columns / rows)));
  const auto grid_rows = std::max<std::size_t>(1, cells / grid_columns);
  const auto first = events.front().timestamp;
  const auto last = events.back().timestamp;
  const auto duration = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(last) - first + 1);
  for (const auto& event : events) {
    const auto elapsed = static_cast<std::uint64_t>(event.timestamp) - first;
    const auto bin = std::min(time_bins - 1, static_cast<std::size_t>(elapsed * time_bins / duration));
    const auto row = std::min(grid_rows - 1, static_cast<std::size_t>(event.y) * grid_rows / rows);
    const auto column = std::min(grid_columns - 1, static_cast<std::size_t>(event.x) * grid_columns / columns);
    const auto cell = row * grid_columns + column;
    if (cell < cells) {
      patterns[bin] |= snnbase::SpikeEvent::payload_bit(2 * cell + (event.polarity ? 1 : 0));
    }
  }
  std::vector<snnbase::SpikeEvent> result;
  result.reserve(time_bins);
  for (const auto pattern : patterns) {
    result.emplace_back(pattern);
  }
  return result;
}

Classifier::Classifier(const std::size_t time_bins, const float novelty_threshold,
                       const float reward_learning_rate)
    : time_bins_(time_bins), neurons_(class_count, std::vector<snnbase::Neuron>(time_bins)) {
  if (time_bins == 0 || !(novelty_threshold >= 0.0F && novelty_threshold <= 1.0F) ||
      !(reward_learning_rate > 0.0F) || !std::isfinite(reward_learning_rate)) {
    throw std::invalid_argument("invalid N-MNIST classifier configuration");
  }
  for (std::size_t label = 0; label < class_count; ++label) {
    for (std::size_t bin = 0; bin < time_bins_; ++bin) {
      neurons_[label][bin] = snnbase::Neuron{
          label * time_bins_ + bin,
          {.initial_stdp_relevance = 1.0F,
           .reward_learning_rate = reward_learning_rate,
           .novelty_threshold = novelty_threshold,
           .prototype_aggregation = snnbase::PrototypeAggregation::maximum}};
    }
  }
}

void Classifier::train(const Dataset& dataset, const std::size_t epochs) {
  if (epochs == 0) {
    throw std::invalid_argument("N-MNIST epochs must be positive");
  }
  for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
    for (const auto& sample : dataset.samples) {
      if (sample.label >= class_count) {
        throw std::invalid_argument("N-MNIST label is outside configured classes");
      }
      const auto sequence = encode_events(sample.events, time_bins_);
      for (std::size_t bin = 0; bin < time_bins_; ++bin) {
        static_cast<void>(neurons_[sample.label][bin].reinforce(sequence[bin], 1.0F));
      }
    }
  }
}

std::pair<std::uint8_t, float> Classifier::predict(const Sample& sample) const {
  const auto sequence = encode_events(sample.events, time_bins_);
  std::uint8_t best_label{};
  float best_score = -std::numeric_limits<float>::infinity();
  for (std::size_t label = 0; label < class_count; ++label) {
    float score{};
    for (std::size_t bin = 0; bin < time_bins_; ++bin) {
      score += neurons_[label][bin].evaluate(sequence[bin]);
    }
    score /= static_cast<float>(time_bins_);
    if (score > best_score) {
      best_label = static_cast<std::uint8_t>(label);
      best_score = score;
    }
  }
  return {best_label, best_score};
}

Evaluation Classifier::evaluate(const Dataset& dataset) const {
  Evaluation result;
  result.total = dataset.size();
  for (const auto& sample : dataset.samples) {
    if (sample.label >= class_count) {
      throw std::invalid_argument("N-MNIST label is outside configured classes");
    }
    const auto [predicted, score] = predict(sample);
    static_cast<void>(score);
    ++result.confusion[sample.label][predicted];
    result.correct += predicted == sample.label ? 1U : 0U;
  }
  return result;
}

std::size_t Classifier::time_bins() const noexcept { return time_bins_; }

}  // namespace snnbase_experiments::nmnist
