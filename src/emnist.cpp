#include <snnbase_experiments/emnist.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace snnbase_experiments::emnist {
namespace {

constexpr std::array splits{byclass, bymerge, balanced, letters, digits, mnist};

std::size_t class_index(const Split& split, std::uint8_t label) {
  if (label < split.label_offset) {
    throw std::invalid_argument("EMNIST label is below the split's label offset");
  }
  const auto index = static_cast<std::size_t>(label - split.label_offset);
  if (index >= split.class_count) {
    throw std::invalid_argument("EMNIST label is outside the split's class range");
  }
  return index;
}

}  // namespace

std::span<const Split> standard_splits() noexcept {
  return splits;
}

const Split& find_split(std::string_view name) {
  const auto match = std::find_if(
      splits.begin(), splits.end(),
      [name](const Split& split) { return split.name == name; });
  if (match == splits.end()) {
    throw std::invalid_argument("unknown EMNIST split: " + std::string(name));
  }
  return *match;
}

mnist::Dataset load_split(const std::filesystem::path& data_dir,
                          const Split& split, bool training,
                          std::size_t limit) {
  const auto prefix =
      "emnist-" + std::string(split.name) + (training ? "-train" : "-test");
  auto dataset = mnist::load_idx_dataset(
      data_dir / (prefix + "-images-idx3-ubyte"),
      data_dir / (prefix + "-labels-idx1-ubyte"), limit,
      static_cast<std::uint8_t>(split.label_offset + split.class_count - 1));
  for (const auto label : dataset.labels) {
    static_cast<void>(class_index(split, label));
  }
  return dataset;
}

Classifier::Classifier(const Split& split, float novelty_threshold,
                       float reward_learning_rate)
    : split_(split) {
  if (!(novelty_threshold >= 0.0F && novelty_threshold <= 1.0F)) {
    throw std::invalid_argument("novelty threshold must be in the range 0-1");
  }
  if (!(reward_learning_rate > 0.0F)) {
    throw std::invalid_argument("reward learning rate must be positive");
  }
  neurons_.reserve(split.class_count);
  for (std::size_t label = 0; label < split.class_count; ++label) {
    neurons_.emplace_back(
        label,
        snnbase::NeuronParameters{
            .initial_stdp_relevance = 1.0F,
            .reward_learning_rate = reward_learning_rate,
            .novelty_threshold = novelty_threshold,
            .prototype_aggregation = snnbase::PrototypeAggregation::maximum,
            .replay_interval = 1000,
            .replay_batch_size = 4,
            .replay_reward = 0.05F});
  }
}

void Classifier::train(const mnist::Dataset& dataset, std::size_t epochs,
                       float threshold_multiplier) {
  if (dataset.images.size() != dataset.labels.size()) {
    throw std::invalid_argument("training image and label counts differ");
  }
  for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
    for (std::size_t index = 0; index < dataset.size(); ++index) {
      const auto label = class_index(split_, dataset.labels[index]);
      static_cast<void>(neurons_[label].reinforce(
          mnist::encode_image(dataset.images[index], threshold_multiplier),
          1.0F));
    }
  }
}

std::uint8_t Classifier::predict(const mnist::Image& image,
                                 float threshold_multiplier) const {
  const auto event = mnist::encode_image(image, threshold_multiplier);
  std::size_t best_label = 0;
  float best_score = -1.0F;
  for (std::size_t label = 0; label < neurons_.size(); ++label) {
    const auto score = neurons_[label].evaluate(event);
    if (score > best_score) {
      best_label = label;
      best_score = score;
    }
  }
  return static_cast<std::uint8_t>(best_label + split_.label_offset);
}

mnist::Evaluation Classifier::evaluate(
    const mnist::Dataset& dataset, float threshold_multiplier) const {
  if (dataset.images.size() != dataset.labels.size()) {
    throw std::invalid_argument("evaluation image and label counts differ");
  }
  mnist::Evaluation result;
  result.total = dataset.size();
  for (std::size_t index = 0; index < dataset.size(); ++index) {
    const auto predicted =
        predict(dataset.images[index], threshold_multiplier);
    result.correct += predicted == dataset.labels[index] ? 1U : 0U;
  }
  return result;
}

namespace {

constexpr std::size_t tile_count = 4;
constexpr std::size_t channel_count = tile_count + 1;
constexpr std::size_t expert_count = 4;
constexpr std::size_t prototypes_per_bank = 64;
constexpr std::size_t detector_count = 6;
constexpr std::size_t patch_grid = 4;
constexpr std::size_t patch_width = 7;

using FeatureEvents = std::array<snnbase::SpikeEvent, channel_count>;

mnist::Image centered_image(const mnist::Image& source, int extra_x = 0,
                            int extra_y = 0) {
  if (source.rows != 28 || source.columns != 28 ||
      source.pixels.size() != 28 * 28) {
    throw std::invalid_argument("structured EMNIST encoding requires 28x28 images");
  }

  std::size_t min_row = source.rows;
  std::size_t min_column = source.columns;
  std::size_t max_row = 0;
  std::size_t max_column = 0;
  bool has_ink = false;
  for (std::size_t row = 0; row < source.rows; ++row) {
    for (std::size_t column = 0; column < source.columns; ++column) {
      if (source.pixels[row * source.columns + column] > 20) {
        min_row = std::min(min_row, row);
        min_column = std::min(min_column, column);
        max_row = std::max(max_row, row);
        max_column = std::max(max_column, column);
        has_ink = true;
      }
    }
  }
  if (!has_ink) {
    return source;
  }

  const auto row_shift =
      (static_cast<int>(source.rows) - 1 - static_cast<int>(min_row) -
       static_cast<int>(max_row)) /
          2 +
      extra_y;
  const auto column_shift =
      (static_cast<int>(source.columns) - 1 - static_cast<int>(min_column) -
       static_cast<int>(max_column)) /
          2 +
      extra_x;
  mnist::Image result{source.rows, source.columns,
                      std::vector<std::uint8_t>(source.pixels.size())};
  for (std::size_t row = 0; row < source.rows; ++row) {
    for (std::size_t column = 0; column < source.columns; ++column) {
      const auto target_row = static_cast<int>(row) + row_shift;
      const auto target_column = static_cast<int>(column) + column_shift;
      if (target_row >= 0 && target_row < static_cast<int>(source.rows) &&
          target_column >= 0 &&
          target_column < static_cast<int>(source.columns)) {
        result.pixels[static_cast<std::size_t>(target_row) * result.columns +
                      static_cast<std::size_t>(target_column)] =
            source.pixels[row * source.columns + column];
      }
    }
  }
  return result;
}

double image_mean(const mnist::Image& image) {
  std::uint64_t sum = 0;
  for (const auto pixel : image.pixels) {
    sum += pixel;
  }
  return static_cast<double>(sum) /
         static_cast<double>(image.pixels.size());
}

snnbase::SpikeEvent encode_tile(const mnist::Image& image,
                                std::size_t tile_index) {
  constexpr std::size_t grid_rows = 6;
  constexpr std::size_t grid_columns = 8;
  constexpr std::size_t cells = grid_rows * grid_columns;
  static_assert(cells * 2 <= snnbase::SpikeEvent::payload_width());

  const auto tile_row = tile_index / 2;
  const auto tile_column = tile_index % 2;
  const auto start_row = tile_row * 14;
  const auto start_column = tile_column * 14;
  std::array<std::uint32_t, cells> sums{};
  std::array<std::uint16_t, cells> counts{};
  for (std::size_t row = 0; row < 14; ++row) {
    for (std::size_t column = 0; column < 14; ++column) {
      const auto grid_row = row * grid_rows / 14;
      const auto grid_column = column * grid_columns / 14;
      const auto cell = grid_row * grid_columns + grid_column;
      sums[cell] += image.pixels[(start_row + row) * image.columns +
                                  start_column + column];
      ++counts[cell];
    }
  }

  const auto mean = image_mean(image);
  snnbase::SpikeEvent::Storage bits = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const auto cell_mean =
        static_cast<double>(sums[cell]) / static_cast<double>(counts[cell]);
    if (cell_mean >= mean * 0.70 && sums[cell] != 0) {
      bits |= snnbase::SpikeEvent::payload_bit(cell);
    }
    if (cell_mean >= mean * 1.40 && sums[cell] != 0) {
      bits |= snnbase::SpikeEvent::payload_bit(cells + cell);
    }
  }
  return snnbase::SpikeEvent(bits);
}

snnbase::SpikeEvent::Storage canonical_pattern(std::size_t detector,
                                               int shift_x = 0,
                                               int shift_y = 0) {
  snnbase::SpikeEvent::Storage bits = 0;
  const auto set = [&bits, shift_x, shift_y](int row, int column) {
    row += shift_y;
    column += shift_x;
    if (row >= 0 && row < static_cast<int>(patch_width) && column >= 0 &&
        column < static_cast<int>(patch_width)) {
      bits |= snnbase::SpikeEvent::payload_bit(
          static_cast<std::size_t>(row) * patch_width +
          static_cast<std::size_t>(column));
    }
  };
  for (int position = 0; position < static_cast<int>(patch_width); ++position) {
    if (detector == 0) {
      set(3, position);
    } else if (detector == 1) {
      set(position, 3);
    } else if (detector == 2) {
      set(position, position);
    } else if (detector == 3) {
      set(position, 6 - position);
    } else if (detector == 4) {
      set(1, position);
      set(position, 1);
    } else {
      set(position == 0 || position == 6 ? position : 0, position);
      set(position == 0 || position == 6 ? position : 6, position);
    }
  }
  if (detector == 5) {
    for (int position = 1; position < 6; ++position) {
      set(position, 0);
      set(position, 6);
    }
  }
  return bits;
}

class PatchFeatureLayer {
 public:
  PatchFeatureLayer() {
    for (std::size_t detector = 0; detector < detectors_.size(); ++detector) {
      detectors_[detector] = snnbase::Neuron{
          detector,
          {.prototype_aggregation = snnbase::PrototypeAggregation::maximum}};
      for (const auto shift : {-1, 0, 1}) {
        detectors_[detector].remember(
            snnbase::SpikeEvent(canonical_pattern(detector, shift, 0)), 1.0F);
        detectors_[detector].remember(
            snnbase::SpikeEvent(canonical_pattern(detector, 0, shift)), 1.0F);
      }
    }
  }

  snnbase::SpikeEvent transform(const mnist::Image& image) const {
    const auto threshold = image_mean(image) * 0.85;
    snnbase::SpikeEvent::Storage features = 0;
    for (std::size_t patch_row = 0; patch_row < patch_grid; ++patch_row) {
      for (std::size_t patch_column = 0; patch_column < patch_grid;
           ++patch_column) {
        snnbase::SpikeEvent::Storage patch_bits = 0;
        for (std::size_t row = 0; row < patch_width; ++row) {
          for (std::size_t column = 0; column < patch_width; ++column) {
            const auto pixel =
                image.pixels[(patch_row * patch_width + row) * image.columns +
                             patch_column * patch_width + column];
            if (pixel >= threshold && pixel != 0) {
              patch_bits |= snnbase::SpikeEvent::payload_bit(
                  row * patch_width + column);
            }
          }
        }
        const auto patch = snnbase::SpikeEvent(patch_bits);
        const auto patch_index = patch_row * patch_grid + patch_column;
        for (std::size_t detector = 0; detector < detectors_.size();
             ++detector) {
          if (detectors_[detector].evaluate(patch) >= 0.16F) {
            features |= snnbase::SpikeEvent::payload_bit(
                patch_index * detector_count + detector);
          }
        }
      }
    }
    return snnbase::SpikeEvent(features);
  }

 private:
  std::array<snnbase::Neuron, detector_count> detectors_;
};

FeatureEvents encode_features(const mnist::Image& image,
                              const PatchFeatureLayer& feature_layer,
                              int shift_x = 0, int shift_y = 0) {
  const auto centered = centered_image(image, shift_x, shift_y);
  FeatureEvents events;
  for (std::size_t tile = 0; tile < tile_count; ++tile) {
    events[tile] = encode_tile(centered, tile);
  }
  events.back() = feature_layer.transform(centered);
  return events;
}

}  // namespace

struct StructuredClassifier::Impl {
  struct Expert {
    std::array<snnbase::Neuron, channel_count> banks;
    std::size_t assignments{};
  };

  Split split;
  float novelty_threshold;
  float reward_learning_rate;
  PatchFeatureLayer feature_layer;
  std::vector<std::array<Expert, expert_count>> classes;

  Impl(const Split& selected_split, float novelty, float learning_rate)
      : split(selected_split),
        novelty_threshold(novelty),
        reward_learning_rate(learning_rate),
        classes(selected_split.class_count) {
    if (!(novelty >= 0.0F && novelty <= 1.0F)) {
      throw std::invalid_argument("novelty threshold must be in the range 0-1");
    }
    if (!(learning_rate > 0.0F) || !std::isfinite(learning_rate)) {
      throw std::invalid_argument("reward learning rate must be positive");
    }
    for (auto& class_experts : classes) {
      for (auto& expert : class_experts) {
        for (auto& bank : expert.banks) {
          bank = snnbase::Neuron{
              {.initial_stdp_relevance = 1.0F,
               .reward_learning_rate = learning_rate,
               .novelty_threshold = novelty,
               .prototype_aggregation =
                   snnbase::PrototypeAggregation::maximum}};
        }
      }
    }
  }

  float expert_score(const Expert& expert, const FeatureEvents& events) const {
    float score = 0.0F;
    for (std::size_t channel = 0; channel < tile_count; ++channel) {
      score += expert.banks[channel].evaluate(events[channel]);
    }
    score += 0.5F * expert.banks.back().evaluate(events.back());
    return score / (static_cast<float>(tile_count) + 0.5F);
  }

  std::pair<std::size_t, float> best_expert(
      std::size_t label, const FeatureEvents& events) const {
    std::size_t best = 0;
    float best_score = -1.0F;
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
      const auto score = expert_score(classes[label][expert], events);
      if (score > best_score) {
        best = expert;
        best_score = score;
      }
    }
    return {best, best_score};
  }

  std::pair<std::size_t, std::size_t> prediction(
      const FeatureEvents& events) const {
    std::size_t best_label = 0;
    std::size_t best_expert_index = 0;
    float best_score = -1.0F;
    for (std::size_t label = 0; label < classes.size(); ++label) {
      const auto [expert, score] = best_expert(label, events);
      if (score > best_score) {
        best_label = label;
        best_expert_index = expert;
        best_score = score;
      }
    }
    return {best_label, best_expert_index};
  }
};

StructuredClassifier::StructuredClassifier(const Split& split,
                                           float novelty_threshold,
                                           float reward_learning_rate)
    : impl_(std::make_unique<Impl>(split, novelty_threshold,
                                  reward_learning_rate)) {}

StructuredClassifier::~StructuredClassifier() = default;
StructuredClassifier::StructuredClassifier(StructuredClassifier&&) noexcept =
    default;
StructuredClassifier& StructuredClassifier::operator=(
    StructuredClassifier&&) noexcept = default;

void StructuredClassifier::train(const mnist::Dataset& dataset,
                                 std::size_t epochs) {
  if (dataset.images.size() != dataset.labels.size()) {
    throw std::invalid_argument("training image and label counts differ");
  }
  constexpr std::array shifts{
      std::pair{0, 0}, std::pair{-1, 0}, std::pair{1, 0},
      std::pair{0, -1}, std::pair{0, 1}};
  for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
    for (std::size_t index = 0; index < dataset.size(); ++index) {
      const auto label = class_index(impl_->split, dataset.labels[index]);
      const auto [shift_x, shift_y] = shifts[(index + epoch) % shifts.size()];
      const auto features = encode_features(dataset.images[index],
                                            impl_->feature_layer, shift_x,
                                            shift_y);
      auto& experts = impl_->classes[label];
      std::size_t selected = 0;
      const auto empty = std::find_if(
          experts.begin(), experts.end(),
          [](const Impl::Expert& expert) { return expert.assignments == 0; });
      if (empty != experts.end()) {
        selected = static_cast<std::size_t>(empty - experts.begin());
      } else {
        float best_score = -1.0F;
        bool has_available = false;
        for (std::size_t candidate = 0; candidate < experts.size();
             ++candidate) {
          const auto available =
              experts[candidate].banks.front().history().size() <
              prototypes_per_bank;
          if (available && !has_available) {
            has_available = true;
            best_score = -1.0F;
          }
          if (available != has_available) {
            continue;
          }
          const auto score = impl_->expert_score(experts[candidate], features);
          if (score > best_score) {
            selected = candidate;
            best_score = score;
          }
        }
      }
      auto& expert = experts[selected];
      for (std::size_t channel = 0; channel < channel_count; ++channel) {
        if (expert.banks[channel].history().size() < prototypes_per_bank) {
          expert.banks[channel].remember(features[channel], 1.0F);
        }
      }
      ++expert.assignments;

      if (index % 16 == 0) {
        const auto [predicted, predicted_expert] =
            impl_->prediction(features);
        if (predicted != label) {
          auto& competitor = impl_->classes[predicted][predicted_expert];
          for (std::size_t channel = 0; channel < channel_count; ++channel) {
            static_cast<void>(
                competitor.banks[channel].reinforce(features[channel], -0.5F));
          }
        }
      }
    }
  }
}

std::uint8_t StructuredClassifier::predict(const mnist::Image& image) const {
  const auto features = encode_features(image, impl_->feature_layer);
  return static_cast<std::uint8_t>(
      impl_->prediction(features).first + impl_->split.label_offset);
}

mnist::Evaluation StructuredClassifier::evaluate(
    const mnist::Dataset& dataset) const {
  if (dataset.images.size() != dataset.labels.size()) {
    throw std::invalid_argument("evaluation image and label counts differ");
  }
  mnist::Evaluation result;
  result.total = dataset.size();
  for (std::size_t index = 0; index < dataset.size(); ++index) {
    result.correct += predict(dataset.images[index]) == dataset.labels[index]
                          ? 1U
                          : 0U;
  }
  return result;
}

}  // namespace snnbase_experiments::emnist
