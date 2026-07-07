#ifndef SNNBASE_EXPERIMENTS_SPIKING_CONV_HPP
#define SNNBASE_EXPERIMENTS_SPIKING_CONV_HPP

#include <snnbase/spiking_conv.hpp>
#include <snnbase_experiments/emnist.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace snnbase_experiments::spiking_conv {

using TrainingConfig = snnbase::spiking_conv::TrainingConfig;
using EpochMetrics = snnbase::spiking_conv::EpochMetrics;
using Evaluation = snnbase::spiking_conv::Evaluation;

class Classifier {
 public:
  explicit Classifier(const emnist::Split& split,
                      TrainingConfig config = {});
  ~Classifier();
  Classifier(Classifier&&) noexcept;
  Classifier& operator=(Classifier&&) noexcept;
  Classifier(const Classifier&) = delete;
  Classifier& operator=(const Classifier&) = delete;

  [[nodiscard]] std::vector<EpochMetrics> train(
      const mnist::Dataset& dataset);
  [[nodiscard]] Evaluation evaluate(const mnist::Dataset& dataset) const;
  [[nodiscard]] std::uint8_t predict(const mnist::Image& image) const;
  [[nodiscard]] std::size_t parameter_count() const noexcept;
  [[nodiscard]] std::size_t neuron_count() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace snnbase_experiments::spiking_conv

#endif
