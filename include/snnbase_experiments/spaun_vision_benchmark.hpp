#ifndef SNNBASE_EXPERIMENTS_SPAUN_VISION_BENCHMARK_HPP
#define SNNBASE_EXPERIMENTS_SPAUN_VISION_BENCHMARK_HPP

#include <snnbase/spiking_conv.hpp>
#include <snnbase_experiments/mnist.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace snnbase_experiments::spaun::vision_benchmark {

struct Config {
  std::filesystem::path data_directory;
  std::filesystem::path load_checkpoint;
  std::filesystem::path save_checkpoint;
  std::size_t train_limit{};
  std::size_t test_limit{};
  std::size_t epochs{5};
  std::size_t batch_size{64};
  std::size_t time_steps{8};
  float learning_rate{0.001F};
  std::uint32_t seed{42};
  bool augment{true};
  bool evaluate_only{};
};

struct Result {
  snnbase::spiking_conv::ModelConfig model;
  snnbase::spiking_conv::TrainingConfig training;
  std::size_t training_samples{};
  std::size_t test_samples{};
  std::size_t parameters{};
  std::vector<snnbase::spiking_conv::EpochMetrics> epochs;
  snnbase::spiking_conv::Evaluation evaluation;
  double training_seconds{};
  double evaluation_seconds{};
};

[[nodiscard]] snnbase::spiking_conv::ModelConfig spaun_a1_model();
[[nodiscard]] snnbase::spiking_conv::TrainingConfig training_config(
    const Config& config);
[[nodiscard]] Result run(const Config& config);

}  // namespace snnbase_experiments::spaun::vision_benchmark

#endif  // SNNBASE_EXPERIMENTS_SPAUN_VISION_BENCHMARK_HPP
