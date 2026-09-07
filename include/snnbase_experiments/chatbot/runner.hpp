#ifndef SNNBASE_EXPERIMENTS_CHATBOT_RUNNER_HPP
#define SNNBASE_EXPERIMENTS_CHATBOT_RUNNER_HPP

#include <snnbase_experiments/chatbot/qwen_weights.hpp>

#include <snnbase/language/decoder.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snnbase_experiments::chatbot {

inline constexpr std::string_view token_protocol =
    "snnbase.chatbot.tokens/v1";
inline constexpr std::string_view qwen3_phase0_model_id =
    "Qwen/Qwen3-0.6B-Base";
inline constexpr std::string_view qwen3_phase0_revision =
    "da87bfb608c14b7cf20ba1ce41287e8de496c0cd";
inline constexpr std::string_view qwen3_phase0_source_checkpoint_sha256 =
    "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba";
inline constexpr std::string_view qwen3_phase0_config_sha256 =
    "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59";
inline constexpr std::string_view qwen3_phase0_tokenizer_fingerprint_sha256 =
    "6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237";
inline constexpr std::size_t qwen3_phase0_dense_tensor_count = 310;
inline constexpr std::uint64_t qwen3_phase0_dense_payload_bytes =
    1'192'099'840ULL;

// Rejects restored Qwen provenance that is well-formed but is not the exact
// Phase 0 base-model oracle supported by this experiment.
void require_qwen3_phase0_checkpoint_provenance(
    const QwenDenseLoadResult& provenance);

struct GenerationConfig {
  std::size_t maximum_new_tokens{64};
  // Zero uses the full model vocabulary for direct C++ callers. Tokenizer
  // frontends must set their exact decodable vocabulary size so padded model
  // rows can never be sampled.
  std::size_t sampling_vocabulary_size{};
  std::uint64_t seed{42};
  double temperature{0.0};
  std::size_t top_k{};
  double top_p{1.0};
  std::vector<std::int64_t> eos_token_ids{};
  // Bounds prefill attention workspace; zero processes the prompt in one call.
  std::size_t prefill_chunk_size{128};
};

struct GenerationMetrics {
  double elapsed_milliseconds{};
  double time_to_first_token_milliseconds{};
  double generated_tokens_per_second{};
  double mean_spike_rate{};
  std::size_t generated_tokens{};
};

struct GenerationResult {
  std::vector<std::int64_t> output_ids{};
  std::string finish_reason{"length"};
  GenerationMetrics metrics{};
};

struct LogitValue {
  std::int64_t token_id{};
  double logit{};
};

struct LogitInspection {
  std::vector<LogitValue> probes{};
  std::vector<LogitValue> top_k{};
};

struct TemporalSiteDiagnostics {
  std::uint64_t element_count{};
  std::uint64_t saturated_count{};
  double absolute_error_sum{};
  double absolute_input_sum{};
  double maximum_scale_ratio{};
  std::uint64_t silent_nonzero_count{};
};

struct TemporalLayerDiagnostics {
  TemporalSiteDiagnostics attention{};
  TemporalSiteDiagnostics feed_forward{};
};

struct TemporalDiagnostics {
  std::vector<TemporalLayerDiagnostics> layers{};
};

struct LanguageModelMetrics {
  std::size_t token_count{};
  std::size_t correct_token_count{};
  double loss{};
  double objective_loss{};
  double perplexity{};
  double token_accuracy{};
  double mean_spike_rate{};
  bool optimizer_updated{};
  std::uint64_t micro_batch_count{};
  std::uint64_t optimizer_step_count{};
  double learning_rate{};
};

struct TrainingState {
  std::uint64_t micro_batch_count{};
  std::uint64_t optimizer_step_count{};
  std::size_t pending_accumulation_steps{};
  double learning_rate{};
};

struct RepositoryBuildProvenance {
  std::string revision{"unknown"};
  bool dirty{true};
};

struct BuildProvenance {
  RepositoryBuildProvenance experiments{};
  RepositoryBuildProvenance snnbase{};
};

struct RunnerConfig {
  snnbase::language::DecoderConfig model{};
  std::string device{"auto"};
  double learning_rate{3.0e-4};
  double weight_decay{1.0e-2};
  double gradient_clip_norm{1.0};
  std::size_t gradient_accumulation_steps{1};
  std::uint64_t total_optimizer_steps{};
  std::uint64_t warmup_optimizer_steps{};
  double minimum_learning_rate_ratio{};
  double spike_rate_target{0.15};
  double spike_rate_penalty{};
  BuildProvenance build_provenance{};
};

// Owns one decoder and optimizer. Tokenization is deliberately outside this
// class: the Python controller must use the exact, pinned upstream tokenizer
// and send integer IDs through the bounded protocol below.
class Runner {
 public:
  explicit Runner(RunnerConfig config);
  ~Runner();
  Runner(Runner&&) noexcept;
  Runner& operator=(Runner&&) noexcept;
  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  [[nodiscard]] GenerationResult generate(
      std::span<const std::int64_t> input_ids,
      const GenerationConfig& generation);
  [[nodiscard]] LogitInspection inspect_next_token_logits(
      std::span<const std::int64_t> input_ids,
      std::span<const std::int64_t> probe_token_ids,
      std::size_t top_k);
  [[nodiscard]] TemporalDiagnostics diagnose_temporal_encoding(
      std::span<const std::int64_t> input_ids);
  [[nodiscard]] LanguageModelMetrics train_step(
      std::span<const std::int64_t> input_ids,
      std::span<const std::uint8_t> loss_mask = {});
  [[nodiscard]] LanguageModelMetrics evaluate(
      std::span<const std::int64_t> input_ids,
      std::span<const std::uint8_t> loss_mask = {});
  [[nodiscard]] TrainingState flush_optimizer();

  // Explicit dense reference pass that records temporal encoder scales.
  // Calibration must use training data, before held-out evaluation.
  void calibrate(std::span<const std::int64_t> input_ids, bool reset = false,
                 double headroom = 4.0);

  void reset_state() noexcept;
  [[nodiscard]] QwenDenseLoadResult load_qwen_weights(
      const std::filesystem::path& archive,
      const QwenDenseArchiveIdentity& identity);
  void save_checkpoint(const std::filesystem::path& path) const;
  // weights_only loads parameters/calibration/provenance onto this runner's
  // device with a fresh optimizer; ordinary resume also restores optimizer,
  // counters and RNG state and therefore requires a matching device type.
  void load_checkpoint(const std::filesystem::path& path,
                       bool weights_only = false);

  [[nodiscard]] const RunnerConfig& config() const noexcept;
  [[nodiscard]] std::string device() const;
  [[nodiscard]] std::size_t parameter_count() const noexcept;
  [[nodiscard]] TrainingState training_state() const noexcept;
  [[nodiscard]] const std::optional<QwenDenseLoadResult>& qwen_weights()
      const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct ProtocolLimits {
  std::size_t maximum_line_bytes{4U * 1024U * 1024U};
  std::size_t maximum_request_id_bytes{128};
  std::size_t maximum_tokens{4096};
  std::size_t maximum_eos_tokens{32};
};

struct ProtocolServeResult {
  // True only when checkpoint_on_flush was nonempty and a successful flush
  // saved the current training state and no later train operation has begun.
  bool checkpoint_current{};
};

// Processes one JSON object per line until EOF or a successful shutdown
// request. Malformed requests receive a structured error and do not terminate
// the service. Resource limits are checked before model execution. When
// checkpoint_on_flush is nonempty, every successful flush writes a fully
// accumulated checkpoint at that path before acknowledging the request.
[[nodiscard]] ProtocolServeResult serve_token_protocol(
    std::istream& input, std::ostream& output, Runner& runner,
    const ProtocolLimits& limits = {},
    const std::filesystem::path& checkpoint_on_flush = {});

}  // namespace snnbase_experiments::chatbot

#endif  // SNNBASE_EXPERIMENTS_CHATBOT_RUNNER_HPP
