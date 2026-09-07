#ifndef SNNBASE_EXPERIMENTS_CHATBOT_QWEN_WEIGHTS_HPP
#define SNNBASE_EXPERIMENTS_CHATBOT_QWEN_WEIGHTS_HPP

#include <snnbase/language/decoder.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace snnbase_experiments::chatbot {

inline constexpr std::uint32_t qwen_dense_archive_version = 1;

// Every field is required. The identity comes from the immutable conversion
// manifest rather than from an untrusted archive's self-description.
// For indexed shards, source_checkpoint_sha256 is the converter manifest's
// canonical weight-file-manifest digest, covering every shard and its index.
// Geometry and tied/untied readout are verified against DecoderConfig.
struct QwenDenseArchiveIdentity {
  std::string model_id;
  std::string revision;
  std::string archive_sha256;
  std::string source_checkpoint_sha256;
  std::string config_sha256;
  std::string tokenizer_fingerprint_sha256;
};

struct QwenDenseLoadResult {
  std::string model_id;
  std::string revision;
  std::string archive_sha256;
  std::string source_checkpoint_sha256;
  std::string config_sha256;
  std::string tokenizer_fingerprint_sha256;
  std::string metadata_sha256;
  std::string payload_sha256;
  std::size_t loaded_tensor_count{};
  std::uint64_t loaded_payload_bytes{};
  std::size_t untouched_runtime_parameter_count{};
};

// Verifies the complete archive and Decoder parameter coverage before copying
// any BF16 payload into the model. Dense parameters must match exactly. Only
// declared per-layer LIF threshold/leak parameters may remain initialized by
// DecoderConfig.
[[nodiscard]] QwenDenseLoadResult load_qwen_dense_weights(
    snnbase::language::Decoder& decoder,
    const std::filesystem::path& archive,
    const QwenDenseArchiveIdentity& expected);

}  // namespace snnbase_experiments::chatbot

#endif  // SNNBASE_EXPERIMENTS_CHATBOT_QWEN_WEIGHTS_HPP
