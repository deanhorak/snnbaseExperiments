#include <snnbase_experiments/chatbot/qwen_weights.hpp>

#include <torch/torch.h>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using snnbase_experiments::chatbot::QwenDenseArchiveIdentity;
using snnbase_experiments::chatbot::load_qwen_dense_weights;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::uint8_t hex_nibble(const char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<std::uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<std::uint8_t>(10 + value - 'a');
  }
  throw std::runtime_error("invalid checked-in archive hex fixture");
}

std::filesystem::path materialize_fixture(
    const std::string& name = "qwen_dense_v1_tiny.hex") {
  const auto source = std::filesystem::path(__FILE__).parent_path() /
                      "fixtures/chatbot" / name;
  std::ifstream input(source);
  if (!input) {
    throw std::runtime_error("could not open Qwen archive hex fixture");
  }
  const auto nonce = std::chrono::high_resolution_clock::now()
                         .time_since_epoch()
                         .count();
  const auto output_path =
      std::filesystem::temp_directory_path() /
      ("snnbase-qwen-dense-v1-tiny-" + std::to_string(nonce) + ".snnq");
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("could not materialize Qwen archive fixture");
  }
  char high{};
  char low{};
  while (input.get(high)) {
    if (std::isspace(static_cast<unsigned char>(high))) {
      continue;
    }
    do {
      if (!input.get(low)) {
        throw std::runtime_error("odd checked-in archive hex fixture");
      }
    } while (std::isspace(static_cast<unsigned char>(low)));
    const auto byte = static_cast<char>((hex_nibble(high) << 4U) | hex_nibble(low));
    output.put(byte);
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write materialized Qwen fixture");
  }
  return output_path;
}

snnbase::language::DecoderConfig tiny_config() {
  return {
      .vocabulary_size = 17,
      .maximum_sequence_length = 16,
      .model_dimension = 8,
      .layer_count = 2,
      .query_head_count = 2,
      .key_value_head_count = 1,
      .head_dimension = 4,
      .query_key_normalization = true,
      .feed_forward_dimension = 12,
      .simulation_steps = 2,
      .spiking = true,
      .rms_epsilon = 1.0e-6,
      .rope_base = 1'000'000.0,
      .seed = 7,
  };
}

torch::Tensor parameter(const snnbase::language::Decoder& decoder,
                        std::string_view name) {
  for (const auto& item : decoder->named_parameters(true)) {
    if (item.key() == name) {
      return item.value();
    }
  }
  throw std::runtime_error("test Decoder parameter is missing: " +
                           std::string(name));
}

void test_exact_dense_load_and_lif_preservation() {
  const auto archive = materialize_fixture();
  snnbase::language::Decoder decoder(tiny_config());
  const auto dense_before = parameter(decoder, "token_embedding.weight").clone();
  const auto lif_before =
      parameter(decoder, "block_0.attention_lif.raw_threshold").clone();
  const QwenDenseArchiveIdentity identity{
      .model_id = "tests/Qwen3Tiny",
      .revision = "0123456789abcdef0123456789abcdef01234567",
      .archive_sha256 =
          "2773ebdca878f7273506e9561123af4c5d9a1613b4cc9a9c7271e6817c7a76d5",
      .source_checkpoint_sha256 =
          "e08cee48dffde59a222b6084d13345365ee8692f6a59ee81f494045d28f51b26",
      .config_sha256 =
          "453b165797127d8f86cb2fe16f03df8b04dd2f24451135120b525e2fe91752e5",
      .tokenizer_fingerprint_sha256 =
          "e2f448429ba76678d32ed7cf99e8d9adb031944a11a170771365a0f76f29a742",
  };
  auto wrong = identity;
  wrong.archive_sha256.front() = '0';
  bool rejected = false;
  try {
    static_cast<void>(load_qwen_dense_weights(decoder, archive, wrong));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "incorrect expected archive hash was accepted");
  require(torch::equal(dense_before, parameter(decoder, "token_embedding.weight")),
          "failed archive validation mutated a dense parameter");

  const auto result = load_qwen_dense_weights(decoder, archive, identity);
  require(result.loaded_tensor_count == 24U &&
              result.untouched_runtime_parameter_count == 8U &&
              result.loaded_payload_bytes == 2304U,
          "Qwen dense load coverage is incomplete");
  require(!torch::equal(dense_before, parameter(decoder, "token_embedding.weight")),
          "Qwen dense payload was not copied");
  require(torch::equal(lif_before,
                       parameter(decoder,
                                 "block_0.attention_lif.raw_threshold")),
          "Qwen dense import changed a runtime-initialized LIF parameter");
  std::filesystem::remove(archive);
}

void test_untied_sharded_second_geometry() {
  const auto archive = materialize_fixture("qwen_dense_v1_untied_sharded.hex");
  auto config = tiny_config();
  config.model_dimension = 12;
  config.layer_count = 3;
  config.query_head_count = 4;
  config.key_value_head_count = 2;
  config.feed_forward_dimension = 20;
  const QwenDenseArchiveIdentity identity{
      .model_id = "tests/Qwen3Tiny",
      .revision = "0123456789abcdef0123456789abcdef01234567",
      .archive_sha256 =
          "bee0456d510292bce99249613fa24be16ecc952cda6a32f12ed3f522729a0be5",
      .source_checkpoint_sha256 =
          "2fb837de9ac08ac275939fb0fa77b12fb99a7d979526df0dbae3f553ae3c4d8a",
      .config_sha256 =
          "aff0e31b326e623359a1b88d95397cc669fa2b3c1bed31730835fa0b6fef17c2",
      .tokenizer_fingerprint_sha256 =
          "e2f448429ba76678d32ed7cf99e8d9adb031944a11a170771365a0f76f29a742",
  };
  snnbase::language::Decoder tied_decoder(config);
  const auto tied_before = parameter(tied_decoder, "token_embedding.weight").clone();
  bool rejected = false;
  try {
    static_cast<void>(load_qwen_dense_weights(tied_decoder, archive, identity));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "untied archive was accepted by a tied Decoder");
  require(torch::equal(tied_before, parameter(tied_decoder, "token_embedding.weight")),
          "tied/untied mismatch mutated the Decoder");
  config.tie_word_embeddings = false;
  snnbase::language::Decoder decoder(config);
  const auto head_before = parameter(decoder, "lm_head.weight").clone();
  const auto result = load_qwen_dense_weights(decoder, archive, identity);
  require(result.loaded_tensor_count == 36U &&
              result.untouched_runtime_parameter_count == 12U &&
              result.loaded_payload_bytes == 8808U,
          "untied second-geometry archive coverage is incomplete");
  require(!torch::equal(head_before, parameter(decoder, "lm_head.weight")),
          "untied language-model head was not imported");
  require(parameter(decoder, "lm_head.weight").data_ptr() !=
              parameter(decoder, "token_embedding.weight").data_ptr(),
          "untied language-model head shares embedding storage");
  std::filesystem::remove(archive);
}

}  // namespace

int main() {
  test_exact_dense_load_and_lif_preservation();
  test_untied_sharded_second_geometry();
}
