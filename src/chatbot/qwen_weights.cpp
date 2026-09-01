#include <snnbase_experiments/chatbot/qwen_weights.hpp>

#include <snnbase_experiments/chatbot/manifest.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace snnbase_experiments::chatbot {
namespace {

constexpr std::array<std::uint8_t, 8> archive_magic{
    'S', 'N', 'N', 'Q', 'W', 'E', 'N', 0};
constexpr std::uint32_t endian_marker = 0x01020304U;
constexpr std::size_t header_size = 256U;
constexpr std::size_t config_size = 88U;
constexpr std::size_t tensor_prefix_size = 56U;
constexpr std::uint32_t required_flags = 0x1fU;
constexpr std::uint8_t dtype_bfloat16 = 1U;
constexpr std::size_t maximum_tensor_count = 4096U;
constexpr std::size_t maximum_rank = 8U;
constexpr std::size_t maximum_name_bytes = 1024U;
constexpr std::uint64_t maximum_metadata_bytes = 64U * 1024U * 1024U;
constexpr std::size_t hash_buffer_size = 1024U * 1024U;

constexpr std::array<std::uint32_t, 64> round_constants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

class Sha256 {
 public:
  void update(std::span<const std::uint8_t> bytes) {
    if (bytes.size() > std::numeric_limits<std::uint64_t>::max() - byte_count_) {
      throw std::overflow_error("SHA-256 input is too large");
    }
    byte_count_ += static_cast<std::uint64_t>(bytes.size());
    for (const auto byte : bytes) {
      buffer_[buffer_size_++] = byte;
      if (buffer_size_ == buffer_.size()) {
        transform(buffer_);
        buffer_size_ = 0U;
      }
    }
  }

  std::array<std::uint8_t, 32> finish() {
    const auto bit_count = byte_count_ * 8U;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56U) {
      std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
                buffer_.end(), 0U);
      transform(buffer_);
      buffer_size_ = 0U;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
              buffer_.begin() + 56, 0U);
    for (std::size_t index = 0; index < 8U; ++index) {
      buffer_[63U - index] =
          static_cast<std::uint8_t>(bit_count >> (index * 8U));
    }
    transform(buffer_);
    std::array<std::uint8_t, 32> result{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      for (std::size_t byte = 0; byte < 4U; ++byte) {
        result[word * 4U + byte] = static_cast<std::uint8_t>(
            state_[word] >> ((3U - byte) * 8U));
      }
    }
    return result;
  }

 private:
  void transform(const std::array<std::uint8_t, 64>& block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16U; ++index) {
      const auto offset = index * 4U;
      words[index] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                     (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
                     (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
                     static_cast<std::uint32_t>(block[offset + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      const auto s0 = std::rotr(words[index - 15U], 7) ^
                      std::rotr(words[index - 15U], 18) ^
                      (words[index - 15U] >> 3U);
      const auto s1 = std::rotr(words[index - 2U], 17) ^
                      std::rotr(words[index - 2U], 19) ^
                      (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = std::tuple{
        state_[0], state_[1], state_[2], state_[3],
        state_[4], state_[5], state_[6], state_[7]};
    for (std::size_t index = 0; index < words.size(); ++index) {
      const auto sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const auto choose = (e & f) ^ ((~e) & g);
      const auto temporary1 =
          h + sum1 + choose + round_constants[index] + words[index];
      const auto sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_{};
  std::uint64_t byte_count_{};
};

std::string hex(std::span<const std::uint8_t> bytes) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(bytes.size() * 2U, '0');
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    result[index * 2U] = digits[bytes[index] >> 4U];
    result[index * 2U + 1U] = digits[bytes[index] & 0x0fU];
  }
  return result;
}

std::uint16_t read_u16(std::span<const std::uint8_t> bytes,
                       std::size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 2U) {
    throw std::runtime_error("Qwen archive metadata is truncated");
  }
  return static_cast<std::uint16_t>(bytes[offset]) |
         (static_cast<std::uint16_t>(bytes[offset + 1U]) << 8U);
}

std::uint32_t read_u32(std::span<const std::uint8_t> bytes,
                       std::size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 4U) {
    throw std::runtime_error("Qwen archive metadata is truncated");
  }
  std::uint32_t result{};
  for (std::size_t index = 0; index < 4U; ++index) {
    result |= static_cast<std::uint32_t>(bytes[offset + index]) << (8U * index);
  }
  return result;
}

std::uint64_t read_u64(std::span<const std::uint8_t> bytes,
                       std::size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 8U) {
    throw std::runtime_error("Qwen archive metadata is truncated");
  }
  std::uint64_t result{};
  for (std::size_t index = 0; index < 8U; ++index) {
    result |= static_cast<std::uint64_t>(bytes[offset + index]) << (8U * index);
  }
  return result;
}

double read_f64(std::span<const std::uint8_t> bytes, std::size_t offset) {
  return std::bit_cast<double>(read_u64(bytes, offset));
}

std::vector<std::uint8_t> read_exact(std::ifstream& input, std::size_t count,
                                     std::string_view label) {
  std::vector<std::uint8_t> result(count);
  if (count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("Qwen archive read exceeds stream limits");
  }
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(count));
  if (!input || static_cast<std::size_t>(input.gcount()) != count) {
    throw std::runtime_error("Qwen archive is truncated while reading " +
                             std::string(label));
  }
  return result;
}

std::string read_name(std::span<const std::uint8_t> bytes,
                      std::size_t& cursor, std::size_t count) {
  if (count == 0U || count > maximum_name_bytes || cursor > bytes.size() ||
      bytes.size() - cursor < count) {
    throw std::runtime_error("Qwen archive contains an invalid tensor name");
  }
  std::string result(reinterpret_cast<const char*>(bytes.data() + cursor), count);
  cursor += count;
  if (result.find('\0') != std::string::npos) {
    throw std::runtime_error("Qwen archive tensor names must not contain NUL");
  }
  return result;
}

std::array<std::uint8_t, 32> hash_region(const std::filesystem::path& path,
                                         std::uint64_t offset,
                                         std::uint64_t length) {
  std::ifstream input(path, std::ios::binary);
  if (!input || offset > static_cast<std::uint64_t>(
                             std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error("could not open/seek Qwen archive for hashing");
  }
  input.seekg(static_cast<std::streamoff>(offset));
  Sha256 hash;
  std::vector<std::uint8_t> buffer(hash_buffer_size);
  while (length != 0U) {
    const auto selected = static_cast<std::size_t>(
        std::min<std::uint64_t>(length, buffer.size()));
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(selected));
    if (!input || static_cast<std::size_t>(input.gcount()) != selected) {
      throw std::runtime_error("Qwen archive ended while hashing a region");
    }
    hash.update(std::span<const std::uint8_t>(buffer.data(), selected));
    length -= selected;
  }
  return hash.finish();
}

struct DenseConfig {
  std::uint64_t vocabulary_size{};
  std::uint64_t maximum_sequence_length{};
  std::uint64_t model_dimension{};
  std::uint64_t layer_count{};
  std::uint64_t query_head_count{};
  std::uint64_t key_value_head_count{};
  std::uint64_t head_dimension{};
  std::uint64_t feed_forward_dimension{};
  double rms_epsilon{};
  double rope_base{};
  std::uint32_t flags{};
};

struct TensorRecord {
  std::string source_name;
  std::string destination_name;
  std::vector<std::int64_t> shape;
  std::uint64_t offset{};
  std::uint64_t length{};
  std::array<std::uint8_t, 32> raw_sha{};
};

struct ParsedArchive {
  DenseConfig config;
  std::string model_id;
  std::string revision;
  std::array<std::uint8_t, 32> source_sha{};
  std::array<std::uint8_t, 32> config_sha{};
  std::array<std::uint8_t, 32> tokenizer_sha{};
  std::array<std::uint8_t, 32> metadata_sha{};
  std::array<std::uint8_t, 32> payload_sha{};
  std::uint64_t payload_offset{};
  std::uint64_t payload_size{};
  std::vector<TensorRecord> tensors;
};

struct ExpectedTensor {
  std::string source;
  std::string destination;
  std::vector<std::int64_t> shape;
};

std::vector<ExpectedTensor> expected_tensors(const DenseConfig& config) {
  const auto hidden = static_cast<std::int64_t>(config.model_dimension);
  const auto query_width = static_cast<std::int64_t>(
      config.query_head_count * config.head_dimension);
  const auto key_value_width = static_cast<std::int64_t>(
      config.key_value_head_count * config.head_dimension);
  const auto feed_forward =
      static_cast<std::int64_t>(config.feed_forward_dimension);
  std::vector<ExpectedTensor> result;
  result.reserve(static_cast<std::size_t>(config.layer_count * 11U + 2U));
  result.push_back({"model.embed_tokens.weight", "token_embedding.weight",
                    {static_cast<std::int64_t>(config.vocabulary_size), hidden}});
  for (std::uint64_t layer = 0; layer < config.layer_count; ++layer) {
    const auto source = "model.layers." + std::to_string(layer);
    const auto destination = "block_" + std::to_string(layer);
    result.push_back({source + ".input_layernorm.weight",
                      destination + ".attention_norm.weight", {hidden}});
    result.push_back({source + ".self_attn.q_proj.weight",
                      destination + ".attention.query.weight",
                      {query_width, hidden}});
    result.push_back({source + ".self_attn.k_proj.weight",
                      destination + ".attention.key.weight",
                      {key_value_width, hidden}});
    result.push_back({source + ".self_attn.v_proj.weight",
                      destination + ".attention.value.weight",
                      {key_value_width, hidden}});
    result.push_back({source + ".self_attn.o_proj.weight",
                      destination + ".attention.output.weight",
                      {hidden, query_width}});
    result.push_back({source + ".self_attn.q_norm.weight",
                      destination + ".attention.q_norm.weight",
                      {static_cast<std::int64_t>(config.head_dimension)}});
    result.push_back({source + ".self_attn.k_norm.weight",
                      destination + ".attention.k_norm.weight",
                      {static_cast<std::int64_t>(config.head_dimension)}});
    result.push_back({source + ".post_attention_layernorm.weight",
                      destination + ".feed_forward_norm.weight", {hidden}});
    result.push_back({source + ".mlp.gate_proj.weight",
                      destination + ".feed_forward_gate.weight",
                      {feed_forward, hidden}});
    result.push_back({source + ".mlp.up_proj.weight",
                      destination + ".feed_forward_up.weight",
                      {feed_forward, hidden}});
    result.push_back({source + ".mlp.down_proj.weight",
                      destination + ".feed_forward_down.weight",
                      {hidden, feed_forward}});
  }
  result.push_back(
      {"model.norm.weight", "final_norm.weight", {hidden}});
  return result;
}

void require_identity(const QwenDenseArchiveIdentity& expected) {
  if (expected.model_id.empty() || !is_git_revision(expected.revision) ||
      !is_sha256(expected.archive_sha256) ||
      !is_sha256(expected.source_checkpoint_sha256) ||
      !is_sha256(expected.config_sha256) ||
      !is_sha256(expected.tokenizer_fingerprint_sha256)) {
    throw std::invalid_argument(
        "Qwen archive identity requires model/revision and all SHA-256 fields");
  }
}

ParsedArchive parse_archive(const std::filesystem::path& path,
                            const QwenDenseArchiveIdentity& expected) {
  require_identity(expected);
  if (sha256_file(path) != expected.archive_sha256) {
    throw std::runtime_error("Qwen archive whole-file SHA-256 mismatch");
  }
  const auto file_size = std::filesystem::file_size(path);
  if (file_size < header_size) {
    throw std::runtime_error("Qwen archive is shorter than its header");
  }
  std::ifstream input(path, std::ios::binary);
  const auto header = read_exact(input, header_size, "header");
  if (!std::equal(archive_magic.begin(), archive_magic.end(), header.begin()) ||
      read_u32(header, 8U) != qwen_dense_archive_version ||
      read_u32(header, 12U) != endian_marker) {
    throw std::runtime_error("Qwen archive magic/version/endianness mismatch");
  }
  const auto tensor_count = read_u32(header, 16U);
  const auto declared_config_size = read_u32(header, 20U);
  const auto metadata_size = read_u64(header, 24U);
  const auto payload_offset = read_u64(header, 32U);
  const auto payload_size = read_u64(header, 40U);
  const auto archive_size = read_u64(header, 48U);
  if (tensor_count == 0U || tensor_count > maximum_tensor_count ||
      declared_config_size != config_size || metadata_size < config_size + 2U ||
      metadata_size > maximum_metadata_bytes ||
      payload_offset < header_size + metadata_size || payload_offset % 64U != 0U ||
      payload_size > archive_size || payload_offset > archive_size - payload_size ||
      payload_offset + payload_size != archive_size || archive_size != file_size) {
    throw std::runtime_error("Qwen archive header bounds are invalid");
  }
  ParsedArchive result;
  std::copy_n(header.begin() + 56, 32, result.source_sha.begin());
  std::copy_n(header.begin() + 88, 32, result.config_sha.begin());
  std::copy_n(header.begin() + 120, 32, result.tokenizer_sha.begin());
  std::copy_n(header.begin() + 152, 32, result.metadata_sha.begin());
  std::copy_n(header.begin() + 184, 32, result.payload_sha.begin());
  result.revision.assign(reinterpret_cast<const char*>(header.data() + 216), 40U);
  result.payload_offset = payload_offset;
  result.payload_size = payload_size;
  if (result.revision != expected.revision ||
      hex(result.source_sha) != expected.source_checkpoint_sha256 ||
      hex(result.config_sha) != expected.config_sha256 ||
      hex(result.tokenizer_sha) != expected.tokenizer_fingerprint_sha256) {
    throw std::runtime_error("Qwen archive immutable identity mismatch");
  }

  const auto metadata = read_exact(
      input, static_cast<std::size_t>(metadata_size), "metadata");
  Sha256 metadata_hasher;
  metadata_hasher.update(metadata);
  if (metadata_hasher.finish() != result.metadata_sha) {
    throw std::runtime_error("Qwen archive metadata SHA-256 mismatch");
  }
  result.config = {
      .vocabulary_size = read_u64(metadata, 0U),
      .maximum_sequence_length = read_u64(metadata, 8U),
      .model_dimension = read_u64(metadata, 16U),
      .layer_count = read_u64(metadata, 24U),
      .query_head_count = read_u64(metadata, 32U),
      .key_value_head_count = read_u64(metadata, 40U),
      .head_dimension = read_u64(metadata, 48U),
      .feed_forward_dimension = read_u64(metadata, 56U),
      .rms_epsilon = read_f64(metadata, 64U),
      .rope_base = read_f64(metadata, 72U),
      .flags = read_u32(metadata, 80U),
  };
  const auto maximum_dimension = static_cast<std::uint64_t>(
      std::numeric_limits<std::int64_t>::max());
  const auto dimensions = std::array{
      result.config.vocabulary_size,
      result.config.maximum_sequence_length,
      result.config.model_dimension,
      result.config.layer_count,
      result.config.query_head_count,
      result.config.key_value_head_count,
      result.config.head_dimension,
      result.config.feed_forward_dimension,
  };
  if (read_u32(metadata, 84U) != 0U || result.config.flags != required_flags ||
      std::any_of(dimensions.begin(), dimensions.end(),
                  [maximum_dimension](const std::uint64_t value) {
                    return value == 0U || value > maximum_dimension;
                  }) ||
      result.config.layer_count > (maximum_tensor_count - 2U) / 11U ||
      result.config.query_head_count % result.config.key_value_head_count != 0U ||
      result.config.head_dimension % 2U != 0U ||
      result.config.query_head_count >
          maximum_dimension / result.config.head_dimension ||
      result.config.key_value_head_count >
          maximum_dimension / result.config.head_dimension ||
      !std::isfinite(result.config.rms_epsilon) ||
      result.config.rms_epsilon <= 0.0 || !std::isfinite(result.config.rope_base) ||
      result.config.rope_base <= 1.0) {
    throw std::runtime_error("Qwen archive decoder configuration is invalid");
  }
  std::size_t cursor = config_size;
  const auto model_id_size = read_u16(metadata, cursor);
  cursor += 2U;
  result.model_id = read_name(metadata, cursor, model_id_size);
  if (result.model_id != expected.model_id) {
    throw std::runtime_error("Qwen archive model id mismatch");
  }

  const auto canonical = expected_tensors(result.config);
  if (canonical.size() != tensor_count) {
    throw std::runtime_error("Qwen archive tensor count does not match its config");
  }
  result.tensors.reserve(tensor_count);
  std::uint64_t next_offset = payload_offset;
  std::unordered_set<std::string> sources;
  std::unordered_set<std::string> destinations;
  for (std::size_t index = 0; index < tensor_count; ++index) {
    if (cursor > metadata.size() || metadata.size() - cursor < tensor_prefix_size) {
      throw std::runtime_error("Qwen archive tensor table is truncated");
    }
    const auto source_size = read_u16(metadata, cursor);
    const auto destination_size = read_u16(metadata, cursor + 2U);
    const auto dtype = metadata[cursor + 4U];
    const auto rank = metadata[cursor + 5U];
    const auto reserved = read_u16(metadata, cursor + 6U);
    TensorRecord tensor;
    tensor.offset = read_u64(metadata, cursor + 8U);
    tensor.length = read_u64(metadata, cursor + 16U);
    std::copy_n(metadata.begin() + static_cast<std::ptrdiff_t>(cursor + 24U),
                32, tensor.raw_sha.begin());
    cursor += tensor_prefix_size;
    if (dtype != dtype_bfloat16 || rank == 0U || rank > maximum_rank ||
        reserved != 0U || tensor.offset != next_offset || tensor.length == 0U ||
        tensor.offset > archive_size || tensor.length > archive_size - tensor.offset) {
      throw std::runtime_error("Qwen archive tensor table entry is invalid");
    }
    std::uint64_t element_count = 1U;
    for (std::size_t dimension = 0; dimension < rank; ++dimension) {
      const auto extent = read_u64(metadata, cursor);
      cursor += 8U;
      if (extent == 0U || extent >
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
          element_count > std::numeric_limits<std::uint64_t>::max() / extent) {
        throw std::runtime_error("Qwen archive tensor shape is invalid");
      }
      element_count *= extent;
      tensor.shape.push_back(static_cast<std::int64_t>(extent));
    }
    if (element_count > std::numeric_limits<std::uint64_t>::max() / 2U ||
        element_count * 2U != tensor.length) {
      throw std::runtime_error("Qwen archive tensor byte length is invalid");
    }
    tensor.source_name = read_name(metadata, cursor, source_size);
    tensor.destination_name = read_name(metadata, cursor, destination_size);
    const auto& expected_tensor = canonical[index];
    if (tensor.source_name != expected_tensor.source ||
        tensor.destination_name != expected_tensor.destination ||
        tensor.shape != expected_tensor.shape ||
        !sources.insert(tensor.source_name).second ||
        !destinations.insert(tensor.destination_name).second) {
      throw std::runtime_error("Qwen archive tensor mapping/shape is not canonical");
    }
    if (next_offset > archive_size - tensor.length) {
      throw std::runtime_error("Qwen archive tensor offsets overflow");
    }
    next_offset += tensor.length;
    result.tensors.push_back(std::move(tensor));
  }
  if (cursor != metadata.size() || next_offset != archive_size) {
    throw std::runtime_error("Qwen archive tensor table does not cover metadata/payload");
  }
  const auto padding_size = payload_offset - header_size - metadata_size;
  const auto padding = read_exact(input, static_cast<std::size_t>(padding_size),
                                  "alignment padding");
  if (std::any_of(padding.begin(), padding.end(),
                  [](std::uint8_t byte) { return byte != 0U; })) {
    throw std::runtime_error("Qwen archive alignment padding is not zero");
  }
  if (hash_region(path, payload_offset, payload_size) != result.payload_sha) {
    throw std::runtime_error("Qwen archive payload SHA-256 mismatch");
  }
  for (const auto& tensor : result.tensors) {
    if (hash_region(path, tensor.offset, tensor.length) != tensor.raw_sha) {
      throw std::runtime_error("Qwen archive raw tensor SHA-256 mismatch: " +
                               tensor.source_name);
    }
  }
  return result;
}

void validate_decoder_config(const DenseConfig& archive,
                             const snnbase::language::DecoderConfig& decoder) {
  const auto positive_equal = [](std::uint64_t archived, std::int64_t actual) {
    return actual > 0 && archived == static_cast<std::uint64_t>(actual);
  };
  if (!positive_equal(archive.vocabulary_size, decoder.vocabulary_size) ||
      decoder.maximum_sequence_length <= 0 ||
      static_cast<std::uint64_t>(decoder.maximum_sequence_length) >
          archive.maximum_sequence_length ||
      !positive_equal(archive.model_dimension, decoder.model_dimension) ||
      !positive_equal(archive.layer_count, decoder.layer_count) ||
      !positive_equal(archive.query_head_count, decoder.query_head_count) ||
      !positive_equal(archive.key_value_head_count,
                      decoder.key_value_head_count) ||
      !positive_equal(archive.head_dimension,
                      snnbase::language::resolved_head_dimension(decoder)) ||
      !positive_equal(archive.feed_forward_dimension,
                      decoder.feed_forward_dimension) ||
      !decoder.query_key_normalization || decoder.rms_epsilon != archive.rms_epsilon ||
      decoder.rope_base != archive.rope_base) {
    throw std::runtime_error("Qwen archive and Decoder configuration mismatch");
  }
}

std::unordered_map<std::string, std::vector<std::int64_t>>
runtime_parameter_shapes(const DenseConfig& config) {
  std::unordered_map<std::string, std::vector<std::int64_t>> result;
  for (std::uint64_t layer = 0; layer < config.layer_count; ++layer) {
    const auto prefix = "block_" + std::to_string(layer) + ".";
    for (const auto& [site, size] :
         std::array<std::pair<std::string_view, std::uint64_t>, 2>{
             std::pair{"attention_lif", config.model_dimension},
             std::pair{"feed_forward_lif", config.feed_forward_dimension}}) {
      result.emplace(prefix + std::string(site) + ".raw_threshold",
                     std::vector<std::int64_t>{static_cast<std::int64_t>(size)});
      result.emplace(prefix + std::string(site) + ".raw_leak",
                     std::vector<std::int64_t>{static_cast<std::int64_t>(size)});
    }
  }
  return result;
}

}  // namespace

QwenDenseLoadResult load_qwen_dense_weights(
    snnbase::language::Decoder& decoder,
    const std::filesystem::path& archive,
    const QwenDenseArchiveIdentity& expected) {
  if (!decoder) {
    throw std::invalid_argument("cannot load Qwen weights into a null Decoder");
  }
  const auto parsed = parse_archive(archive, expected);
  validate_decoder_config(parsed.config, decoder->config());

  std::unordered_map<std::string, torch::Tensor> parameters;
  for (const auto& item : decoder->named_parameters(true)) {
    parameters.emplace(item.key(), item.value());
  }
  const auto runtime = runtime_parameter_shapes(parsed.config);
  if (parameters.size() != parsed.tensors.size() + runtime.size()) {
    throw std::runtime_error("Decoder named-parameter count is not exact for Qwen import");
  }
  for (const auto& tensor : parsed.tensors) {
    const auto found = parameters.find(tensor.destination_name);
    if (found == parameters.end() || !found->second.is_floating_point() ||
        found->second.sizes().vec() != tensor.shape) {
      throw std::runtime_error("Decoder dense parameter missing/invalid: " +
                               tensor.destination_name);
    }
  }
  for (const auto& [name, shape] : runtime) {
    const auto found = parameters.find(name);
    if (found == parameters.end() || !found->second.is_floating_point() ||
        found->second.sizes().vec() != shape) {
      throw std::runtime_error("Decoder runtime LIF parameter missing/invalid: " + name);
    }
  }
  for (const auto& [name, parameter] : parameters) {
    static_cast<void>(parameter);
    const auto dense = std::any_of(
        parsed.tensors.begin(), parsed.tensors.end(),
        [&name](const TensorRecord& tensor) {
          return tensor.destination_name == name;
        });
    if (!dense && !runtime.contains(name)) {
      throw std::runtime_error("undeclared Decoder parameter during Qwen import: " + name);
    }
  }

  std::ifstream input(archive, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not reopen Qwen archive for loading");
  }
  torch::NoGradGuard no_grad;
  for (const auto& tensor : parsed.tensors) {
    if (tensor.offset > static_cast<std::uint64_t>(
                            std::numeric_limits<std::streamoff>::max()) ||
        tensor.length > static_cast<std::uint64_t>(
                            std::numeric_limits<std::streamsize>::max())) {
      throw std::runtime_error("Qwen archive tensor exceeds stream limits");
    }
    input.seekg(static_cast<std::streamoff>(tensor.offset));
    auto source = torch::empty(
        tensor.shape, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));
    input.read(static_cast<char*>(source.data_ptr()),
               static_cast<std::streamsize>(tensor.length));
    if (!input || static_cast<std::uint64_t>(input.gcount()) != tensor.length) {
      throw std::runtime_error("Qwen archive ended while loading " +
                               tensor.destination_name);
    }
    auto& destination = parameters.at(tensor.destination_name);
    destination.copy_(source.to(destination.options()));
  }
  return {
      .model_id = parsed.model_id,
      .revision = parsed.revision,
      .archive_sha256 = expected.archive_sha256,
      .source_checkpoint_sha256 = hex(parsed.source_sha),
      .config_sha256 = hex(parsed.config_sha),
      .tokenizer_fingerprint_sha256 = hex(parsed.tokenizer_sha),
      .metadata_sha256 = hex(parsed.metadata_sha),
      .payload_sha256 = hex(parsed.payload_sha),
      .loaded_tensor_count = parsed.tensors.size(),
      .loaded_payload_bytes = parsed.payload_size,
      .untouched_runtime_parameter_count = runtime.size(),
  };
}

}  // namespace snnbase_experiments::chatbot
