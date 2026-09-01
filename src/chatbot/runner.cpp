#include <snnbase_experiments/chatbot/runner.hpp>

#include <snnbase_experiments/chatbot/manifest.hpp>

#include <ATen/Context.h>
#include <torch/cuda.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/torch.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace snnbase_experiments::chatbot {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::int64_t runner_checkpoint_format_version = 2;
constexpr double pi = 3.141592653589793238462643383279502884;

void set_default_rng_state(const torch::Device& device,
                           const torch::Tensor& state) {
  auto generator = at::globalContext().defaultGenerator(device);
  generator.set_state(state);
}

void seed_default_rng(const torch::Device& device, const std::uint64_t seed) {
  auto generator = at::globalContext().defaultGenerator(device);
  std::lock_guard<std::mutex> lock(generator.mutex());
  generator.set_current_seed(seed);
}

class RngStateGuard {
 public:
  explicit RngStateGuard(torch::Device device)
      : device_(std::move(device)),
        cpu_state_(at::globalContext()
                       .defaultGenerator(torch::Device(torch::kCPU))
                       .get_state()) {
    if (device_.is_cuda()) {
      device_state_ =
          at::globalContext().defaultGenerator(device_).get_state();
    }
  }

  ~RngStateGuard() {
    try {
      set_default_rng_state(torch::Device(torch::kCPU), cpu_state_);
      if (device_state_.defined()) {
        set_default_rng_state(device_, device_state_);
      }
    } catch (...) {
      // Restoring a process-global RNG is best-effort in a noexcept destructor.
    }
  }

  RngStateGuard(const RngStateGuard&) = delete;
  RngStateGuard& operator=(const RngStateGuard&) = delete;

 private:
  torch::Device device_;
  torch::Tensor cpu_state_;
  torch::Tensor device_state_;
};

torch::Device select_device(std::string_view requested) {
  if (requested == "cpu") {
    return torch::Device(torch::kCPU);
  }
  if (requested == "cuda") {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("CUDA was requested but is unavailable");
    }
    return torch::Device(torch::kCUDA);
  }
  if (requested == "auto") {
    return torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                       : torch::Device(torch::kCPU);
  }
  throw std::invalid_argument("device must be auto, cpu, or cuda");
}

void validate_runner_config(const RunnerConfig& config) {
  if (!std::isfinite(config.learning_rate) ||
      !std::isfinite(config.weight_decay) ||
      !std::isfinite(config.gradient_clip_norm) ||
      !(config.learning_rate > 0.0) || !(config.weight_decay >= 0.0) ||
      !(config.gradient_clip_norm > 0.0) ||
      config.gradient_accumulation_steps == 0U ||
      config.gradient_accumulation_steps >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      config.total_optimizer_steps >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      config.warmup_optimizer_steps >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      !std::isfinite(config.minimum_learning_rate_ratio) ||
      config.minimum_learning_rate_ratio < 0.0 ||
      config.minimum_learning_rate_ratio > 1.0 ||
      (config.total_optimizer_steps == 0U &&
       config.warmup_optimizer_steps != 0U) ||
      (config.total_optimizer_steps != 0U &&
       config.warmup_optimizer_steps > config.total_optimizer_steps) ||
      !std::isfinite(config.spike_rate_target) ||
      config.spike_rate_target < 0.0 || config.spike_rate_target > 1.0 ||
      !std::isfinite(config.spike_rate_penalty) ||
      config.spike_rate_penalty < 0.0) {
    throw std::invalid_argument("invalid chatbot optimizer configuration");
  }
}

void validate_generation(const GenerationConfig& generation,
                         const snnbase::language::DecoderConfig& model) {
  const auto sampling_vocabulary_size =
      generation.sampling_vocabulary_size == 0U
          ? static_cast<std::size_t>(model.vocabulary_size)
          : generation.sampling_vocabulary_size;
  if (generation.maximum_new_tokens == 0U ||
      !std::isfinite(generation.temperature) || generation.temperature < 0.0 ||
      !std::isfinite(generation.top_p) || generation.top_p <= 0.0 ||
      generation.top_p > 1.0 ||
      generation.maximum_new_tokens >
          static_cast<std::size_t>(model.maximum_sequence_length) ||
      sampling_vocabulary_size == 0U ||
      sampling_vocabulary_size >
          static_cast<std::size_t>(model.vocabulary_size)) {
    throw std::invalid_argument("invalid generation configuration");
  }
  if (generation.top_k > sampling_vocabulary_size) {
    throw std::invalid_argument("top_k exceeds the vocabulary size");
  }
  for (const auto token : generation.eos_token_ids) {
    if (token < 0 ||
        static_cast<std::size_t>(token) >= sampling_vocabulary_size) {
      throw std::invalid_argument("stop token is outside the vocabulary");
    }
  }
}

torch::Tensor make_token_tensor(std::span<const std::int64_t> input_ids,
                                const torch::Device& device) {
  if (input_ids.empty()) {
    throw std::invalid_argument("input_ids must not be empty");
  }
  auto tensor = torch::from_blob(
                    const_cast<std::int64_t*>(input_ids.data()),
                    {1, static_cast<std::int64_t>(input_ids.size())},
                    torch::TensorOptions().dtype(torch::kInt64))
                    .clone();
  return tensor.to(device);
}

torch::Tensor sample_next_token(torch::Tensor logits,
                                const GenerationConfig& generation) {
  if (generation.sampling_vocabulary_size != 0U) {
    logits = logits.narrow(
        -1, 0,
        static_cast<std::int64_t>(generation.sampling_vocabulary_size));
  }
  if (generation.temperature == 0.0) {
    return logits.argmax(-1, true);
  }
  logits = logits / generation.temperature;
  if (generation.top_k > 0U &&
      generation.top_k < static_cast<std::size_t>(logits.size(-1))) {
    const auto selected = logits.topk(
        static_cast<std::int64_t>(generation.top_k), -1, true, true);
    const auto threshold =
        std::get<0>(selected).select(-1,
                                     static_cast<std::int64_t>(
                                         generation.top_k - 1U))
            .unsqueeze(-1);
    logits = logits.masked_fill(
        logits < threshold, -std::numeric_limits<double>::infinity());
  }
  if (generation.top_p < 1.0) {
    const auto sorted = logits.sort(-1, true);
    auto sorted_logits = std::get<0>(sorted);
    const auto sorted_indices = std::get<1>(sorted);
    const auto probabilities = torch::softmax(sorted_logits, -1);
    const auto cumulative = probabilities.cumsum(-1);
    const auto remove = (cumulative - probabilities) >= generation.top_p;
    sorted_logits = sorted_logits.masked_fill(
        remove, -std::numeric_limits<double>::infinity());
    const auto retained_probabilities = torch::softmax(sorted_logits, -1);
    const auto selected_position = torch::multinomial(retained_probabilities, 1);
    return sorted_indices.gather(-1, selected_position);
  }
  return torch::multinomial(torch::softmax(logits, -1), 1);
}

bool is_stop_token(std::int64_t token,
                   const std::vector<std::int64_t>& stop_tokens) {
  return std::find(stop_tokens.begin(), stop_tokens.end(), token) !=
         stop_tokens.end();
}

struct LossComputation {
  torch::Tensor negative_log_likelihood;
  torch::Tensor objective;
  std::size_t token_count{};
  std::size_t correct_token_count{};
  torch::Tensor spike_rate;
};

LossComputation compute_loss(
    snnbase::language::Decoder& model, const torch::Tensor& tokens,
    std::span<const std::uint8_t> loss_mask, const RunnerConfig& config) {
  if (tokens.size(1) < 2) {
    throw std::invalid_argument("language-model loss requires at least two tokens");
  }
  if (!loss_mask.empty() &&
      loss_mask.size() != static_cast<std::size_t>(tokens.size(1))) {
    throw std::invalid_argument("loss_mask length does not match input_ids");
  }

  const auto output = model->forward(tokens);
  auto logits = output.logits.slice(1, 0, tokens.size(1) - 1)
                    .reshape({tokens.size(1) - 1,
                              output.logits.size(2)});
  auto targets = tokens.slice(1, 1, tokens.size(1)).reshape({-1});
  std::size_t selected_count = static_cast<std::size_t>(targets.size(0));
  if (!loss_mask.empty()) {
    for (const auto value : loss_mask) {
      if (value > 1U) {
        throw std::invalid_argument("loss_mask values must be zero or one");
      }
    }
    auto mask = torch::from_blob(
                    const_cast<std::uint8_t*>(loss_mask.data()),
                    {static_cast<std::int64_t>(loss_mask.size())},
                    torch::TensorOptions().dtype(torch::kUInt8))
                    .clone()
                    .to(tokens.device())
                    .slice(0, 1, tokens.size(1))
                    .to(torch::kBool);
    const auto indices = torch::nonzero(mask).reshape({-1});
    selected_count = static_cast<std::size_t>(indices.numel());
    if (selected_count == 0U) {
      throw std::invalid_argument("loss_mask selects no target tokens");
    }
    logits = logits.index_select(0, indices);
    targets = targets.index_select(0, indices);
  }
  auto negative_log_likelihood =
      torch::nn::functional::cross_entropy(logits, targets);
  const auto correct_token_count = static_cast<std::size_t>(
      logits.argmax(-1).eq(targets).sum().item<std::int64_t>());
  auto objective = negative_log_likelihood;
  if (config.spike_rate_penalty > 0.0 && config.model.spiking) {
    objective = objective +
                config.spike_rate_penalty *
                    (output.spikes.mean_rate - config.spike_rate_target)
                        .square();
  }
  return {.negative_log_likelihood = std::move(negative_log_likelihood),
          .objective = std::move(objective),
          .token_count = selected_count,
          .correct_token_count = correct_token_count,
          .spike_rate = output.spikes.mean_rate};
}

LanguageModelMetrics materialize_metrics(const LossComputation& computation,
                                         const TrainingState& state,
                                         const bool optimizer_updated) {
  const auto loss = computation.negative_log_likelihood.item<double>();
  return {.token_count = computation.token_count,
          .correct_token_count = computation.correct_token_count,
          .loss = loss,
          .objective_loss = computation.objective.item<double>(),
          .perplexity = std::exp(std::min(loss, 80.0)),
          .token_accuracy =
              static_cast<double>(computation.correct_token_count) /
              static_cast<double>(computation.token_count),
          .mean_spike_rate = computation.spike_rate.item<double>(),
          .optimizer_updated = optimizer_updated,
          .micro_batch_count = state.micro_batch_count,
          .optimizer_step_count = state.optimizer_step_count,
          .learning_rate = state.learning_rate};
}

double learning_rate_for_step(const RunnerConfig& config,
                              const std::uint64_t one_based_step) {
  if (config.total_optimizer_steps == 0U) {
    return config.learning_rate;
  }
  if (config.warmup_optimizer_steps > 0U &&
      one_based_step <= config.warmup_optimizer_steps) {
    return config.learning_rate *
           static_cast<double>(one_based_step) /
           static_cast<double>(config.warmup_optimizer_steps);
  }
  // The peak is the final warmup update, or the first update when warmup is
  // disabled.  Cosine decay spans the intervals after that peak.  This keeps
  // a one-update schedule at the configured learning rate instead of applying
  // the terminal (often zero) rate to its only optimizer update.
  const auto peak_step =
      std::max<std::uint64_t>(config.warmup_optimizer_steps, 1U);
  const auto decay_intervals = config.total_optimizer_steps - peak_step;
  if (decay_intervals == 0U) {
    return config.learning_rate;
  }
  const auto elapsed =
      one_based_step > peak_step ? one_based_step - peak_step : 0U;
  const auto progress = std::min(
      1.0,
      static_cast<double>(elapsed) / static_cast<double>(decay_intervals));
  const auto cosine = 0.5 * (1.0 + std::cos(pi * progress));
  return config.learning_rate *
         (config.minimum_learning_rate_ratio +
          (1.0 - config.minimum_learning_rate_ratio) * cosine);
}

torch::Tensor model_integer_signature(
    const snnbase::language::DecoderConfig& config) {
  return torch::tensor(
      {config.vocabulary_size, config.maximum_sequence_length,
       config.model_dimension, config.layer_count, config.query_head_count,
       config.key_value_head_count, config.head_dimension,
       config.feed_forward_dimension, config.simulation_steps,
       static_cast<std::int64_t>(config.spiking),
       static_cast<std::int64_t>(config.query_key_normalization),
       static_cast<std::int64_t>(config.lif.learn_threshold),
       static_cast<std::int64_t>(config.lif.learn_leak),
       static_cast<std::int64_t>(config.lif.signed_spikes)},
      torch::TensorOptions().dtype(torch::kInt64));
}

torch::Tensor model_floating_signature(
    const snnbase::language::DecoderConfig& config) {
  return torch::tensor(
      {config.rms_epsilon, config.rope_base,
       static_cast<double>(config.lif.initial_threshold),
       static_cast<double>(config.lif.initial_leak),
       static_cast<double>(config.lif.surrogate_slope)},
      torch::TensorOptions().dtype(torch::kFloat64));
}

torch::Tensor runner_integer_signature(const RunnerConfig& config) {
  return torch::tensor(
      {static_cast<std::int64_t>(config.gradient_accumulation_steps),
       static_cast<std::int64_t>(config.total_optimizer_steps),
       static_cast<std::int64_t>(config.warmup_optimizer_steps)},
      torch::TensorOptions().dtype(torch::kInt64));
}

torch::Tensor runner_floating_signature(const RunnerConfig& config) {
  return torch::tensor(
      {config.learning_rate, config.weight_decay, config.gradient_clip_norm,
       config.minimum_learning_rate_ratio, config.spike_rate_target,
       config.spike_rate_penalty},
      torch::TensorOptions().dtype(torch::kFloat64));
}

void require_signature(torch::serialize::InputArchive& archive,
                       const std::string& name,
                       const torch::Tensor& expected) {
  torch::Tensor actual;
  if (!archive.try_read(name, actual) ||
      !torch::equal(actual.to(torch::kCPU), expected)) {
    throw std::runtime_error("chatbot checkpoint configuration mismatch: " +
                             name);
  }
}

std::string read_checkpoint_string(torch::serialize::InputArchive& archive,
                                   const std::string& name) {
  c10::IValue value;
  if (!archive.try_read(name, value) || !value.isString()) {
    throw std::runtime_error("chatbot checkpoint is missing string: " + name);
  }
  return value.toStringRef();
}

std::string json_escape(std::string_view value) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(value.size());
  for (const auto character : value) {
    const auto byte = static_cast<unsigned char>(character);
    switch (character) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\b':
        result += "\\b";
        break;
      case '\f':
        result += "\\f";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          result += "\\u00";
          result.push_back(hex[byte >> 4U]);
          result.push_back(hex[byte & 0x0fU]);
        } else {
          result.push_back(character);
        }
    }
  }
  return result;
}

class ProtocolParser {
 public:
  struct Request {
    std::string protocol;
    std::string request_id;
    std::string operation;
    std::vector<std::int64_t> input_ids;
    std::vector<std::uint8_t> loss_mask;
    std::vector<std::int64_t> probe_token_ids;
    GenerationConfig generation;
    bool reset_state{};
    std::unordered_set<std::string> fields;
  };

  ProtocolParser(std::string_view line, const ProtocolLimits& limits)
      : line_(line), limits_(limits) {}

  Request parse() {
    Request request;
    bool saw_protocol = false;
    bool saw_request_id = false;
    bool saw_operation = false;
    skip_whitespace();
    expect('{');
    skip_whitespace();
    if (consume('}')) {
      fail("request object is empty");
    }
    while (true) {
      const auto key = parse_string();
      if (!request.fields.insert(key).second) {
        fail("duplicate request field: " + key);
      }
      skip_whitespace();
      expect(':');
      skip_whitespace();
      if (key == "protocol") {
        request.protocol = parse_string();
        saw_protocol = true;
      } else if (key == "request_id") {
        request.request_id = parse_string();
        saw_request_id = true;
      } else if (key == "op") {
        request.operation = parse_string();
        saw_operation = true;
      } else if (key == "input_ids") {
        request.input_ids = parse_integer_array(limits_.maximum_tokens,
                                                "input_ids");
      } else if (key == "loss_mask") {
        const auto values =
            parse_integer_array(limits_.maximum_tokens, "loss_mask");
        request.loss_mask.reserve(values.size());
        for (const auto value : values) {
          if (value > 1) {
            fail("loss_mask values must be zero or one");
          }
          request.loss_mask.push_back(static_cast<std::uint8_t>(value));
        }
      } else if (key == "probe_token_ids") {
        request.probe_token_ids = parse_integer_array(
            limits_.maximum_eos_tokens, "probe_token_ids");
      } else if (key == "max_new_tokens") {
        request.generation.maximum_new_tokens = checked_size(parse_uint64());
      } else if (key == "sampling_vocabulary_size") {
        request.generation.sampling_vocabulary_size =
            checked_size(parse_uint64());
      } else if (key == "seed") {
        request.generation.seed = parse_uint64();
      } else if (key == "temperature") {
        request.generation.temperature = parse_number();
      } else if (key == "top_k") {
        request.generation.top_k = checked_size(parse_uint64());
      } else if (key == "top_p") {
        request.generation.top_p = parse_number();
      } else if (key == "eos_token_ids") {
        request.generation.eos_token_ids = parse_integer_array(
            limits_.maximum_eos_tokens, "eos_token_ids");
      } else if (key == "reset_state") {
        request.reset_state = parse_bool();
      } else {
        fail("unknown request field: " + key);
      }
      skip_whitespace();
      if (consume('}')) {
        break;
      }
      expect(',');
      skip_whitespace();
    }
    skip_whitespace();
    if (position_ != line_.size()) {
      fail("trailing characters after request object");
    }
    if (!saw_protocol || !saw_request_id || !saw_operation) {
      fail("request requires protocol, request_id, and op");
    }
    if (request.protocol != token_protocol) {
      fail("unsupported token protocol");
    }
    if (request.request_id.empty() ||
        request.request_id.size() > limits_.maximum_request_id_bytes) {
      fail("request_id is empty or exceeds its byte limit");
    }
    return request;
  }

  static void validate_operation_contract(const Request& request) {
    const auto validate = [&request](
                              const std::initializer_list<std::string_view>
                                  required,
                              const std::initializer_list<std::string_view>
                                  optional) {
      std::unordered_set<std::string> allowed{"protocol", "request_id",
                                              "op"};
      for (const auto field : required) {
        allowed.emplace(field);
        if (!request.fields.contains(std::string(field))) {
          throw std::invalid_argument(request.operation +
                                      " requires request field: " +
                                      std::string(field));
        }
      }
      for (const auto field : optional) {
        allowed.emplace(field);
      }
      std::vector<std::string> extras;
      for (const auto& field : request.fields) {
        if (!allowed.contains(field)) {
          extras.push_back(field);
        }
      }
      if (!extras.empty()) {
        std::sort(extras.begin(), extras.end());
        throw std::invalid_argument(request.operation +
                                    " does not accept request field: " +
                                    extras.front());
      }
    };

    if (request.operation == "generate") {
      validate({"input_ids", "max_new_tokens", "seed",
                "sampling_vocabulary_size"},
               {"temperature", "top_k", "top_p", "eos_token_ids"});
    } else if (request.operation == "inspect") {
      validate({"input_ids", "probe_token_ids", "top_k"}, {});
    } else if (request.operation == "train" ||
               request.operation == "evaluate") {
      validate({"input_ids"}, {"loss_mask", "reset_state"});
    } else if (request.operation == "metadata" ||
               request.operation == "reset" ||
               request.operation == "flush" ||
               request.operation == "shutdown") {
      validate({}, {});
    }
  }

 private:
  [[noreturn]] void fail(std::string message) const {
    throw std::invalid_argument(std::move(message));
  }

  static std::size_t checked_size(const std::uint64_t value) {
    if (value > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("integer exceeds size_t");
    }
    return static_cast<std::size_t>(value);
  }

  static std::uint32_t hex_digit(const char character) {
    if (character >= '0' && character <= '9') {
      return static_cast<std::uint32_t>(character - '0');
    }
    if (character >= 'a' && character <= 'f') {
      return static_cast<std::uint32_t>(character - 'a' + 10);
    }
    if (character >= 'A' && character <= 'F') {
      return static_cast<std::uint32_t>(character - 'A' + 10);
    }
    throw std::invalid_argument("invalid hexadecimal digit");
  }

  std::uint32_t parse_hex_quad() {
    if (position_ + 4U > line_.size()) {
      fail("truncated Unicode escape");
    }
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4U; ++index) {
      try {
        value = (value << 4U) | hex_digit(line_[position_++]);
      } catch (const std::invalid_argument&) {
        fail("invalid Unicode escape");
      }
    }
    return value;
  }

  static void append_utf8(std::string& output, const std::uint32_t codepoint) {
    if (codepoint <= 0x7fU) {
      output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
      output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
      output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
      output.push_back(
          static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
      output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
      output.push_back(
          static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
      output.push_back(
          static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
  }

  void parse_unicode_escape(std::string& output) {
    auto codepoint = parse_hex_quad();
    if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
      if (position_ + 2U > line_.size() || line_[position_] != '\\' ||
          line_[position_ + 1U] != 'u') {
        fail("high surrogate is not followed by a low surrogate");
      }
      position_ += 2U;
      const auto low = parse_hex_quad();
      if (low < 0xdc00U || low > 0xdfffU) {
        fail("invalid low surrogate");
      }
      codepoint =
          0x10000U + ((codepoint - 0xd800U) << 10U) + (low - 0xdc00U);
    } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
      fail("unpaired low surrogate");
    }
    append_utf8(output, codepoint);
  }

  std::string parse_string() {
    expect('"');
    std::string value;
    while (position_ < line_.size()) {
      const auto character = line_[position_++];
      if (character == '"') {
        return value;
      }
      if (static_cast<unsigned char>(character) < 0x20U) {
        fail("unescaped control character in JSON string");
      }
      if (character != '\\') {
        value.push_back(character);
        continue;
      }
      if (position_ == line_.size()) {
        fail("truncated JSON escape");
      }
      switch (const auto escaped = line_[position_++]) {
        case '"':
        case '\\':
        case '/':
          value.push_back(escaped);
          break;
        case 'b':
          value.push_back('\b');
          break;
        case 'f':
          value.push_back('\f');
          break;
        case 'n':
          value.push_back('\n');
          break;
        case 'r':
          value.push_back('\r');
          break;
        case 't':
          value.push_back('\t');
          break;
        case 'u':
          parse_unicode_escape(value);
          break;
        default:
          fail("unsupported JSON escape");
      }
    }
    fail("unterminated JSON string");
  }

  std::string_view parse_number_text() {
    const auto begin = position_;
    if (consume('-')) {
    }
    if (position_ == line_.size()) {
      fail("truncated JSON number");
    }
    if (consume('0')) {
      if (position_ < line_.size() && line_[position_] >= '0' &&
          line_[position_] <= '9') {
        fail("JSON number has a leading zero");
      }
    } else {
      if (line_[position_] < '1' || line_[position_] > '9') {
        fail("invalid JSON number");
      }
      while (position_ < line_.size() && line_[position_] >= '0' &&
             line_[position_] <= '9') {
        ++position_;
      }
    }
    if (consume('.')) {
      const auto fraction = position_;
      while (position_ < line_.size() && line_[position_] >= '0' &&
             line_[position_] <= '9') {
        ++position_;
      }
      if (position_ == fraction) {
        fail("JSON number has no fractional digits");
      }
    }
    if (position_ < line_.size() &&
        (line_[position_] == 'e' || line_[position_] == 'E')) {
      ++position_;
      if (position_ < line_.size() &&
          (line_[position_] == '+' || line_[position_] == '-')) {
        ++position_;
      }
      const auto exponent = position_;
      while (position_ < line_.size() && line_[position_] >= '0' &&
             line_[position_] <= '9') {
        ++position_;
      }
      if (position_ == exponent) {
        fail("JSON number has no exponent digits");
      }
    }
    return line_.substr(begin, position_ - begin);
  }

  std::uint64_t parse_uint64() {
    const auto text = parse_number_text();
    if (text.empty() || text.front() == '-' || text.find_first_of(".eE") !=
                                                  std::string_view::npos) {
      fail("expected a nonnegative JSON integer");
    }
    std::uint64_t value = 0;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
      fail("JSON integer is out of range");
    }
    return value;
  }

  double parse_number() {
    const auto text = parse_number_text();
    std::string owned(text);
    char* end = nullptr;
    errno = 0;
    const auto value = std::strtod(owned.c_str(), &end);
    if (errno == ERANGE || end != owned.c_str() + owned.size() ||
        !std::isfinite(value)) {
      fail("JSON number is out of range");
    }
    return value;
  }

  bool parse_bool() {
    if (line_.substr(position_, 4) == "true") {
      position_ += 4;
      return true;
    }
    if (line_.substr(position_, 5) == "false") {
      position_ += 5;
      return false;
    }
    fail("expected a JSON boolean");
  }

  std::vector<std::int64_t> parse_integer_array(const std::size_t limit,
                                                std::string_view label) {
    std::vector<std::int64_t> result;
    expect('[');
    skip_whitespace();
    if (consume(']')) {
      return result;
    }
    while (true) {
      if (result.size() >= limit) {
        fail(std::string(label) + " exceeds its element limit");
      }
      const auto value = parse_uint64();
      if (value > static_cast<std::uint64_t>(
                      std::numeric_limits<std::int64_t>::max())) {
        fail(std::string(label) + " contains an out-of-range integer");
      }
      result.push_back(static_cast<std::int64_t>(value));
      skip_whitespace();
      if (consume(']')) {
        break;
      }
      expect(',');
      skip_whitespace();
    }
    return result;
  }

  void skip_whitespace() noexcept {
    while (position_ < line_.size() &&
           (line_[position_] == ' ' || line_[position_] == '\t' ||
            line_[position_] == '\r' || line_[position_] == '\n')) {
      ++position_;
    }
  }

  bool consume(const char expected) noexcept {
    if (position_ < line_.size() && line_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void expect(const char expected) {
    if (!consume(expected)) {
      fail(std::string("expected '") + expected + "'");
    }
  }

  std::string_view line_;
  const ProtocolLimits& limits_;
  std::size_t position_{};
};

enum class LineStatus { line, end, too_long };

LineStatus read_bounded_line(std::istream& input, std::string& line,
                             const std::size_t maximum_bytes) {
  line.clear();
  bool too_long = false;
  char character = 0;
  while (input.get(character)) {
    if (character == '\n') {
      return too_long ? LineStatus::too_long : LineStatus::line;
    }
    if (line.size() < maximum_bytes) {
      line.push_back(character);
    } else {
      too_long = true;
    }
  }
  if (input.bad()) {
    throw std::runtime_error("failed while reading token protocol input");
  }
  if (line.empty() && !too_long) {
    return LineStatus::end;
  }
  return too_long ? LineStatus::too_long : LineStatus::line;
}

void write_prefix(std::ostream& output, std::string_view request_id,
                  const bool ok) {
  output << "{\"protocol\":\"" << token_protocol
         << "\",\"request_id\":\"" << json_escape(request_id)
         << "\",\"ok\":" << (ok ? "true" : "false");
}

void write_error(std::ostream& output, std::string_view request_id,
                 std::string_view code, std::string_view message) {
  write_prefix(output, request_id, false);
  output << ",\"error\":{\"code\":\"" << json_escape(code)
         << "\",\"message\":\"" << json_escape(message) << "\"}}\n";
  output.flush();
}

void write_metrics(std::ostream& output, const LanguageModelMetrics& metrics) {
  output << "{\"token_count\":" << metrics.token_count
         << ",\"correct_token_count\":" << metrics.correct_token_count
         << ",\"loss\":" << std::setprecision(17) << metrics.loss
         << ",\"objective_loss\":" << metrics.objective_loss
         << ",\"perplexity\":" << metrics.perplexity
         << ",\"token_accuracy\":" << metrics.token_accuracy
         << ",\"mean_spike_rate\":" << metrics.mean_spike_rate
         << ",\"optimizer_updated\":"
         << (metrics.optimizer_updated ? "true" : "false")
         << ",\"micro_batch_count\":" << metrics.micro_batch_count
         << ",\"optimizer_step_count\":" << metrics.optimizer_step_count
         << ",\"learning_rate\":" << metrics.learning_rate << '}';
}

void write_training_state(std::ostream& output, const TrainingState& state) {
  output << "{\"micro_batch_count\":" << state.micro_batch_count
         << ",\"optimizer_step_count\":" << state.optimizer_step_count
         << ",\"pending_accumulation_steps\":"
         << state.pending_accumulation_steps << ",\"learning_rate\":"
         << std::setprecision(17) << state.learning_rate << '}';
}

}  // namespace

struct Runner::Impl {
  explicit Impl(RunnerConfig selected_config)
      : config(std::move(selected_config)),
        selected_device(select_device(config.device)),
        model(config.model),
        optimizer(model->parameters(),
                  torch::optim::AdamWOptions(config.learning_rate)
                      .weight_decay(config.weight_decay)) {
    validate_runner_config(config);
    model->to(selected_device);
    set_learning_rate(learning_rate_for_step(config, 1U));
  }

  void set_learning_rate(const double value) {
    optimizer.defaults().set_lr(value);
    for (auto& group : optimizer.param_groups()) {
      group.options().set_lr(value);
    }
    current_learning_rate = value;
  }

  void apply_optimizer_update(const std::size_t accumulated_steps) {
    if (accumulated_steps == 0U) {
      return;
    }
    if (accumulated_steps < config.gradient_accumulation_steps) {
      const auto correction =
          static_cast<double>(config.gradient_accumulation_steps) /
          static_cast<double>(accumulated_steps);
      for (const auto& parameter : model->parameters()) {
        if (parameter.grad().defined()) {
          parameter.grad().mul_(correction);
        }
      }
    }
    set_learning_rate(
        learning_rate_for_step(config, optimizer_step_count + 1U));
    torch::nn::utils::clip_grad_norm_(model->parameters(),
                                     config.gradient_clip_norm);
    optimizer.step();
    ++optimizer_step_count;
    pending_accumulation_steps = 0U;
    optimizer.zero_grad();
  }

  [[nodiscard]] TrainingState state() const noexcept {
    return {.micro_batch_count = micro_batch_count,
            .optimizer_step_count = optimizer_step_count,
            .pending_accumulation_steps = pending_accumulation_steps,
            .learning_rate = current_learning_rate};
  }

  RunnerConfig config;
  torch::Device selected_device;
  snnbase::language::Decoder model;
  torch::optim::AdamW optimizer;
  std::uint64_t micro_batch_count{};
  std::uint64_t optimizer_step_count{};
  std::size_t pending_accumulation_steps{};
  double current_learning_rate{};
  std::optional<QwenDenseLoadResult> qwen_weights;
};

Runner::Runner(RunnerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

Runner::~Runner() = default;
Runner::Runner(Runner&&) noexcept = default;
Runner& Runner::operator=(Runner&&) noexcept = default;

GenerationResult Runner::generate(
    const std::span<const std::int64_t> input_ids,
    const GenerationConfig& generation) {
  validate_generation(generation, impl_->config.model);
  if (input_ids.size() + generation.maximum_new_tokens >
      static_cast<std::size_t>(impl_->config.model.maximum_sequence_length)) {
    throw std::invalid_argument("prompt plus completion exceeds model context");
  }
  RngStateGuard restore_rng(impl_->selected_device);
  seed_default_rng(torch::Device(torch::kCPU), generation.seed);
  if (impl_->selected_device.is_cuda()) {
    seed_default_rng(impl_->selected_device, generation.seed);
  }
  torch::NoGradGuard no_grad;
  const auto was_training = impl_->model->is_training();
  impl_->model->eval();
  impl_->model->reset_state();
  const auto started = Clock::now();
  auto tokens = make_token_tensor(input_ids, impl_->selected_device);
  snnbase::language::KeyValueCache cache;
  auto output = impl_->model->forward_cached(tokens, cache);
  GenerationResult result;
  result.output_ids.reserve(generation.maximum_new_tokens);
  double rate_sum = 0.0;
  std::size_t rate_samples = 0;
  for (std::size_t index = 0; index < generation.maximum_new_tokens; ++index) {
    rate_sum += output.spikes.mean_rate.item<double>();
    ++rate_samples;
    const auto logits = output.logits.select(1, output.logits.size(1) - 1);
    const auto next = sample_next_token(logits, generation);
    const auto token = next.item<std::int64_t>();
    result.output_ids.push_back(token);
    if (index == 0U) {
      result.metrics.time_to_first_token_milliseconds =
          std::chrono::duration<double, std::milli>(Clock::now() - started)
              .count();
    }
    if (is_stop_token(token, generation.eos_token_ids)) {
      result.finish_reason = "eos";
      break;
    }
    if (index + 1U < generation.maximum_new_tokens) {
      output = impl_->model->forward_cached(next, cache);
    }
  }
  const auto elapsed = std::chrono::duration<double, std::milli>(
      Clock::now() - started);
  result.metrics = {
      .elapsed_milliseconds = elapsed.count(),
      .time_to_first_token_milliseconds =
          result.metrics.time_to_first_token_milliseconds,
      .generated_tokens_per_second =
          elapsed.count() <= 0.0
              ? 0.0
              : 1000.0 * static_cast<double>(result.output_ids.size()) /
                    elapsed.count(),
      .mean_spike_rate = rate_samples == 0U ? 0.0 : rate_sum / rate_samples,
      .generated_tokens = result.output_ids.size()};
  impl_->model->train(was_training);
  impl_->model->reset_state();
  return result;
}

LogitInspection Runner::inspect_next_token_logits(
    const std::span<const std::int64_t> input_ids,
    const std::span<const std::int64_t> probe_token_ids,
    const std::size_t top_k) {
  if (probe_token_ids.empty()) {
    throw std::invalid_argument("logit inspection requires probe_token_ids");
  }
  if (top_k == 0U ||
      top_k > static_cast<std::size_t>(impl_->config.model.vocabulary_size)) {
    throw std::invalid_argument("logit inspection top_k is outside the vocabulary");
  }
  std::unordered_set<std::int64_t> unique_probes;
  for (const auto token : probe_token_ids) {
    if (token < 0 || token >= impl_->config.model.vocabulary_size ||
        !unique_probes.insert(token).second) {
      throw std::invalid_argument(
          "probe_token_ids must be unique and inside the vocabulary");
    }
  }
  torch::NoGradGuard no_grad;
  const auto was_training = impl_->model->is_training();
  impl_->model->eval();
  impl_->model->reset_state();
  const auto tokens = make_token_tensor(input_ids, impl_->selected_device);
  const auto output = impl_->model->forward(tokens);
  const auto final_logits = output.logits
                                .select(0, 0)
                                .select(0, output.logits.size(1) - 1)
                                .to(torch::kFloat32)
                                .to(torch::kCPU)
                                .contiguous();
  LogitInspection result;
  result.probes.reserve(probe_token_ids.size());
  for (const auto token : probe_token_ids) {
    result.probes.push_back(
        {.token_id = token, .logit = final_logits[token].item<double>()});
  }
  const auto selected = final_logits.topk(
      static_cast<std::int64_t>(top_k), -1, true, true);
  const auto values = std::get<0>(selected);
  const auto indices = std::get<1>(selected);
  result.top_k.reserve(top_k);
  for (std::size_t index = 0; index < top_k; ++index) {
    const auto position = static_cast<std::int64_t>(index);
    result.top_k.push_back({.token_id = indices[position].item<std::int64_t>(),
                            .logit = values[position].item<double>()});
  }
  impl_->model->train(was_training);
  impl_->model->reset_state();
  return result;
}

LanguageModelMetrics Runner::train_step(
    const std::span<const std::int64_t> input_ids,
    const std::span<const std::uint8_t> loss_mask) {
  auto tokens = make_token_tensor(input_ids, impl_->selected_device);
  impl_->model->train();
  impl_->model->reset_state();
  if (impl_->pending_accumulation_steps == 0U) {
    impl_->optimizer.zero_grad();
  }
  auto computation =
      compute_loss(impl_->model, tokens, loss_mask, impl_->config);
  (computation.objective /
   static_cast<double>(impl_->config.gradient_accumulation_steps))
      .backward();
  ++impl_->micro_batch_count;
  ++impl_->pending_accumulation_steps;
  const auto optimizer_updated =
      impl_->pending_accumulation_steps ==
      impl_->config.gradient_accumulation_steps;
  if (optimizer_updated) {
    impl_->apply_optimizer_update(impl_->pending_accumulation_steps);
  }
  impl_->model->reset_state();
  return materialize_metrics(computation, impl_->state(), optimizer_updated);
}

LanguageModelMetrics Runner::evaluate(
    const std::span<const std::int64_t> input_ids,
    const std::span<const std::uint8_t> loss_mask) {
  torch::NoGradGuard no_grad;
  const auto was_training = impl_->model->is_training();
  impl_->model->eval();
  impl_->model->reset_state();
  const auto tokens = make_token_tensor(input_ids, impl_->selected_device);
  const auto computation =
      compute_loss(impl_->model, tokens, loss_mask, impl_->config);
  const auto metrics = materialize_metrics(computation, impl_->state(), false);
  impl_->model->train(was_training);
  impl_->model->reset_state();
  return metrics;
}

TrainingState Runner::flush_optimizer() {
  impl_->apply_optimizer_update(impl_->pending_accumulation_steps);
  impl_->model->reset_state();
  return impl_->state();
}

void Runner::reset_state() noexcept { impl_->model->reset_state(); }

QwenDenseLoadResult Runner::load_qwen_weights(
    const std::filesystem::path& archive,
    const QwenDenseArchiveIdentity& identity) {
  if (impl_->micro_batch_count != 0U || impl_->optimizer_step_count != 0U ||
      impl_->pending_accumulation_steps != 0U) {
    throw std::runtime_error(
        "Qwen weights may only be loaded before runner training begins");
  }
  auto result = snnbase_experiments::chatbot::load_qwen_dense_weights(
      impl_->model, archive, identity);
  impl_->qwen_weights = result;
  impl_->model->to(impl_->selected_device);
  impl_->model->reset_state();
  return result;
}

void Runner::save_checkpoint(const std::filesystem::path& path) const {
  if (path.empty() || path.filename().empty()) {
    throw std::invalid_argument("checkpoint path must name a file");
  }
  if (impl_->pending_accumulation_steps != 0U) {
    throw std::runtime_error(
        "flush pending gradient accumulation before saving a checkpoint");
  }
  if (impl_->micro_batch_count >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      impl_->optimizer_step_count >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::runtime_error("training counters exceed checkpoint range");
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  auto partial_path = path;
  partial_path += ".partial";
  if (std::filesystem::exists(partial_path)) {
    throw std::runtime_error("checkpoint partial path already exists: " +
                             partial_path.string());
  }
  torch::serialize::OutputArchive archive;
  archive.write("format_version",
                torch::tensor(runner_checkpoint_format_version,
                              torch::TensorOptions().dtype(torch::kInt64)));
  archive.write("model_integer_signature",
                model_integer_signature(impl_->config.model));
  archive.write("model_floating_signature",
                model_floating_signature(impl_->config.model));
  archive.write("runner_integer_signature",
                runner_integer_signature(impl_->config));
  archive.write("runner_floating_signature",
                runner_floating_signature(impl_->config));
  archive.write(
      "training_counters",
      torch::tensor(
          {static_cast<std::int64_t>(impl_->micro_batch_count),
           static_cast<std::int64_t>(impl_->optimizer_step_count)},
          torch::TensorOptions().dtype(torch::kInt64)));
  archive.write(
      "current_learning_rate",
      torch::tensor(impl_->current_learning_rate,
                    torch::TensorOptions().dtype(torch::kFloat64)));
  archive.write(
      "selected_device_type",
      torch::tensor(static_cast<std::int64_t>(impl_->selected_device.type()),
                    torch::TensorOptions().dtype(torch::kInt64)));
  archive.write("cpu_rng_state",
                at::globalContext()
                    .defaultGenerator(torch::Device(torch::kCPU))
                    .get_state());
  if (impl_->selected_device.is_cuda()) {
    archive.write("device_rng_state",
                  at::globalContext()
                      .defaultGenerator(impl_->selected_device)
                      .get_state());
  }
  archive.write(
      "has_qwen_weights",
      torch::tensor(static_cast<std::int64_t>(impl_->qwen_weights.has_value()),
                    torch::TensorOptions().dtype(torch::kInt64)));
  if (impl_->qwen_weights.has_value()) {
    const auto& qwen = *impl_->qwen_weights;
    archive.write("qwen_model_id", c10::IValue(qwen.model_id));
    archive.write("qwen_revision", c10::IValue(qwen.revision));
    archive.write("qwen_archive_sha256", c10::IValue(qwen.archive_sha256));
    archive.write("qwen_source_checkpoint_sha256",
                  c10::IValue(qwen.source_checkpoint_sha256));
    archive.write("qwen_config_sha256", c10::IValue(qwen.config_sha256));
    archive.write("qwen_tokenizer_fingerprint_sha256",
                  c10::IValue(qwen.tokenizer_fingerprint_sha256));
    archive.write("qwen_metadata_sha256", c10::IValue(qwen.metadata_sha256));
    archive.write("qwen_payload_sha256", c10::IValue(qwen.payload_sha256));
    archive.write(
        "qwen_load_counts",
        torch::tensor(
            {static_cast<std::int64_t>(qwen.loaded_tensor_count),
             static_cast<std::int64_t>(qwen.loaded_payload_bytes),
             static_cast<std::int64_t>(
                 qwen.untouched_runtime_parameter_count)},
            torch::TensorOptions().dtype(torch::kInt64)));
  }
  torch::serialize::OutputArchive model_archive;
  impl_->model->save(model_archive);
  archive.write("model", model_archive);
  torch::serialize::OutputArchive optimizer_archive;
  impl_->optimizer.save(optimizer_archive);
  archive.write("optimizer", optimizer_archive);
  archive.save_to(partial_path.string());
  std::error_code rename_error;
  std::filesystem::rename(partial_path, path, rename_error);
  if (rename_error) {
    throw std::runtime_error("could not finalize chatbot checkpoint: " +
                             rename_error.message());
  }
}

void Runner::load_checkpoint(const std::filesystem::path& path) {
  if (impl_->pending_accumulation_steps != 0U) {
    throw std::runtime_error(
        "cannot replace a runner with pending gradient accumulation");
  }
  torch::serialize::InputArchive archive;
  archive.load_from(path.string(), impl_->selected_device);
  torch::Tensor format_version;
  if (!archive.try_read("format_version", format_version) ||
      format_version.item<std::int64_t>() !=
          runner_checkpoint_format_version) {
    throw std::runtime_error("unsupported chatbot runner checkpoint format");
  }
  require_signature(archive, "model_integer_signature",
                    model_integer_signature(impl_->config.model));
  require_signature(archive, "model_floating_signature",
                    model_floating_signature(impl_->config.model));
  require_signature(archive, "runner_integer_signature",
                    runner_integer_signature(impl_->config));
  require_signature(archive, "runner_floating_signature",
                    runner_floating_signature(impl_->config));

  torch::Tensor restored_learning_rate_tensor;
  if (!archive.try_read("current_learning_rate",
                        restored_learning_rate_tensor) ||
      restored_learning_rate_tensor.scalar_type() != torch::kFloat64 ||
      restored_learning_rate_tensor.numel() != 1) {
    throw std::runtime_error(
        "chatbot checkpoint has invalid current learning rate");
  }
  const auto restored_learning_rate =
      restored_learning_rate_tensor.to(torch::kCPU).item<double>();
  if (!std::isfinite(restored_learning_rate) ||
      restored_learning_rate < 0.0 ||
      restored_learning_rate > impl_->config.learning_rate) {
    throw std::runtime_error(
        "chatbot checkpoint has invalid current learning rate");
  }

  torch::Tensor source_device_type;
  if (!archive.try_read("selected_device_type", source_device_type) ||
      source_device_type.scalar_type() != torch::kInt64 ||
      source_device_type.numel() != 1) {
    throw std::runtime_error(
        "chatbot checkpoint has invalid selected device type");
  }
  const auto saved_device_type =
      source_device_type.to(torch::kCPU).item<std::int64_t>();
  const auto current_device_type =
      static_cast<std::int64_t>(impl_->selected_device.type());
  if (saved_device_type != current_device_type) {
    throw std::runtime_error(
        "chatbot checkpoint device type does not match the current runner");
  }

  torch::serialize::InputArchive model_archive;
  torch::serialize::InputArchive optimizer_archive;
  if (!archive.try_read("model", model_archive) ||
      !archive.try_read("optimizer", optimizer_archive)) {
    throw std::runtime_error(
        "chatbot checkpoint requires model and optimizer state");
  }
  impl_->model->load(model_archive);
  impl_->optimizer.load(optimizer_archive);

  torch::Tensor counters;
  if (!archive.try_read("training_counters", counters) ||
      counters.scalar_type() != torch::kInt64 || counters.numel() != 2) {
    throw std::runtime_error(
        "chatbot checkpoint has invalid training counters");
  }
  counters = counters.to(torch::kCPU).reshape({-1});
  const auto micro_batches = counters[0].item<std::int64_t>();
  const auto optimizer_steps = counters[1].item<std::int64_t>();
  if (micro_batches < 0 || optimizer_steps < 0) {
    throw std::runtime_error(
        "chatbot checkpoint has negative training counters");
  }
  const auto expected_learning_rate = learning_rate_for_step(
      impl_->config,
      optimizer_steps == 0 ? 1U : static_cast<std::uint64_t>(optimizer_steps));
  if (restored_learning_rate != expected_learning_rate) {
    throw std::runtime_error(
        "chatbot checkpoint current learning rate is inconsistent with its "
        "training counters");
  }
  // LibTorch 2.3 does not restore optimizer-group learning-rate options while
  // newer versions do. The validated explicit scalar is authoritative and is
  // applied uniformly so resume metadata and the next update are portable.
  impl_->set_learning_rate(restored_learning_rate);
  impl_->micro_batch_count = static_cast<std::uint64_t>(micro_batches);
  impl_->optimizer_step_count = static_cast<std::uint64_t>(optimizer_steps);
  impl_->pending_accumulation_steps = 0U;

  torch::Tensor has_qwen_weights;
  if (!archive.try_read("has_qwen_weights", has_qwen_weights) ||
      has_qwen_weights.numel() != 1) {
    throw std::runtime_error(
        "chatbot checkpoint is missing Qwen provenance state");
  }
  const auto has_qwen = has_qwen_weights.item<std::int64_t>();
  if (has_qwen != 0 && has_qwen != 1) {
    throw std::runtime_error("chatbot checkpoint has invalid Qwen provenance");
  }
  impl_->qwen_weights.reset();
  if (has_qwen == 1) {
    torch::Tensor counts;
    if (!archive.try_read("qwen_load_counts", counts) ||
        counts.scalar_type() != torch::kInt64 || counts.numel() != 3) {
      throw std::runtime_error(
          "chatbot checkpoint has invalid Qwen load counters");
    }
    counts = counts.to(torch::kCPU).reshape({-1});
    const auto tensors = counts[0].item<std::int64_t>();
    const auto bytes = counts[1].item<std::int64_t>();
    const auto runtime_parameters = counts[2].item<std::int64_t>();
    if (tensors <= 0 || bytes <= 0 || runtime_parameters < 0) {
      throw std::runtime_error(
          "chatbot checkpoint has invalid Qwen load counters");
    }
    impl_->qwen_weights = QwenDenseLoadResult{
        .model_id = read_checkpoint_string(archive, "qwen_model_id"),
        .revision = read_checkpoint_string(archive, "qwen_revision"),
        .archive_sha256 =
            read_checkpoint_string(archive, "qwen_archive_sha256"),
        .source_checkpoint_sha256 = read_checkpoint_string(
            archive, "qwen_source_checkpoint_sha256"),
        .config_sha256 =
            read_checkpoint_string(archive, "qwen_config_sha256"),
        .tokenizer_fingerprint_sha256 = read_checkpoint_string(
            archive, "qwen_tokenizer_fingerprint_sha256"),
        .metadata_sha256 =
            read_checkpoint_string(archive, "qwen_metadata_sha256"),
        .payload_sha256 =
            read_checkpoint_string(archive, "qwen_payload_sha256"),
        .loaded_tensor_count = static_cast<std::size_t>(tensors),
        .loaded_payload_bytes = static_cast<std::uint64_t>(bytes),
        .untouched_runtime_parameter_count =
            static_cast<std::size_t>(runtime_parameters)};
    const auto& qwen = *impl_->qwen_weights;
    if (qwen.model_id.empty() || !is_git_revision(qwen.revision) ||
        !is_sha256(qwen.archive_sha256) ||
        !is_sha256(qwen.source_checkpoint_sha256) ||
        !is_sha256(qwen.config_sha256) ||
        !is_sha256(qwen.tokenizer_fingerprint_sha256) ||
        !is_sha256(qwen.metadata_sha256) ||
        !is_sha256(qwen.payload_sha256)) {
      throw std::runtime_error(
          "chatbot checkpoint has malformed Qwen provenance");
    }
  }

  torch::Tensor cpu_rng_state;
  if (!archive.try_read("cpu_rng_state", cpu_rng_state)) {
    throw std::runtime_error("chatbot checkpoint is missing CPU RNG state");
  }
  set_default_rng_state(torch::Device(torch::kCPU),
                        cpu_rng_state.to(torch::kCPU));
  if (impl_->selected_device.is_cuda()) {
    torch::Tensor device_rng_state;
    if (!archive.try_read("device_rng_state", device_rng_state)) {
      throw std::runtime_error(
          "CUDA resume requires a checkpoint with CUDA RNG state");
    }
    set_default_rng_state(impl_->selected_device,
                          device_rng_state.to(impl_->selected_device));
  }
  impl_->model->to(impl_->selected_device);
  impl_->model->reset_state();
}

const RunnerConfig& Runner::config() const noexcept { return impl_->config; }

std::string Runner::device() const { return impl_->selected_device.str(); }

std::size_t Runner::parameter_count() const noexcept {
  return impl_->model->parameter_count();
}

TrainingState Runner::training_state() const noexcept {
  return impl_->state();
}

const std::optional<QwenDenseLoadResult>& Runner::qwen_weights()
    const noexcept {
  return impl_->qwen_weights;
}

void require_qwen3_phase0_checkpoint_provenance(
    const QwenDenseLoadResult& provenance) {
  const auto mismatch = [](const std::string_view actual,
                           const std::string_view expected,
                           const std::string_view field) {
    if (actual != expected) {
      throw std::runtime_error(
          "restored Qwen checkpoint does not match the supported Phase 0 "
          "oracle: " +
          std::string(field));
    }
  };
  mismatch(provenance.model_id, qwen3_phase0_model_id, "model_id");
  mismatch(provenance.revision, qwen3_phase0_revision, "revision");
  mismatch(provenance.source_checkpoint_sha256,
           qwen3_phase0_source_checkpoint_sha256,
           "source_checkpoint_sha256");
  mismatch(provenance.config_sha256, qwen3_phase0_config_sha256,
           "config_sha256");
  mismatch(provenance.tokenizer_fingerprint_sha256,
           qwen3_phase0_tokenizer_fingerprint_sha256,
           "tokenizer_fingerprint_sha256");
  if (provenance.loaded_tensor_count != qwen3_phase0_dense_tensor_count) {
    throw std::runtime_error(
        "restored Qwen checkpoint does not match the supported Phase 0 "
        "oracle: loaded_tensor_count");
  }
  if (provenance.loaded_payload_bytes !=
      qwen3_phase0_dense_payload_bytes) {
    throw std::runtime_error(
        "restored Qwen checkpoint does not match the supported Phase 0 "
        "oracle: loaded_payload_bytes");
  }
}

ProtocolServeResult serve_token_protocol(
    std::istream& input, std::ostream& output, Runner& runner,
    const ProtocolLimits& limits,
    const std::filesystem::path& checkpoint_on_flush) {
  if (limits.maximum_line_bytes == 0U ||
      limits.maximum_request_id_bytes == 0U || limits.maximum_tokens == 0U ||
      limits.maximum_eos_tokens == 0U) {
    throw std::invalid_argument("token protocol limits must be positive");
  }
  bool checkpoint_current = false;
  std::string line;
  while (true) {
    const auto status =
        read_bounded_line(input, line, limits.maximum_line_bytes);
    if (status == LineStatus::end) {
      break;
    }
    if (status == LineStatus::too_long) {
      write_error(output, "", "request_too_large",
                  "request exceeds maximum_line_bytes");
      continue;
    }
    std::string request_id;
    try {
      const auto request = ProtocolParser(line, limits).parse();
      request_id = request.request_id;
      ProtocolParser::validate_operation_contract(request);
      if (request.operation == "shutdown") {
        write_prefix(output, request_id, true);
        output << ",\"shutdown\":true}\n";
        output.flush();
        break;
      }
      if (request.operation == "reset") {
        runner.reset_state();
        write_prefix(output, request_id, true);
        output << ",\"reset\":true}\n";
        output.flush();
        continue;
      }
      if (request.operation == "flush") {
        const auto state = runner.flush_optimizer();
        if (!checkpoint_on_flush.empty()) {
          runner.save_checkpoint(checkpoint_on_flush);
          checkpoint_current = true;
        }
        write_prefix(output, request_id, true);
        output << ",\"training_state\":";
        write_training_state(output, state);
        output << "}\n";
        output.flush();
        continue;
      }
      if (request.operation == "metadata") {
        const auto& model = runner.config().model;
        const auto state = runner.training_state();
        write_prefix(output, request_id, true);
        output << ",\"metadata\":{\"device\":\""
               << json_escape(runner.device()) << "\",\"parameter_count\":"
               << runner.parameter_count() << ",\"vocabulary_size\":"
               << model.vocabulary_size << ",\"maximum_sequence_length\":"
               << model.maximum_sequence_length << ",\"model_dimension\":"
               << model.model_dimension << ",\"layer_count\":"
               << model.layer_count << ",\"simulation_steps\":"
               << model.simulation_steps << ",\"head_dimension\":"
               << snnbase::language::resolved_head_dimension(model)
               << ",\"query_key_normalization\":"
               << (model.query_key_normalization ? "true" : "false")
               << ",\"spiking\":"
               << (model.spiking ? "true" : "false")
               << ",\"build_provenance\":{\"experiments\":{\"revision\":\""
               << json_escape(
                      runner.config().build_provenance.experiments.revision)
               << "\",\"dirty\":"
               << (runner.config().build_provenance.experiments.dirty
                       ? "true"
                       : "false")
               << "},\"snnbase\":{\"revision\":\""
               << json_escape(runner.config().build_provenance.snnbase.revision)
               << "\",\"dirty\":"
               << (runner.config().build_provenance.snnbase.dirty ? "true"
                                                                  : "false")
               << "}}"
               << ",\"training_state\":";
        write_training_state(output, state);
        output << ",\"qwen_weights\":";
        if (runner.qwen_weights().has_value()) {
          const auto& qwen = *runner.qwen_weights();
          output << "{\"model_id\":\"" << json_escape(qwen.model_id)
                 << "\",\"revision\":\"" << json_escape(qwen.revision)
                 << "\",\"archive_sha256\":\""
                 << qwen.archive_sha256
                 << "\",\"source_checkpoint_sha256\":\""
                 << qwen.source_checkpoint_sha256
                 << "\",\"config_sha256\":\"" << qwen.config_sha256
                 << "\",\"tokenizer_fingerprint_sha256\":\""
                 << qwen.tokenizer_fingerprint_sha256
                 << "\",\"metadata_sha256\":\"" << qwen.metadata_sha256
                 << "\",\"payload_sha256\":\"" << qwen.payload_sha256
                 << "\",\"loaded_tensor_count\":"
                 << qwen.loaded_tensor_count
                 << ",\"loaded_payload_bytes\":"
                 << qwen.loaded_payload_bytes << '}';
        } else {
          output << "null";
        }
        output << "}}\n";
        output.flush();
        continue;
      }
      if (request.operation == "generate") {
        if (request.input_ids.empty()) {
          throw std::invalid_argument("generate requires nonempty input_ids");
        }
        const auto result = runner.generate(request.input_ids,
                                            request.generation);
        write_prefix(output, request_id, true);
        output << ",\"output_ids\":[";
        for (std::size_t index = 0; index < result.output_ids.size(); ++index) {
          if (index != 0U) {
            output << ',';
          }
          output << result.output_ids[index];
        }
        output << "],\"finish_reason\":\""
               << json_escape(result.finish_reason)
               << "\",\"metrics\":{\"generated_tokens\":"
               << result.metrics.generated_tokens << ",\"latency_ms\":"
               << std::setprecision(17) << result.metrics.elapsed_milliseconds
               << ",\"time_to_first_token_ms\":"
               << result.metrics.time_to_first_token_milliseconds
               << ",\"generated_tokens_per_second\":"
               << result.metrics.generated_tokens_per_second
               << ",\"mean_spike_rate\":"
               << result.metrics.mean_spike_rate << "}}\n";
        output.flush();
        continue;
      }
      if (request.operation == "inspect") {
        if (request.input_ids.empty()) {
          throw std::invalid_argument("inspect requires nonempty input_ids");
        }
        const auto inspection = runner.inspect_next_token_logits(
            request.input_ids, request.probe_token_ids,
            request.generation.top_k);
        const auto write_values = [&output](
                                      const std::vector<LogitValue>& values) {
          output << '[';
          for (std::size_t index = 0; index < values.size(); ++index) {
            if (index != 0U) {
              output << ',';
            }
            output << "{\"token_id\":" << values[index].token_id
                   << ",\"logit\":" << std::setprecision(17)
                   << values[index].logit << '}';
          }
          output << ']';
        };
        write_prefix(output, request_id, true);
        output << ",\"final_position\":{\"probes\":";
        write_values(inspection.probes);
        output << ",\"top_k\":";
        write_values(inspection.top_k);
        output << "}}\n";
        output.flush();
        continue;
      }
      if (request.operation == "train" || request.operation == "evaluate") {
        if (request.input_ids.empty()) {
          throw std::invalid_argument(request.operation +
                                      " requires nonempty input_ids");
        }
        if (request.fields.contains("loss_mask") &&
            request.loss_mask.size() != request.input_ids.size()) {
          throw std::invalid_argument(
              "loss_mask length does not match input_ids");
        }
        if (request.reset_state) {
          runner.reset_state();
        }
        LanguageModelMetrics metrics;
        if (request.operation == "train") {
          // Fail closed if a training call throws after partially mutating
          // gradients or parameters: the prior checkpoint is no longer
          // asserted to represent the in-memory runner.
          checkpoint_current = false;
          metrics = runner.train_step(request.input_ids, request.loss_mask);
        } else {
          metrics = runner.evaluate(request.input_ids, request.loss_mask);
        }
        write_prefix(output, request_id, true);
        output << ",\"metrics\":";
        write_metrics(output, metrics);
        output << "}\n";
        output.flush();
        continue;
      }
      write_error(output, request_id, "unsupported_operation",
                  "unsupported op: " + request.operation);
    } catch (const std::invalid_argument& error) {
      write_error(output, request_id, "invalid_request", error.what());
    } catch (const std::exception& error) {
      write_error(output, request_id, "model_error", error.what());
    }
  }
  return {.checkpoint_current = checkpoint_current};
}

}  // namespace snnbase_experiments::chatbot
