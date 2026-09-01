#include <snnbase_experiments/chatbot/runner.hpp>
#include <snnbase_experiments/project_config.hpp>

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using snnbase_experiments::chatbot::Runner;
using snnbase_experiments::chatbot::RunnerConfig;
using snnbase_experiments::chatbot::qwen3_phase0_config_sha256;
using snnbase_experiments::chatbot::qwen3_phase0_model_id;
using snnbase_experiments::chatbot::qwen3_phase0_revision;
using snnbase_experiments::chatbot::qwen3_phase0_source_checkpoint_sha256;
using snnbase_experiments::chatbot::
    qwen3_phase0_tokenizer_fingerprint_sha256;

[[noreturn]] void usage(const char* executable, std::string_view error = {}) {
  if (!error.empty()) {
    std::cerr << "error: " << error << "\n\n";
  }
  std::cerr
      << "Usage: " << executable << " serve [options]\n"
      << "       " << executable << " self-test\n\n"
      << "serve reads snnbase.chatbot.tokens/v1 JSON objects from stdin and "
         "writes one JSON response per line.\n\n"
      << "Options:\n"
      << "  --checkpoint PATH          Load an existing model checkpoint\n"
      << "  --save-checkpoint PATH     Save model state after shutdown/EOF\n"
      << "  --qwen3-0.6b               Use the exact pinned Qwen3 architecture\n"
      << "  --qwen-archive PATH        Load a converted Qwen dense archive\n"
      << "  --qwen-archive-sha256 HEX  Expected converted-archive SHA-256\n"
      << "  --device auto|cpu|cuda     Execution device (default: auto)\n"
      << "  --vocabulary-size N        Output vocabulary (default: 151936)\n"
      << "  --context-tokens N         Maximum prompt plus completion (128)\n"
      << "  --model-dimension N        Hidden width (256)\n"
      << "  --layers N                 Decoder blocks (4)\n"
      << "  --query-heads N            Query heads (8)\n"
      << "  --kv-heads N               Key/value heads (2)\n"
      << "  --head-dimension N         Explicit attention head width\n"
      << "  --qk-norm                  Enable Qwen3 per-head Q/K RMSNorm\n"
      << "  --no-qk-norm               Disable Q/K RMSNorm (default)\n"
      << "  --ffn-dimension N          SwiGLU intermediate width (768)\n"
      << "  --rms-epsilon F            RMSNorm epsilon (1e-6 for Qwen3)\n"
      << "  --rope-base F              RoPE frequency base (1e6 for Qwen3)\n"
      << "  --simulation-steps N       LIF microsteps per activation (4)\n"
      << "  --ann                      Bypass LIF sites for the ANN control\n"
      << "  --snn                      Enable LIF sites (default)\n"
      << "  --threshold F              Initial LIF threshold (1.0)\n"
      << "  --leak F                   Initial LIF leak (0.9)\n"
      << "  --surrogate-slope F        Surrogate sigmoid slope (4.0)\n"
      << "  --learning-rate F          AdamW learning rate (3e-4)\n"
      << "  --weight-decay F           AdamW weight decay (1e-2)\n"
      << "  --gradient-clip F          Gradient norm limit (1.0)\n"
      << "  --gradient-accumulation N  Micro-batches per update (1)\n"
      << "  --total-optimizer-steps N  Cosine schedule length (0=constant)\n"
      << "  --warmup-optimizer-steps N Linear warmup update count (0)\n"
      << "  --minimum-lr-ratio F       Final/base learning-rate ratio (0)\n"
      << "  --spike-rate-target F      Regularizer target in [0,1] (0.15)\n"
      << "  --spike-rate-penalty F     Nonnegative regularizer weight (0)\n"
      << "  --seed N                   Model initialization seed (42)\n";
  throw std::invalid_argument(error.empty() ? "help requested"
                                             : std::string(error));
}

std::uint64_t parse_unsigned(std::string_view value,
                             std::string_view option) {
  std::uint64_t result = 0;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (value.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != value.data() + value.size()) {
    throw std::invalid_argument(std::string(option) +
                                " requires a nonnegative integer");
  }
  return result;
}

std::int64_t parse_positive_i64(std::string_view value,
                                std::string_view option) {
  const auto parsed = parse_unsigned(value, option);
  if (parsed == 0U ||
      parsed > static_cast<std::uint64_t>(
                   std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(std::string(option) +
                                " requires a positive int64 value");
  }
  return static_cast<std::int64_t>(parsed);
}

double parse_finite(std::string_view value, std::string_view option) {
  std::size_t consumed = 0;
  double result = 0.0;
  try {
    result = std::stod(std::string(value), &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(option) +
                                " requires a finite number");
  }
  if (consumed != value.size() || !std::isfinite(result)) {
    throw std::invalid_argument(std::string(option) +
                                " requires a finite number");
  }
  return result;
}

RunnerConfig default_config() {
  RunnerConfig config;
  config.model = {
      .vocabulary_size = 151936,
      .maximum_sequence_length = 128,
      .model_dimension = 256,
      .layer_count = 4,
      .query_head_count = 8,
      .key_value_head_count = 2,
      .feed_forward_dimension = 768,
      .simulation_steps = 4,
      .rms_epsilon = 1.0e-6,
      .rope_base = 1000000.0,
      .seed = 42,
      .lif = {.initial_threshold = 1.0F,
              .initial_leak = 0.9F,
              .surrogate_slope = 4.0F,
              .learn_threshold = true,
              .learn_leak = true,
              .signed_spikes = true},
  };
  config.build_provenance = {
      .experiments =
          {.revision = std::string(
               snnbase_experiments::config::experiments_git_revision),
           .dirty = snnbase_experiments::config::experiments_git_dirty},
      .snnbase =
          {.revision =
               std::string(snnbase_experiments::config::snnbase_git_revision),
           .dirty = snnbase_experiments::config::snnbase_git_dirty},
  };
  return config;
}

struct Options {
  std::string command;
  RunnerConfig runner{default_config()};
  std::filesystem::path checkpoint;
  std::filesystem::path save_checkpoint;
  std::filesystem::path qwen_archive;
  std::string qwen_archive_sha256;
  bool qwen3_06b{};
  bool model_shape_override{};
  bool context_override{};
};

void apply_qwen3_06b_config(Options& options) {
  const auto selected_context = options.context_override
                                    ? options.runner.model.maximum_sequence_length
                                    : 512;
  options.runner.model.vocabulary_size = 151936;
  options.runner.model.maximum_sequence_length = selected_context;
  options.runner.model.model_dimension = 1024;
  options.runner.model.layer_count = 28;
  options.runner.model.query_head_count = 16;
  options.runner.model.key_value_head_count = 8;
  options.runner.model.head_dimension = 128;
  options.runner.model.query_key_normalization = true;
  options.runner.model.feed_forward_dimension = 3072;
  options.runner.model.rms_epsilon = 1.0e-6;
  options.runner.model.rope_base = 1'000'000.0;
}

Options parse_options(const int argc, char** argv) {
  if (argc < 2) {
    usage(argv[0], "a command is required");
  }
  Options options;
  options.command = argv[1];
  if (options.command != "serve" && options.command != "self-test") {
    usage(argv[0], "unknown command: " + options.command);
  }
  if (options.command == "self-test" && argc != 2) {
    usage(argv[0], "self-test does not accept options");
  }
  for (int index = 2; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--help") {
      usage(argv[0]);
    }
    if (option == "--ann") {
      options.runner.model.spiking = false;
      continue;
    }
    if (option == "--snn") {
      options.runner.model.spiking = true;
      continue;
    }
    if (option == "--qk-norm") {
      options.runner.model.query_key_normalization = true;
      options.model_shape_override = true;
      continue;
    }
    if (option == "--no-qk-norm") {
      options.runner.model.query_key_normalization = false;
      options.model_shape_override = true;
      continue;
    }
    if (option == "--qwen3-0.6b") {
      options.qwen3_06b = true;
      continue;
    }
    if (index + 1 >= argc) {
      usage(argv[0], std::string(option) + " requires a value");
    }
    const std::string_view value = argv[++index];
    if (option == "--checkpoint") {
      options.checkpoint = value;
    } else if (option == "--save-checkpoint") {
      options.save_checkpoint = value;
    } else if (option == "--qwen-archive") {
      options.qwen_archive = value;
    } else if (option == "--qwen-archive-sha256") {
      options.qwen_archive_sha256 = value;
    } else if (option == "--device") {
      options.runner.device = value;
    } else if (option == "--vocabulary-size") {
      options.runner.model.vocabulary_size = parse_positive_i64(value, option);
      options.model_shape_override = true;
    } else if (option == "--context-tokens") {
      options.runner.model.maximum_sequence_length =
          parse_positive_i64(value, option);
      options.context_override = true;
    } else if (option == "--model-dimension") {
      options.runner.model.model_dimension = parse_positive_i64(value, option);
      options.model_shape_override = true;
    } else if (option == "--layers") {
      options.runner.model.layer_count = parse_positive_i64(value, option);
      options.model_shape_override = true;
    } else if (option == "--query-heads") {
      options.runner.model.query_head_count = parse_positive_i64(value, option);
      options.model_shape_override = true;
    } else if (option == "--kv-heads") {
      options.runner.model.key_value_head_count =
          parse_positive_i64(value, option);
      options.model_shape_override = true;
    } else if (option == "--head-dimension") {
      options.runner.model.head_dimension = parse_positive_i64(value, option);
      options.model_shape_override = true;
    } else if (option == "--ffn-dimension") {
      options.runner.model.feed_forward_dimension =
          parse_positive_i64(value, option);
      options.model_shape_override = true;
    } else if (option == "--rms-epsilon") {
      options.runner.model.rms_epsilon = parse_finite(value, option);
      options.model_shape_override = true;
    } else if (option == "--rope-base") {
      options.runner.model.rope_base = parse_finite(value, option);
      options.model_shape_override = true;
    } else if (option == "--simulation-steps") {
      options.runner.model.simulation_steps =
          parse_positive_i64(value, option);
    } else if (option == "--threshold") {
      options.runner.model.lif.initial_threshold =
          static_cast<float>(parse_finite(value, option));
    } else if (option == "--leak") {
      options.runner.model.lif.initial_leak =
          static_cast<float>(parse_finite(value, option));
    } else if (option == "--surrogate-slope") {
      options.runner.model.lif.surrogate_slope =
          static_cast<float>(parse_finite(value, option));
    } else if (option == "--learning-rate") {
      options.runner.learning_rate = parse_finite(value, option);
    } else if (option == "--weight-decay") {
      options.runner.weight_decay = parse_finite(value, option);
    } else if (option == "--gradient-clip") {
      options.runner.gradient_clip_norm = parse_finite(value, option);
    } else if (option == "--gradient-accumulation") {
      options.runner.gradient_accumulation_steps =
          static_cast<std::size_t>(parse_positive_i64(value, option));
    } else if (option == "--total-optimizer-steps") {
      options.runner.total_optimizer_steps = parse_unsigned(value, option);
    } else if (option == "--warmup-optimizer-steps") {
      options.runner.warmup_optimizer_steps = parse_unsigned(value, option);
    } else if (option == "--minimum-lr-ratio") {
      options.runner.minimum_learning_rate_ratio = parse_finite(value, option);
    } else if (option == "--spike-rate-target") {
      options.runner.spike_rate_target = parse_finite(value, option);
    } else if (option == "--spike-rate-penalty") {
      options.runner.spike_rate_penalty = parse_finite(value, option);
    } else if (option == "--seed") {
      options.runner.model.seed = parse_unsigned(value, option);
    } else {
      usage(argv[0], "unknown option: " + std::string(option));
    }
  }
  if (options.qwen3_06b) {
    if (options.model_shape_override) {
      usage(argv[0],
            "--qwen3-0.6b cannot be combined with architecture-shape overrides");
    }
    apply_qwen3_06b_config(options);
  }
  if (options.qwen_archive.empty() != options.qwen_archive_sha256.empty()) {
    usage(argv[0],
          "--qwen-archive and --qwen-archive-sha256 must be supplied together");
  }
  if (!options.qwen_archive.empty() && !options.qwen3_06b) {
    usage(argv[0], "--qwen-archive requires --qwen3-0.6b");
  }
  if (!options.qwen_archive.empty() && !options.checkpoint.empty()) {
    usage(argv[0], "Qwen import and runner checkpoint load are mutually exclusive");
  }
  return options;
}

void run_self_test() {
  auto config = default_config();
  config.device = "cpu";
  config.model.vocabulary_size = 17;
  config.model.maximum_sequence_length = 16;
  config.model.model_dimension = 16;
  config.model.layer_count = 1;
  config.model.query_head_count = 4;
  config.model.key_value_head_count = 2;
  config.model.feed_forward_dimension = 32;
  config.model.simulation_steps = 2;
  config.learning_rate = 2.0e-3;
  config.weight_decay = 0.0;
  Runner runner(config);
  const std::vector<std::int64_t> tokens{1, 2, 3, 4, 5};
  const std::vector<std::uint8_t> mask{0, 0, 1, 1, 1};
  const auto before = runner.evaluate(tokens, mask);
  for (std::size_t step = 0; step < 10U; ++step) {
    static_cast<void>(runner.train_step(tokens, mask));
  }
  const auto after = runner.evaluate(tokens, mask);
  if (!(after.loss < before.loss)) {
    throw std::runtime_error("self-test training did not reduce loss");
  }
  std::cout << "{\"ok\":true,\"loss_before\":" << before.loss
            << ",\"loss_after\":" << after.loss
            << ",\"parameter_count\":" << runner.parameter_count()
            << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    if (options.command == "self-test") {
      run_self_test();
      return 0;
    }
    Runner runner(options.runner);
    if (!options.qwen_archive.empty()) {
      const auto result = runner.load_qwen_weights(
          options.qwen_archive,
          {.model_id = std::string(qwen3_phase0_model_id),
           .revision = std::string(qwen3_phase0_revision),
           .archive_sha256 = options.qwen_archive_sha256,
           .source_checkpoint_sha256 =
               std::string(qwen3_phase0_source_checkpoint_sha256),
           .config_sha256 = std::string(qwen3_phase0_config_sha256),
           .tokenizer_fingerprint_sha256 =
               std::string(qwen3_phase0_tokenizer_fingerprint_sha256)});
      std::cerr << "loaded " << result.loaded_tensor_count
                << " verified Qwen dense tensors from "
                << options.qwen_archive << '\n';
    }
    if (!options.checkpoint.empty()) {
      runner.load_checkpoint(options.checkpoint);
      if (runner.qwen_weights().has_value()) {
        snnbase_experiments::chatbot::
            require_qwen3_phase0_checkpoint_provenance(
                *runner.qwen_weights());
      }
    }
    const auto serve_result =
        snnbase_experiments::chatbot::serve_token_protocol(
            std::cin, std::cout, runner, {}, options.save_checkpoint);
    if (!options.save_checkpoint.empty() &&
        !serve_result.checkpoint_current) {
      static_cast<void>(runner.flush_optimizer());
      runner.save_checkpoint(options.save_checkpoint);
    }
    return 0;
  } catch (const std::invalid_argument& error) {
    if (std::string_view(error.what()) == "help requested") {
      return 0;
    }
    std::cerr << "chatbot_experiment: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "chatbot_experiment: " << error.what() << '\n';
    return 1;
  }
}
