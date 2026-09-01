#include <snnbase_experiments/chatbot/runner.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using snnbase_experiments::chatbot::GenerationConfig;
using snnbase_experiments::chatbot::Runner;
using snnbase_experiments::chatbot::RunnerConfig;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::filesystem::path temporary_checkpoint(const std::string& stem) {
  static std::uint64_t counter = 0;
  const auto nonce = static_cast<std::uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  return std::filesystem::temp_directory_path() /
         (stem + "-" + std::to_string(nonce) + "-" +
          std::to_string(counter++) + ".pt");
}

void require_protocol_error(const std::string& responses,
                            const std::string& request_id,
                            const std::string& message) {
  const auto marker = "\"request_id\":\"" + request_id + "\"";
  const auto begin = responses.find(marker);
  require(begin != std::string::npos,
          "protocol response is missing request_id " + request_id);
  const auto end = responses.find('\n', begin);
  const auto response = responses.substr(begin, end - begin);
  require(response.find("\"ok\":false") != std::string::npos &&
              response.find("\"code\":\"invalid_request\"") !=
                  std::string::npos &&
              response.find(message) != std::string::npos,
          "protocol request " + request_id +
              " did not return its expected contract error");
}

RunnerConfig tiny_config() {
  RunnerConfig config;
  config.model = {
      .vocabulary_size = 23,
      .maximum_sequence_length = 20,
      .model_dimension = 16,
      .layer_count = 2,
      .query_head_count = 4,
      .key_value_head_count = 2,
      .feed_forward_dimension = 32,
      .simulation_steps = 3,
      .rms_epsilon = 1.0e-5,
      .rope_base = 10000.0,
      .seed = 13,
      .lif = {.initial_threshold = 0.35F,
              .initial_leak = 0.7F,
              .surrogate_slope = 4.0F,
              .learn_threshold = true,
              .learn_leak = true,
              .signed_spikes = true},
  };
  config.device = "cpu";
  config.learning_rate = 2.0e-3;
  config.weight_decay = 0.0;
  config.gradient_clip_norm = 1.0;
  return config;
}

void test_masked_training_and_evaluation() {
  Runner runner(tiny_config());
  const std::vector<std::int64_t> tokens{1, 2, 3, 4, 5, 6};
  const std::vector<std::uint8_t> mask{0, 0, 0, 1, 1, 1};
  const auto before = runner.evaluate(tokens, mask);
  require(before.token_count == 3U,
          "assistant loss mask selected the wrong number of targets");
  require(std::isfinite(before.loss) && before.loss > 0.0,
          "initial language-model loss is invalid");
  require(std::isfinite(before.perplexity) && before.perplexity >= 1.0,
          "initial perplexity is invalid");
  require(before.mean_spike_rate >= 0.0 && before.mean_spike_rate <= 1.0,
          "spike rate is outside [0,1]");
  require(before.correct_token_count <= before.token_count &&
              before.token_accuracy >= 0.0 && before.token_accuracy <= 1.0,
          "token-accuracy metrics are inconsistent");
  require(before.objective_loss >= before.loss,
          "regularized objective is smaller than token NLL");

  for (std::size_t step = 0; step < 30U; ++step) {
    const auto metrics = runner.train_step(tokens, mask);
    require(metrics.token_count == 3U && std::isfinite(metrics.loss),
            "training step returned invalid metrics");
  }
  const auto after = runner.evaluate(tokens, mask);
  require(after.loss < before.loss,
          "tiny repeated training did not reduce masked loss");

  bool rejected_empty_mask = false;
  try {
    static_cast<void>(runner.evaluate(
        tokens, std::vector<std::uint8_t>(tokens.size(), 0U)));
  } catch (const std::invalid_argument&) {
    rejected_empty_mask = true;
  }
  require(rejected_empty_mask, "an all-zero loss mask was accepted");
}

void test_nonfinite_runner_configuration_is_rejected() {
  const auto rejected = [](RunnerConfig config) {
    try {
      Runner runner(std::move(config));
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  auto learning_rate = tiny_config();
  learning_rate.learning_rate = std::numeric_limits<double>::infinity();
  require(rejected(learning_rate), "infinite learning rate was accepted");
  auto weight_decay = tiny_config();
  weight_decay.weight_decay = std::numeric_limits<double>::infinity();
  require(rejected(weight_decay), "infinite weight decay was accepted");
  auto gradient_clip = tiny_config();
  gradient_clip.gradient_clip_norm =
      std::numeric_limits<double>::infinity();
  require(rejected(gradient_clip), "infinite gradient clip was accepted");
}

void test_gradient_accumulation_flush_and_schedule() {
  auto config = tiny_config();
  config.gradient_accumulation_steps = 3;
  config.total_optimizer_steps = 4;
  config.warmup_optimizer_steps = 1;
  config.minimum_learning_rate_ratio = 0.1;
  config.spike_rate_penalty = 1.0e-3;
  Runner runner(config);
  const std::vector<std::int64_t> tokens{1, 2, 3, 4, 5};
  const auto first = runner.train_step(tokens);
  require(!first.optimizer_updated && first.optimizer_step_count == 0U &&
              runner.training_state().pending_accumulation_steps == 1U,
          "first accumulated micro-batch unexpectedly updated parameters");
  static_cast<void>(runner.train_step(tokens));
  const auto third = runner.train_step(tokens);
  require(third.optimizer_updated && third.optimizer_step_count == 1U &&
              runner.training_state().pending_accumulation_steps == 0U,
          "configured accumulation boundary did not update parameters");
  static_cast<void>(runner.train_step(tokens));
  const auto flushed = runner.flush_optimizer();
  require(flushed.optimizer_step_count == 2U &&
              flushed.pending_accumulation_steps == 0U &&
              flushed.learning_rate > 0.0 &&
              flushed.learning_rate <= config.learning_rate,
          "partial gradient accumulation was not flushed correctly");
}

void test_cosine_schedule_endpoints() {
  const std::vector<std::int64_t> tokens{1, 2, 3, 4, 5};

  auto single_step_config = tiny_config();
  single_step_config.total_optimizer_steps = 1;
  single_step_config.minimum_learning_rate_ratio = 0.0;
  Runner single_step(single_step_config);
  const auto single = single_step.train_step(tokens);
  require(single.optimizer_updated &&
              std::abs(single.learning_rate - single_step_config.learning_rate) <
                  1.0e-15,
          "a one-update cosine schedule did not use the configured learning rate");

  auto decay_config = tiny_config();
  decay_config.total_optimizer_steps = 4;
  decay_config.minimum_learning_rate_ratio = 0.25;
  Runner decay(decay_config);
  const auto first = decay.train_step(tokens);
  require(std::abs(first.learning_rate - decay_config.learning_rate) < 1.0e-15,
          "cosine decay did not begin at the configured learning rate");
  static_cast<void>(decay.train_step(tokens));
  static_cast<void>(decay.train_step(tokens));
  const auto final = decay.train_step(tokens);
  require(std::abs(final.learning_rate -
                   decay_config.learning_rate *
                       decay_config.minimum_learning_rate_ratio) < 1.0e-15,
          "cosine decay did not finish at the configured minimum rate");

  Runner scheduled_source(decay_config);
  static_cast<void>(scheduled_source.train_step(tokens));
  static_cast<void>(scheduled_source.train_step(tokens));
  const auto source_state = scheduled_source.training_state();
  const auto checkpoint =
      temporary_checkpoint("snnbase-chatbot-scheduled-lr");
  std::filesystem::remove(checkpoint);
  scheduled_source.save_checkpoint(checkpoint);
  Runner scheduled_restored(decay_config);
  scheduled_restored.load_checkpoint(checkpoint);
  const auto restored_state = scheduled_restored.training_state();
  require(restored_state.optimizer_step_count == source_state.optimizer_step_count &&
              std::abs(restored_state.learning_rate - source_state.learning_rate) <
                  1.0e-15,
          "scheduled checkpoint did not restore the reported current learning rate");
  const auto source_next = scheduled_source.train_step(tokens);
  const auto restored_next = scheduled_restored.train_step(tokens);
  require(source_next.optimizer_step_count == restored_next.optimizer_step_count &&
              std::abs(source_next.learning_rate - restored_next.learning_rate) <
                  1.0e-15,
          "scheduled checkpoint changed the next optimizer learning rate");
  std::filesystem::remove(checkpoint);
}

void test_bounded_logit_inspection() {
  Runner runner(tiny_config());
  const std::vector<std::int64_t> tokens{1, 2, 3, 4};
  const std::vector<std::int64_t> probes{0, 7, 22};
  const auto inspected =
      runner.inspect_next_token_logits(tokens, probes, 5U);
  require(inspected.probes.size() == probes.size() &&
              inspected.top_k.size() == 5U,
          "logit inspection returned the wrong bounds");
  for (std::size_t index = 0; index < probes.size(); ++index) {
    require(inspected.probes[index].token_id == probes[index] &&
                std::isfinite(inspected.probes[index].logit),
            "logit probe result is invalid");
  }
  for (std::size_t index = 1; index < inspected.top_k.size(); ++index) {
    require(inspected.top_k[index - 1U].logit >= inspected.top_k[index].logit,
            "top-k logits are not sorted descending");
  }
}

void test_seeded_generation_and_checkpoint() {
  Runner runner(tiny_config());
  const std::vector<std::int64_t> prompt{1, 2, 3};
  const GenerationConfig sampling{.maximum_new_tokens = 4,
                                  .sampling_vocabulary_size = 23,
                                  .seed = 99,
                                  .temperature = 0.8,
                                  .top_k = 7,
                                  .top_p = 0.9,
                                  .eos_token_ids = {22}};
  torch::manual_seed(1729);
  const auto expected_rng_continuation = torch::rand({8});
  torch::manual_seed(1729);
  const auto first = runner.generate(prompt, sampling);
  const auto actual_rng_continuation = torch::rand({8});
  require(torch::equal(actual_rng_continuation, expected_rng_continuation),
          "generation changed the caller's CPU random-generator state");
  const auto second = runner.generate(prompt, sampling);
  require(first.output_ids == second.output_ids,
          "seeded sampling is not repeatable");
  require(first.output_ids.size() <= sampling.maximum_new_tokens,
          "generation exceeded maximum_new_tokens");
  require(first.metrics.generated_tokens == first.output_ids.size() &&
              first.metrics.elapsed_milliseconds >=
                  first.metrics.time_to_first_token_milliseconds &&
              first.metrics.generated_tokens_per_second >= 0.0,
          "generation metrics are inconsistent");

  auto single_token_sampling = sampling;
  single_token_sampling.maximum_new_tokens = 2;
  single_token_sampling.sampling_vocabulary_size = 1;
  single_token_sampling.top_k = 0;
  single_token_sampling.eos_token_ids.clear();
  const auto vocabulary_bounded =
      runner.generate(prompt, single_token_sampling);
  require(vocabulary_bounded.output_ids == std::vector<std::int64_t>({0, 0}),
          "generation sampled a padded row beyond the tokenizer vocabulary");

  const std::vector<std::int64_t> training_tokens{1, 2, 3, 4, 5, 6};
  static_cast<void>(runner.train_step(training_tokens));
  static_cast<void>(runner.train_step(training_tokens));
  const auto trained_output = runner.generate(prompt, sampling);

  const auto checkpoint = temporary_checkpoint("snnbase-chatbot-runner");
  runner.save_checkpoint(checkpoint);
  Runner restored(tiny_config());
  restored.load_checkpoint(checkpoint);
  const auto restored_output = restored.generate(prompt, sampling);
  require(trained_output.output_ids == restored_output.output_ids,
          "checkpoint round trip changed deterministic generation");
  require(restored.training_state().micro_batch_count == 2U &&
              restored.training_state().optimizer_step_count == 2U,
          "checkpoint did not restore training counters");
  const auto original_next = runner.train_step(training_tokens);
  const auto restored_next = restored.train_step(training_tokens);
  require(std::abs(original_next.loss - restored_next.loss) < 1.0e-7 &&
              runner.generate(prompt, sampling).output_ids ==
                  restored.generate(prompt, sampling).output_ids,
          "checkpoint did not restore optimizer-resume equivalence");

  auto incompatible_config = tiny_config();
  incompatible_config.gradient_accumulation_steps = 2;
  Runner incompatible(incompatible_config);
  bool rejected_incompatible = false;
  try {
    incompatible.load_checkpoint(checkpoint);
  } catch (const std::runtime_error&) {
    rejected_incompatible = true;
  }
  require(rejected_incompatible,
          "checkpoint accepted an incompatible optimizer configuration");
  std::filesystem::remove(checkpoint);
}

void test_qwen_checkpoint_provenance_contract() {
  using snnbase_experiments::chatbot::QwenDenseLoadResult;
  using snnbase_experiments::chatbot::qwen3_phase0_config_sha256;
  using snnbase_experiments::chatbot::qwen3_phase0_dense_payload_bytes;
  using snnbase_experiments::chatbot::qwen3_phase0_dense_tensor_count;
  using snnbase_experiments::chatbot::qwen3_phase0_model_id;
  using snnbase_experiments::chatbot::qwen3_phase0_revision;
  using snnbase_experiments::chatbot::
      qwen3_phase0_source_checkpoint_sha256;
  using snnbase_experiments::chatbot::
      qwen3_phase0_tokenizer_fingerprint_sha256;

  QwenDenseLoadResult provenance{
      .model_id = std::string(qwen3_phase0_model_id),
      .revision = std::string(qwen3_phase0_revision),
      .archive_sha256 = std::string(64U, 'a'),
      .source_checkpoint_sha256 =
          std::string(qwen3_phase0_source_checkpoint_sha256),
      .config_sha256 = std::string(qwen3_phase0_config_sha256),
      .tokenizer_fingerprint_sha256 =
          std::string(qwen3_phase0_tokenizer_fingerprint_sha256),
      .metadata_sha256 = std::string(64U, 'b'),
      .payload_sha256 = std::string(64U, 'c'),
      .loaded_tensor_count = qwen3_phase0_dense_tensor_count,
      .loaded_payload_bytes = qwen3_phase0_dense_payload_bytes,
      .untouched_runtime_parameter_count = 0U};
  snnbase_experiments::chatbot::
      require_qwen3_phase0_checkpoint_provenance(provenance);
  const auto rejected = [](const QwenDenseLoadResult& candidate) {
    try {
      snnbase_experiments::chatbot::
          require_qwen3_phase0_checkpoint_provenance(candidate);
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };

  auto mismatch = provenance;
  mismatch.model_id = "Qwen/Qwen3-0.6B";
  require(rejected(mismatch), "restored Qwen model_id mismatch was accepted");
  mismatch = provenance;
  mismatch.revision[0] = '0';
  require(rejected(mismatch), "restored Qwen revision mismatch was accepted");
  mismatch = provenance;
  mismatch.source_checkpoint_sha256[0] = '0';
  require(rejected(mismatch),
          "restored Qwen source checkpoint mismatch was accepted");
  mismatch = provenance;
  mismatch.config_sha256[0] = '0';
  require(rejected(mismatch), "restored Qwen config mismatch was accepted");
  mismatch = provenance;
  mismatch.tokenizer_fingerprint_sha256[0] = '0';
  require(rejected(mismatch),
          "restored Qwen tokenizer fingerprint mismatch was accepted");
  mismatch = provenance;
  --mismatch.loaded_tensor_count;
  require(rejected(mismatch),
          "restored Qwen dense tensor-count mismatch was accepted");
  mismatch = provenance;
  --mismatch.loaded_payload_bytes;
  require(rejected(mismatch),
          "restored Qwen dense payload-byte mismatch was accepted");
}

void test_bounded_jsonl_protocol() {
  Runner runner(tiny_config());
  const auto checkpoint =
      temporary_checkpoint("snnbase-chatbot-protocol-flush");
  std::filesystem::remove(checkpoint);
  std::istringstream input(
      "{\"protocol\":\"snnbase.chatbot.tokens/v1\",\"request_id\":\"m1\",\"op\":\"metadata\"}\n"
      "{\"protocol\":\"wrong\",\"request_id\":\"bad\",\"op\":\"metadata\"}\n"
      "{\"protocol\":\"snnbase.chatbot.tokens/v1\",\"request_id\":\"g1\",\"op\":\"generate\",\"input_ids\":[1,2],\"max_new_tokens\":2,\"sampling_vocabulary_size\":23,\"seed\":7,\"temperature\":0}\n"
      "{\"protocol\":\"snnbase.chatbot.tokens/v1\",\"request_id\":\"i1\",\"op\":\"inspect\",\"input_ids\":[1,2],\"probe_token_ids\":[0,7],\"top_k\":3}\n"
      "{\"protocol\":\"snnbase.chatbot.tokens/v1\",\"request_id\":\"f1\",\"op\":\"flush\"}\n"
      "{\"protocol\":\"snnbase.chatbot.tokens/v1\",\"request_id\":\"s1\",\"op\":\"shutdown\"}\n");
  std::ostringstream output;
  const auto serve_result =
      snnbase_experiments::chatbot::serve_token_protocol(
          input, output, runner, {}, checkpoint);
  const auto text = output.str();
  require(text.find("\"request_id\":\"m1\",\"ok\":true") !=
              std::string::npos &&
              text.find("\"parameter_count\":") != std::string::npos &&
              text.find(
                  "\"build_provenance\":{\"experiments\":{\"revision\":\"unknown\",\"dirty\":true},\"snnbase\":{\"revision\":\"unknown\",\"dirty\":true}}") !=
                  std::string::npos,
          "metadata protocol response is incomplete");
  require(text.find("\"request_id\":\"\",\"ok\":false") !=
              std::string::npos &&
              text.find("unsupported token protocol") != std::string::npos,
          "invalid protocol request was not rejected");
  require(text.find("\"request_id\":\"g1\",\"ok\":true") !=
              std::string::npos &&
              text.find("\"output_ids\":[") != std::string::npos,
          "generation protocol response is incomplete");
  require(text.find("\"request_id\":\"f1\",\"ok\":true") !=
              std::string::npos &&
              text.find("\"training_state\":") != std::string::npos,
          "flush protocol response is incomplete");
  require(text.find("\"request_id\":\"i1\",\"ok\":true") !=
              std::string::npos &&
              text.find("\"final_position\":{\"probes\":[") !=
                  std::string::npos,
          "inspect protocol response is incomplete");
  require(text.find("\"request_id\":\"s1\",\"ok\":true") !=
              std::string::npos,
          "shutdown response is missing");
  require(std::filesystem::is_regular_file(checkpoint),
          "successful flush did not materialize the configured checkpoint");
  require(serve_result.checkpoint_current,
          "shutdown lost the current flush checkpoint state");
  Runner restored(tiny_config());
  restored.load_checkpoint(checkpoint);
  require(restored.training_state().pending_accumulation_steps == 0U,
          "protocol flush checkpoint retained partial accumulation");
  std::filesystem::remove(checkpoint);
}

void test_exact_operation_field_contracts() {
  Runner runner(tiny_config());
  std::istringstream input(
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"m_extra","op":"metadata","input_ids":[]})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"r_extra","op":"reset","seed":1})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"f_extra","op":"flush","reset_state":false})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"s_extra","op":"shutdown","top_k":1})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"g_missing_input","op":"generate","max_new_tokens":1,"seed":1})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"g_missing_max","op":"generate","input_ids":[1],"seed":1})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"g_missing_seed","op":"generate","input_ids":[1],"max_new_tokens":1})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"g_extra","op":"generate","input_ids":[1],"max_new_tokens":1,"sampling_vocabulary_size":23,"seed":1,"loss_mask":[1]})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"g_missing_vocab","op":"generate","input_ids":[1],"max_new_tokens":1,"seed":1})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"i_missing_input","op":"inspect","probe_token_ids":[1],"top_k":1})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"i_missing_probes","op":"inspect","input_ids":[1],"top_k":1})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"i_missing_top","op":"inspect","input_ids":[1],"probe_token_ids":[1]})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"i_extra","op":"inspect","input_ids":[1],"probe_token_ids":[1],"top_k":1,"seed":1})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"t_missing","op":"train"})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"t_extra","op":"train","input_ids":[1,2],"top_p":0.5})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"t_empty_mask","op":"train","input_ids":[1,2],"loss_mask":[]})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"e_missing","op":"evaluate"})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"e_extra","op":"evaluate","input_ids":[1,2],"eos_token_ids":[2]})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"e_short_mask","op":"evaluate","input_ids":[1,2],"loss_mask":[1]})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"done","op":"shutdown"})"
      "\n");
  std::ostringstream output;
  const auto serve_result =
      snnbase_experiments::chatbot::serve_token_protocol(input, output, runner);
  const auto responses = output.str();

  require_protocol_error(
      responses, "m_extra",
      "metadata does not accept request field: input_ids");
  require_protocol_error(responses, "r_extra",
                         "reset does not accept request field: seed");
  require_protocol_error(
      responses, "f_extra",
      "flush does not accept request field: reset_state");
  require_protocol_error(responses, "s_extra",
                         "shutdown does not accept request field: top_k");
  require_protocol_error(responses, "g_missing_input",
                         "generate requires request field: input_ids");
  require_protocol_error(responses, "g_missing_max",
                         "generate requires request field: max_new_tokens");
  require_protocol_error(responses, "g_missing_seed",
                         "generate requires request field: seed");
  require_protocol_error(
      responses, "g_extra",
      "generate does not accept request field: loss_mask");
  require_protocol_error(
      responses, "g_missing_vocab",
      "generate requires request field: sampling_vocabulary_size");
  require_protocol_error(responses, "i_missing_input",
                         "inspect requires request field: input_ids");
  require_protocol_error(responses, "i_missing_probes",
                         "inspect requires request field: probe_token_ids");
  require_protocol_error(responses, "i_missing_top",
                         "inspect requires request field: top_k");
  require_protocol_error(responses, "i_extra",
                         "inspect does not accept request field: seed");
  require_protocol_error(responses, "t_missing",
                         "train requires request field: input_ids");
  require_protocol_error(responses, "t_extra",
                         "train does not accept request field: top_p");
  require_protocol_error(responses, "t_empty_mask",
                         "loss_mask length does not match input_ids");
  require_protocol_error(responses, "e_missing",
                         "evaluate requires request field: input_ids");
  require_protocol_error(
      responses, "e_extra",
      "evaluate does not accept request field: eos_token_ids");
  require_protocol_error(responses, "e_short_mask",
                         "loss_mask length does not match input_ids");
  require(responses.find("\"request_id\":\"done\",\"ok\":true") !=
              std::string::npos,
          "an invalid shutdown terminated service before a valid shutdown");
  require(!serve_result.checkpoint_current,
          "service without a checkpoint path reported a current checkpoint");
}

void test_protocol_checkpoint_currency() {
  auto config = tiny_config();
  config.gradient_accumulation_steps = 2;
  const auto checkpoint =
      temporary_checkpoint("snnbase-chatbot-protocol-currency");
  std::filesystem::remove(checkpoint);

  Runner dirty_runner(config);
  std::istringstream dirty_input(
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"f1","op":"flush"})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"t1","op":"train","input_ids":[1,2,3,4]})"
      "\n");
  std::ostringstream dirty_output;
  const auto dirty_result =
      snnbase_experiments::chatbot::serve_token_protocol(
          dirty_input, dirty_output, dirty_runner, {}, checkpoint);
  require(!dirty_result.checkpoint_current,
          "training after a successful checkpoint flush was not marked dirty");

  Runner stale(config);
  stale.load_checkpoint(checkpoint);
  require(stale.training_state().micro_batch_count == 0U,
          "pre-training flush checkpoint unexpectedly contains later state");

  if (!dirty_result.checkpoint_current) {
    static_cast<void>(dirty_runner.flush_optimizer());
    dirty_runner.save_checkpoint(checkpoint);
  }
  Runner exit_saved(config);
  exit_saved.load_checkpoint(checkpoint);
  require(exit_saved.training_state().micro_batch_count == 1U &&
              exit_saved.training_state().optimizer_step_count == 1U &&
              exit_saved.training_state().pending_accumulation_steps == 0U,
          "exit save did not flush and persist dirty accumulated training");

  Runner current_runner(config);
  std::istringstream current_input(
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"t2","op":"train","input_ids":[1,2,3,4]})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"f2","op":"flush"})"
      "\n"
      R"({"protocol":"snnbase.chatbot.tokens/v1","request_id":"s2","op":"shutdown"})"
      "\n");
  std::ostringstream current_output;
  const auto current_result =
      snnbase_experiments::chatbot::serve_token_protocol(
          current_input, current_output, current_runner, {}, checkpoint);
  require(current_result.checkpoint_current,
          "successful post-training flush was not reported current");
  Runner current(config);
  current.load_checkpoint(checkpoint);
  require(current.training_state().micro_batch_count == 1U &&
              current.training_state().optimizer_step_count == 1U &&
              current.training_state().pending_accumulation_steps == 0U,
          "current flush checkpoint did not contain accumulated training");
  std::filesystem::remove(checkpoint);
}

}  // namespace

int main() {
  test_masked_training_and_evaluation();
  test_nonfinite_runner_configuration_is_rejected();
  test_gradient_accumulation_flush_and_schedule();
  test_cosine_schedule_endpoints();
  test_bounded_logit_inspection();
  test_seeded_generation_and_checkpoint();
  test_qwen_checkpoint_provenance_contract();
  test_bounded_jsonl_protocol();
  test_exact_operation_field_contracts();
  test_protocol_checkpoint_currency();
}
