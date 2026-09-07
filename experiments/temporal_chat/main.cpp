#include "model.hpp"
#include <charconv>
#include <iostream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

namespace {
constexpr std::size_t maximum_line = 65536, maximum_token = 2000000;
std::string line() {
  std::string result; char ch;
  while (std::cin.get(ch)) {
    if (ch == '\n') return result;
    if (result.size() >= maximum_line) throw std::invalid_argument("protocol line too long");
    result.push_back(ch);
  }
  if (!result.empty()) return result;
  throw std::invalid_argument("unexpected end of protocol");
}
class Fields {
 public:
  explicit Fields(std::string value) : text_(std::move(value)), stream_(text_) {}
  void word(const char* expected) {
    std::string value; if (!(stream_ >> value) || value != expected)
      throw std::invalid_argument("unexpected protocol command");
  }
  std::size_t number(std::size_t min, std::size_t max) {
    std::string value; std::size_t result = 0;
    if (!(stream_ >> value)) throw std::invalid_argument("missing protocol integer");
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < min || result > max)
      throw std::invalid_argument("protocol integer outside limits");
    return result;
  }
  void end() { std::string value; if (stream_ >> value) throw std::invalid_argument("trailing protocol fields"); }
 private:
  std::string text_; std::istringstream stream_;
};
std::vector<temporal_chat::Token> tokens(Fields& fields, std::size_t size) {
  std::vector<temporal_chat::Token> values;
  for (std::size_t i = 0; i < size; ++i)
    values.push_back(static_cast<temporal_chat::Token>(fields.number(0, maximum_token)));
  return values;
}
std::vector<temporal_chat::Example> examples(std::size_t count) {
  std::vector<temporal_chat::Example> values;
  for (std::size_t i = 0; i < count; ++i) {
    Fields fields(line()); fields.word("PAIR");
    const auto np = fields.number(1, 128), na = fields.number(1, 32);
    auto prompt = tokens(fields, np), answer = tokens(fields, na); fields.end();
    values.push_back({std::move(prompt), std::move(answer)});
  }
  return values;
}
void ids(const std::vector<temporal_chat::Token>& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
  std::cout << ']';
}
void metric(const char* name, const temporal_chat::Metrics& m) {
  std::cout << '"' << name << "\":{\"loss\":" << m.loss << ",\"accuracy\":" << m.accuracy << ",\"tokens\":" << m.tokens << '}';
}
}
int main() {
  try {
    std::cout << std::setprecision(12);
    Fields header(line()); header.word("SNNBASE_TEMPORAL_CHAT_V1");
    const auto seed = static_cast<std::uint32_t>(header.number(0, std::numeric_limits<std::uint32_t>::max()));
    const auto epochs = header.number(1, 300);
    const auto stop = static_cast<temporal_chat::Token>(header.number(0, maximum_token));
    const auto nt = header.number(1, 128), nh = header.number(1, 32);
    const auto stdp = header.number(0, 1), lesion = header.number(0, 1); header.end();
    const auto train = examples(nt), held = examples(nh);
    std::set<std::vector<temporal_chat::Token>> train_prompts;
    std::set<temporal_chat::Token> outputs;
    std::size_t sample_count = 0;
    for (const auto& e : train) {
      if (!train_prompts.insert(e.prompt).second) throw std::invalid_argument("duplicate training prompt");
      outputs.insert(e.answer.begin(), e.answer.end()); sample_count += e.answer.size();
      if (e.answer.back() != stop || std::find(e.answer.begin(), e.answer.end() - 1, stop) != e.answer.end() - 1)
        throw std::invalid_argument("training answer must contain stop only at the end");
    }
    std::set<std::vector<temporal_chat::Token>> held_prompts;
    for (const auto& e : held) {
      if (!held_prompts.insert(e.prompt).second) throw std::invalid_argument("duplicate held-out prompt");
      if (train_prompts.contains(e.prompt)) throw std::invalid_argument("held-out prompt overlaps training");
      if (e.answer.back() != stop || std::find(e.answer.begin(), e.answer.end() - 1, stop) != e.answer.end() - 1)
        throw std::invalid_argument("held-out answer must contain stop only at the end");
    }
    if (sample_count > 2048 || outputs.size() > 512 ||
        static_cast<double>(sample_count) * outputs.size() * temporal_chat::Model::features * epochs > 1.5e9)
      throw std::invalid_argument("training exceeds toy experiment resource budget");
    temporal_chat::Model model({outputs.begin(), outputs.end()}, seed, lesion != 0);
    const double warmup_change = stdp ? model.warmup(train) : 0.0;
    // Deterministic reservoir features can be replayed because its synapses
    // are fixed; this avoids rerunning spike integration on every epoch.
    const auto train_features = model.encode(train);
    const auto untrained = model.evaluate(model.encode(held));
    model.fit(train_features, epochs); model.freeze();
    const auto trained_train = model.evaluate(train_features);
    const auto held_features = model.encode(held);
    const auto trained = model.evaluate(held_features);
    std::vector<double> counts(outputs.size(), 0.1);
    for (const auto& sample : train_features) counts[sample.target] += 1.0;
    const auto total = std::accumulate(counts.begin(), counts.end(), 0.0);
    const auto best = static_cast<std::size_t>(std::max_element(counts.begin(), counts.end()) - counts.begin());
    temporal_chat::Metrics unigram; unigram.tokens = held_features.size();
    for (const auto& sample : held_features) {
      unigram.loss -= std::log(counts[sample.target] / total);
      unigram.accuracy += sample.target == best ? 1.0 : 0.0;
    }
    unigram.loss /= unigram.tokens; unigram.accuracy /= unigram.tokens;
    std::cout << "{\"ok\":true,\"protocol\":\"snnbase-temporal-chat-v1\",\"event\":\"trained\",\"seed\":" << seed
              << ",\"stdp_enabled\":" << (stdp ? "true" : "false") << ",\"reservoir_lesioned\":" << (lesion ? "true" : "false")
              << ",\"stdp_absolute_weight_change\":" << warmup_change << ",\"epochs\":" << epochs << ",\"neurons\":" << model.neuron_count()
              << ",\"synapses\":" << model.synapse_count() << ",\"readout_features\":" << temporal_chat::Model::features
              << ",\"readout_parameters\":" << model.weights().size() << ",\"output_vocabulary\":";
    ids(model.vocabulary()); std::cout << ','; metric("untrained_held_out", untrained);
    std::cout << ','; metric("unigram_held_out", unigram);
    std::cout << ','; metric("trained_train", trained_train);
    std::cout << ','; metric("trained_held_out", trained);
    std::cout << ",\"held_out_generations\":[";
    std::size_t exact = 0;
    for (std::size_t i = 0; i < held.size(); ++i) {
      auto generated = model.generate(held[i].prompt, stop, 32);
      exact += generated == held[i].answer ? 1 : 0;
      if (i) std::cout << ',';
      std::cout << "{\"ids\":"; ids(generated);
      std::cout << ",\"expected_ids\":"; ids(held[i].answer); std::cout << '}';
    }
    std::cout << "],\"held_out_exact_match\":" << static_cast<double>(exact) / held.size() << "}\n" << std::flush;
    // Inference cannot call fit after freeze, and never changes weights.
    while (std::cin.peek() != std::char_traits<char>::eof()) {
      auto request = line(); if (request == "QUIT") break;
      Fields fields(std::move(request)); fields.word("GENERATE");
      const auto maximum = fields.number(1, 64), count = fields.number(1, 128);
      const auto prompt = tokens(fields, count); fields.end();
      const auto generated = model.generate(prompt, stop, maximum);
      std::cout << "{\"ok\":true,\"event\":\"generated\",\"ids\":";
      ids(generated);
      std::cout << ",\"spikes\":" << model.spike_count() << ",\"stopped\":"
                << (generated.back() == stop ? "true" : "false") << "}\n" << std::flush;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "temporal_chat: " << error.what() << '\n'; return 2;
  }
}
