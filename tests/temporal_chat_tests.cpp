#include "model.hpp"
#include <iostream>
#include <stdexcept>
using temporal_chat::Model;
using temporal_chat::Example;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
int main() {
  try {
    Model model({10, 11, 12, 99});
    const std::vector<Example> train{{{1, 2, 7}, {10, 12, 99}}, {{1, 3, 7}, {11, 12, 99}},
                                    {{4, 2, 7}, {10, 12, 99}}, {{4, 3, 7}, {11, 12, 99}}};
    const std::vector<Example> held{{{5, 2, 7}, {10, 12, 99}}, {{5, 3, 7}, {11, 12, 99}}};
    model.reset(); static_cast<void>(model.observe(2)); const auto a = model.observe(7);
    model.reset(); static_cast<void>(model.observe(3)); const auto b = model.observe(7);
    require(a != b, "spiking context does not distinguish histories with the same final token");
    model.reset(); static_cast<void>(model.observe(2)); require(a == model.observe(7), "reset is not deterministic");
    require(model.spike_count() > 0, "no neurons fired");
    const auto train_samples = model.encode(train), held_samples = model.encode(held);
    const auto before = model.evaluate(held_samples);
    model.fit(train_samples, 120); model.freeze();
    const auto after = model.evaluate(held_samples);
    require(after.loss < before.loss * 0.7, "held-out loss did not improve through learning");
    require(model.evaluate(train_samples).accuracy > 0.95, "readout failed to learn tiny task");
    const auto weights = model.weights();
    const auto left = model.generate(train[0].prompt, 99, 8);
    const auto right = model.generate(train[1].prompt, 99, 8);
    require(left == train[0].answer && right == train[1].answer, "autoregressive generation ignored prompt or failed to learn sequence");
    require(left != right, "generation is prompt-independent");
    require(weights == model.weights(), "inference changed learned weights");
    bool blocked = false;
    try { model.fit(train_samples, 1); } catch (const std::logic_error&) { blocked = true; }
    require(blocked, "frozen model accepted training");
    Model replay({10, 11, 12, 99}); replay.fit(replay.encode(train), 120); replay.freeze();
    require(replay.weights() == weights, "seeded training is not reproducible");
    require(replay.generate(train[0].prompt, 99, 8) == left, "seeded generation differs");
    std::cout << "temporal chat learning, causal state, reset, autoregression, and freeze passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
