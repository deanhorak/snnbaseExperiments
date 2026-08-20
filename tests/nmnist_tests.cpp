#include <snnbase_experiments/nmnist.hpp>

#include <cassert>
#include <filesystem>
#include <fstream>

namespace {

void write_record(std::ofstream& output, unsigned char x, unsigned char y,
                  bool polarity, std::uint32_t timestamp) {
  const unsigned char bytes[] = {
      x, y, static_cast<unsigned char>((polarity ? 0x80U : 0U) | ((timestamp >> 16U) & 0x7FU)),
      static_cast<unsigned char>((timestamp >> 8U) & 0xFFU),
      static_cast<unsigned char>(timestamp & 0xFFU)};
  output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

}  // namespace

int main() {
  const auto root = std::filesystem::temp_directory_path() / "snnbase-nmnist-test";
  const auto sample_path = root / "Train" / "3" / "sample.bin";
  std::filesystem::create_directories(sample_path.parent_path());
  {
    std::ofstream output(sample_path, std::ios::binary);
    assert(output);
    write_record(output, 1, 2, false, 4);
    write_record(output, 3, 4, true, 12);
  }
  const auto events = snnbase_experiments::nmnist::load_events(sample_path);
  assert(events.size() == 2);
  assert(events[0].timestamp == 4 && !events[0].polarity);
  assert(events[1].timestamp == 12 && events[1].polarity);
  const auto sequence = snnbase_experiments::nmnist::encode_events(events, 2);
  assert(sequence.size() == 2);
  assert(!sequence[0].empty() && !sequence[1].empty());
  const snnbase_experiments::nmnist::Dataset dataset{
      {{{0, events},
        {1, {{20, 20, false, 4}, {21, 20, true, 12}}}}}};
  snnbase_experiments::nmnist::Classifier classifier{2, 0.5F, 0.5F};
  classifier.train(dataset);
  const auto evaluation = classifier.evaluate(dataset);
  assert(evaluation.total == dataset.size());
  assert(evaluation.correct == dataset.size());
  std::filesystem::remove_all(root);
}
