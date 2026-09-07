#include <snnbase_experiments/nmnist.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

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
    require(static_cast<bool>(output), "could not create event fixture");
    write_record(output, 1, 2, false, 4);
    write_record(output, 3, 4, true, 12);
  }
  const auto events = snnbase_experiments::nmnist::load_events(sample_path);
  require(events.size() == 2, "event count mismatch");
  require(events[0].timestamp == 4 && !events[0].polarity, "first event mismatch");
  require(events[1].timestamp == 12 && events[1].polarity, "second event mismatch");
  const auto sequence = snnbase_experiments::nmnist::encode_events(events, 2);
  require(sequence.size() == 2, "encoded sequence length mismatch");
  require(!sequence[0].empty() && !sequence[1].empty(), "encoded event is empty");
  const snnbase_experiments::nmnist::Dataset dataset{
      {{{0, events},
        {1, {{20, 20, false, 4}, {21, 20, true, 12}}}}}};
  snnbase_experiments::nmnist::Classifier classifier{2, 0.5F, 0.5F};
  classifier.train(dataset);
  const auto evaluation = classifier.evaluate(dataset);
  require(evaluation.total == dataset.size(), "evaluation count mismatch");
  require(evaluation.correct == dataset.size(), "classifier did not learn fixture");
  std::filesystem::remove_all(root);
}
