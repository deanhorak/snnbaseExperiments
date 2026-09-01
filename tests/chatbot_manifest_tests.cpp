#include <snnbase_experiments/chatbot/manifest.hpp>

#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using namespace snnbase_experiments::chatbot;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

FileDigest file(std::string path, std::string hash, std::uint64_t bytes) {
  return {.path = std::move(path), .sha256 = std::move(hash), .bytes = bytes};
}

RunManifest valid_manifest() {
  constexpr auto zeros64 =
      "0000000000000000000000000000000000000000000000000000000000000000";
  constexpr auto ones40 = "1111111111111111111111111111111111111111";
  constexpr auto twos40 = "2222222222222222222222222222222222222222";
  RunManifest manifest{
      .run_id = "chatbot-ann-seed42-smoke",
      .created_utc = "2026-09-01T12:34:56Z",
      .status = RunStatus::planned,
      .architecture = ChatbotArchitecture::ann,
      .seed = 42,
      .publishable = false,
      .experiments_repository =
          {.path = "/src/snnbaseExperiments", .revision = ones40, .dirty = true},
      .library_repository =
          {.path = "/src/snnbase", .revision = twos40, .dirty = false},
      .executable = file("build/chatbot_experiment", zeros64, 300),
      .config = file("configs/chatbot/ann-baseline.json", zeros64, 100),
      .dataset =
          {.name = "generated-test-dialogues",
           .version = "fixture-v1",
           .source_uri = "https://example.invalid/dataset-card",
           .license = "CC0-1.0",
           .source = file("data/chatbot/train.jsonl", zeros64, 200),
           .records = 10,
           .split = {.seed = 42,
                     .validation_basis_points = 1000,
                     .test_basis_points = 1000,
                     .train_records = 8,
                     .validation_records = 1,
                     .test_records = 1}},
      .reference_model =
          {.repository = std::string(qwen3_reference_repository),
           .revision = std::string(qwen3_reference_revision),
           .license = "Apache-2.0",
           .files = {file(
               "config.json",
               "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
               727)}},
      .tokenizer =
          {.repository = std::string(qwen3_reference_repository),
           .revision = std::string(qwen3_reference_revision),
           .license = "Apache-2.0",
           .files =
               {file("tokenizer.json",
                     "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
                     1),
                file("tokenizer_config.json",
                     "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
                     1)}},
      .command = {"chatbot_experiment", "--config", "ann-baseline.json"}};
  return manifest;
}

void test_sha256(const std::filesystem::path& path) {
  require(sha256_hex("") ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "SHA-256 empty-string vector failed");
  require(sha256_hex("abc") ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256 abc vector failed");
  require(
      sha256_hex(
          "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
      "SHA-256 multi-block vector failed");
  {
    std::ofstream output(path, std::ios::binary);
    output << "abc";
  }
  const auto digest = digest_file(path);
  require(digest.sha256 == sha256_hex("abc"),
          "file SHA-256 differs from byte SHA-256");
  require(digest.bytes == 3U, "file digest has the wrong byte count");
}

void test_manifest_serialization() {
  const auto manifest = valid_manifest();
  validate_run_manifest(manifest);
  std::ostringstream output;
  write_run_manifest_json(output, manifest);
  const auto json = output.str();
  require(json.find("\"schema_version\": \"snnbase-chatbot-run-v1\"") !=
              std::string::npos,
          "serialized manifest omitted its schema version");
  require(json.find(std::string(qwen3_reference_revision)) != std::string::npos,
          "serialized manifest omitted the immutable model revision");
  require(json.find("\"assistant_only_loss\":true") != std::string::npos,
          "serialized manifest omitted the assistant-only loss contract");
  require(json.find("\"publishable\": false") != std::string::npos,
          "serialized manifest changed the publication gate");
}

void test_manifest_rejections() {
  {
    auto manifest = valid_manifest();
    manifest.reference_model.revision = "main";
    bool threw = false;
    try {
      validate_run_manifest(manifest);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "manifest accepted a mutable model revision");
  }
  {
    auto manifest = valid_manifest();
    manifest.dataset.source.sha256 = "unknown";
    bool threw = false;
    try {
      validate_run_manifest(manifest);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "manifest accepted an unresolved dataset hash");
  }
  {
    auto manifest = valid_manifest();
    manifest.publishable = true;
    manifest.status = RunStatus::completed;
    bool threw = false;
    try {
      validate_run_manifest(manifest);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "publication gate accepted a dirty, evidence-free run");
  }
}

void test_publishable_manifest() {
  auto manifest = valid_manifest();
  manifest.publishable = true;
  manifest.status = RunStatus::completed;
  manifest.experiments_repository.dirty = false;
  manifest.toolchain = {.compiler = "gcc 13.3",
                        .cmake = "cmake 3.28",
                        .torch = "2.5.1",
                        .cuda = "12.1"};
  manifest.hardware = {.cpu = "test cpu", .gpu = "test gpu"};
  manifest.artifacts = {
      {.kind = "checkpoint",
       .file = file(
           "checkpoint.safetensors",
           "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
           123)}};
  manifest.metrics = {{.id = "validation_loss",
                       .value = 1.25,
                       .unit = "nats_per_token"}};
  validate_run_manifest(manifest);
}

}  // namespace

int main() {
  const auto directory = std::filesystem::temp_directory_path() /
                         "snnbase-chatbot-manifest-tests";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  try {
    test_sha256(directory / "abc.txt");
    test_manifest_serialization();
    test_manifest_rejections();
    test_publishable_manifest();
    if (const auto* output_path =
            std::getenv("SNNBASE_CHATBOT_MANIFEST_TEST_OUTPUT");
        output_path != nullptr) {
      std::ofstream output(output_path, std::ios::binary);
      if (!output) {
        throw std::runtime_error("could not create manifest schema fixture");
      }
      write_run_manifest_json(output, valid_manifest());
    }
  } catch (...) {
    std::filesystem::remove_all(directory);
    throw;
  }
  std::filesystem::remove_all(directory);
}
