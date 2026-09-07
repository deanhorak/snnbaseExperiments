# snnbase Experiments

Reproducible C++20 experiments built on
[`snnbase`](https://github.com/deanhorak/snnbase). The repository keeps experimental code,
configuration, validation, and result-reporting conventions separate from the core library.

The repository also contains a scaled Spaun-inspired experiment built on
`snnbase`, including A0--A7 task protocols, direct spiking recall routes,
reward-plastic action selection, a delayed spiking counting chain, a two-joint
arm environment, and an optional native OpenGL GUI. It is not yet behaviorally
equivalent to Spaun: host orchestration still selects routes and performs parts
of A5--A7. See
[docs/SPAUN_EXPERIMENT.md](docs/SPAUN_EXPERIMENT.md) for scope, fidelity limits,
build instructions, and controls. The current task-specific results and memory
sweep are recorded in
[SPAUN_BEHAVIORAL_BENCHMARK.md](docs/results/SPAUN_BEHAVIORAL_BENCHMARK.md) and
[SPAUN_MEMORY_PARAMETER_SWEEP.md](docs/results/SPAUN_MEMORY_PARAMETER_SWEEP.md).

## Experiments

### MNIST digit classification

The first harness reads the standard MNIST IDX files, spatially pools each 28×28 grayscale image
into a `snnbase::SpikeEvent`, trains one prototype neuron per digit, and reports accuracy plus a
confusion matrix.

This is an experimental baseline, not a claim of state-of-the-art accuracy. Its purpose is to
exercise `snnbase` on real input data with a repeatable measurement path.

### EMNIST character battery

The EMNIST battery runs all six official tasks: ByClass, ByMerge, Balanced,
Letters, Digits, and EMNIST-MNIST. It reports train time, test time, and
accuracy for each split. Its default structured network uses centered
multi-threshold tiles, learned patch features, class experts, hard-negative
reinforcement, and translation augmentation. Pass `--architecture baseline`
to reproduce the original single-event classifier.

### Deep convolutional spiking classifier

`emnist_deep` adapts EMNIST splits to the library-owned temporal residual SNN.
It evolves LIF membrane state over explicit timesteps, uses exact autograd
through temporal batch normalization, trains thresholds and leaks with a
surrogate gradient, and reports validation accuracy and spike rate separately
from the final test evaluation. The experiment retains only dataset loading,
command-line orchestration, and reporting.

### CIFAR-10 color baseline

`cifar10_experiment` reads the standard CIFAR-10 binary batches, encodes each
32x32 RGB image through `snnbase::spiking_conv::Classifier`, and trains a
same-padding residual spiking convolution model with RGB rate coding,
surrogate-gradient Adam updates, optional SEW residual merging, and checkpoint
support. This is the current native `snnbase` CIFAR path.

`cifar10_deep` is the CIFAR-facing path for `snnbase::temporal::Classifier`.
Its stem and three residual stages preserve the time axis throughout, apply
SEW-add to per-timestep spikes, downsample spatially, and use global pooling
plus a population readout instead of a parameter-heavy flattened dense head.
AdamW, label smoothing, crop/flip/Cutout augmentation, cosine decay with
warmup, stratified validation, spike-rate regularization, and deterministic
seeds are part of the library training API.

### N-MNIST event-stream baseline

`nmnist_experiment` reads native N-MNIST address-event recordings from
`Train/<digit>/*.bin` and `Test/<digit>/*.bin`. It preserves timestamps and
polarity across configurable temporal bins rather than repeating a static
image. This is a benchmark harness; no dataset result is claimed until the
official data and complete run artifacts are available.

Download the official `Train.zip` and `Test.zip` manually from the
[N-MNIST dataset page](https://sites.google.com/site/garrickorchard/datasets/n-mnist),
then validate and extract them:

```sh
./scripts/install_nmnist.sh /path/to/Train.zip /path/to/Test.zip
./build/nmnist_experiment --epochs 1 --time-bins 10
```

### Phase 0 spiking chatbot track

The chatbot work now has two explicit tracks:

- exact `Qwen/Qwen3-0.6B-Base` dense archive import in ANN mode, a bounded
  final-position logit oracle, and a hybrid SNN mode that retains the imported
  dense tensors while adding experiment-owned LIF dynamics; and
- independently trainable compact ANN and hybrid-SNN controls using the same
  verified tokenizer, data contract, and evaluation driver.

The deterministic converter maps all 310 Qwen dense tensors. A local ignored
three-case CPU float32 oracle run matched every probe ID and ordered top-16 ID
within `atol=rtol=1e-5`; this is development evidence, not a promoted durable
baseline. The leakage-remediated pinned OASST2 snapshot below is approved for
internal research under the restrictions in the [dataset approval
record](docs/OASST2_DATASET_APPROVAL.md); this is neither a universal safety
certification nor legal approval to redistribute the data or deploy a public
service. No full training or held-out quality run has been performed, and no
latency, power, or energy claim is supported. The training driver validates and
consumes the checked JSON configs, then launches the core with their canonical
argument arrays. See [the chatbot experiment
contract](docs/CHATBOT_EXPERIMENT.md) and [baseline
status](docs/results/CHATBOT_BASELINE.md).

## Prerequisites

- CMake 3.20 or newer
- A C++20 compiler
- Ninja (when using the provided presets)
- LibTorch 2.3 or newer, with CUDA support for GPU training
- A sibling checkout at `../snnbase`, or an explicit `SNNBASE_SOURCE_DIR`
- `curl`, `gzip`, and `unzip` to use the dataset download helpers
- Python 3.10 or newer for the optional Qwen reference tools; the offline
  contract suite itself uses only the Python standard library

## Build and test

```sh
git clone https://github.com/deanhorak/snnbase.git ../snnbase
cmake --preset default
cmake --build --preset default
ctest --preset default
```

The chatbot data, manifest, protocol, and local reference-fixture contracts can
be checked without `snnbase`, LibTorch, model assets, package installation, or
network access:

```sh
./scripts/check_chatbot_contracts.sh
```

For a checkout elsewhere:

```sh
cmake -S . -B build -G Ninja \
  -DSNNBASE_SOURCE_DIR=/path/to/snnbase \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch/share/cmake
cmake --build build
ctest --test-dir build --output-on-failure
```

The project configures `snnbase` with 128-bit spike events, 96-bit payloads, 256 prototypes per
class, and weighted bitwise matching.

For the LibTorch temporal backend, use the pinned CUDA 12.1/GCC 11 image and
toolchain specification in [containers/temporal.Dockerfile](containers/temporal.Dockerfile).
It matches the validated PyTorch 2.5.1+cu121 configuration and avoids an
inherited `/opt/libtorch` library path. The container recipe has not yet been
built in CI; it has been locally built and passed the temporal library test,
and is the required environment for the five-seed release run.

## Run MNIST

Download the four uncompressed IDX files:

```sh
./scripts/download_mnist.sh
```

They are installed under the ignored `data/mnist/` directory. That location is configured through
the `SNNBASE_EXPERIMENTS_MNIST_DATA_DIR` CMake cache variable, so the executable and local dataset
smoke test do not depend on the shell's working directory. Override it at configure time when data
is stored elsewhere:

```sh
cmake --preset default -DSNNBASE_EXPERIMENTS_MNIST_DATA_DIR=/path/to/mnist
```

Run a quick subset:

```sh
./build/mnist_experiment --train-limit 10000 --test-limit 1000
```

Run the full dataset:

```sh
./build/mnist_experiment --epochs 1
```

With the dataset installed, CTest includes a full accuracy regression test requiring at least 85%:

```sh
ctest --test-dir build -R mnist_accuracy --output-on-failure
```

The initial baseline is documented in
[docs/results/MNIST_BASELINE.md](docs/results/MNIST_BASELINE.md).

## Run EMNIST

Download and extract the
[official NIST binary archive](https://www.nist.gov/itl/products-and-services/emnist-dataset):

```sh
./scripts/download_emnist.sh
```

Run the complete battery:

```sh
./build-release/emnist_battery
```

Run one split or the registered regression battery:

```sh
./build-release/emnist_battery --splits letters
ctest --test-dir build-release -L emnist --output-on-failure
```

Measured results are documented in
[docs/results/EMNIST_BATTERY.md](docs/results/EMNIST_BATTERY.md).

Train the deep SNN:

```sh
./build-release/emnist_deep --split mnist --epochs 30
./build-release/emnist_deep --split digits --epochs 3
```

Architecture and results are documented in
[docs/results/EMNIST_DEEP_SNN.md](docs/results/EMNIST_DEEP_SNN.md).

## Run CIFAR-10

Download and extract the binary CIFAR-10 batches:

```sh
./scripts/download_cifar10.sh
```

Run a quick subset:

```sh
./build-release/cifar10_experiment --train-limit 10000 --test-limit 1000
```

Run the full baseline:

```sh
./build-release/cifar10_experiment --epochs 3 --channels 8 --time-steps 4 --no-normalize
```

Run the residual spiking-conv path on a small smoke subset:

```sh
./build-release/cifar10_deep --epochs 1 --width 4 --train-limit 20 --test-limit 10
```

Run the temporal residual SNN and save resumable model/optimizer/epoch state:

```sh
./build-release/cifar10_deep --epochs 30 --width 32 --blocks 2 \
  --time-steps 4 --seed 42 --device cuda \
  --save-checkpoint checkpoints/cifar10-w32.ckpt
```

Measured results are documented in
[docs/results/CIFAR10_BASELINE.md](docs/results/CIFAR10_BASELINE.md).

The temporal architecture upgrade, full reruns, and before/after accuracy
comparison are documented in
[docs/results/TEMPORAL_ARCHITECTURE_UPGRADE.md](docs/results/TEMPORAL_ARCHITECTURE_UPGRADE.md).

Use `--help` for all options. Dataset files are ignored by Git and must not be committed.

## Run the chatbot tracks

Install the pinned reference tooling and download the immutable Phase 0 Qwen
snapshot (both steps require network access), then verify it and regenerate the
checked-in tokenizer/logit goldens:

```sh
python3 -m venv .venv-chatbot-reference
.venv-chatbot-reference/bin/pip install -r requirements/qwen-reference.lock
SNNBASE_REFERENCE_PYTHON=.venv-chatbot-reference/bin/python \
  ./scripts/download_qwen_assets.sh \
  --model-id Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --expected-tokenizer-fingerprint \
    6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237 \
  --output-dir artifacts/chatbot/qwen3-0.6b-base \
  --include-weights
SNNBASE_REFERENCE_PYTHON=.venv-chatbot-reference/bin/python \
  ./scripts/check_qwen_reference.sh \
  --assets-dir artifacts/chatbot/qwen3-0.6b-base \
  --include-logits
```

The requirements file pins direct package versions but does not contain wheel
hashes or all transitive artifacts; retain the resolved environment manifest
for a promoted reference run. Once the environment and snapshot exist, the
`check_qwen_reference.sh` step itself is offline. Offline implementation checks
remain available through `./scripts/check_chatbot_contracts.sh`.

The tested `snnbase` language-backend revision is
`c282306ee3b80ccc5123fcb8b2da78ae51ed09fe`. Check out that exact revision,
then configure it explicitly and run the tiny CPU smoke tests:

```sh
./scripts/configure_chatbot.sh \
  --snnbase-source /path/to/snnbase \
  --snnbase-revision c282306ee3b80ccc5123fcb8b2da78ae51ed09fe \
  --torch-prefix /path/to/libtorch/share/cmake \
  --build-dir build-chatbot \
  --build-type Release
./scripts/run_chatbot_smoke.sh --build-dir build-chatbot
```

The default configure path pins CUDA 12.1, GCC 11, and SM86. For a CPU-only
LibTorch distribution, add `--toolchain none`; keep `--build-type Release` for
oracle measurements. These are separate supported recipes, and a run manifest
must record which one produced the executable.

The smoke command proves tensor shapes, forward/backward execution, checkpoint
plumbing, and the bounded token protocol on a tiny synthetic sequence. It does
not establish conversational quality.

Convert the exact verified Qwen checkpoint and run the bounded ANN oracle:

```sh
python3 tools/qwen_convert.py convert \
  --assets-dir artifacts/chatbot/qwen3-0.6b-base \
  --model-id Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --expected-tokenizer-fingerprint \
    6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237 \
  --output-dir artifacts/chatbot/qwen3-0.6b-base-dense-v1
./scripts/run_with_chatbot_libtorch.sh --build-dir build-chatbot -- \
  python3 tools/qwen_oracle_gate.py \
  --runner ./build-chatbot/chatbot_experiment \
  --archive artifacts/chatbot/qwen3-0.6b-base-dense-v1/qwen-dense.snnq \
  --archive-sha256 \
    333c8be1b3fde5013ee8d1dcb96f23051dd934f8e1f359aff329e7e5cf8ccac8 \
  --reference tests/fixtures/chatbot/qwen3_phase0_reference_logits.json \
  --output artifacts/chatbot/qwen3-oracle-gate.json
```

Both paths use the same immutable prepared-token contract. The approved
internal-research dataset is derived from the OASST2 ready-tree archive at
revision `179dd21fc55192153d94adb0e0ce8f69e222bf75`. With that exact archive
already at `data/source/oasst2/2023-11-05_oasst2_ready.trees.jsonl.gz`, this
command is offline and creates one immutable conversion bundle:

```sh
.venv-chatbot-reference/bin/python tools/oasst2_convert.py \
  --profile quality05 \
  --assets-dir artifacts/chatbot/qwen3-0.6b-base \
  --model-id Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --expected-tokenizer-fingerprint \
    6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237 \
  --input data/source/oasst2/2023-11-05_oasst2_ready.trees.jsonl.gz \
  --expected-source-sha256 \
    7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b \
  --expected-source-bytes 54370156 \
  --source-uri \
    https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/179dd21fc55192153d94adb0e0ce8f69e222bf75/2023-11-05_oasst2_ready.trees.jsonl.gz \
  --source-version 179dd21fc55192153d94adb0e0ce8f69e222bf75 \
  --source-license Apache-2.0 \
  --license-reviewed --policy-reviewed \
  --output-dir \
    artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-v1 \
  --max-input-tokens 512
```

The `quality05` profile is the recommended development profile; it is a
reproducible selection policy, not a safety certification. Its baseline output
after exact normalized-root-prompt deduplication is 12,427 conversations
(9,990 train, 1,165 validation, and 1,272 test). The versioned fuzzy/semantic
audit in the [approval record](docs/OASST2_DATASET_APPROVAL.md) identified 168
high-confidence cross-split pairs and deterministically excluded 144
lower-priority records. The approved internal-research output contains 12,283
conversations: 9,853 train, 1,158 validation, and 1,272 test. The exact
token-aware 512-token gate is the sequence-length
control; the converter's default 32,768-byte aggregate-content limit remains a
pre-tokenization resource bound. The `conservative-zero` profile is retained as
an audit alternative and is not the default; after the same deduplication it
keeps 1,700 conversations (1,407 train, 140 validation, and 153 test) and nearly
eliminates multi-turn material. Neither profile establishes model quality or
external-use legal approval. The review flags attest only that the declared
Apache-2.0 metadata and the explicit policy were reviewed.

Prepare the converter's canonical `conversations.jsonl`, binding the next
manifest to the canonical SHA-256 recorded by `conversion-manifest.json`, then
run the compact ANN control:

```sh
OASST2_CONVERSION_DIR=artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-leakage-reviewed-v6
OASST2_CANONICAL_SHA256="$(
  python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["output"]["sha256"])' \
    "$OASST2_CONVERSION_DIR/conversion-manifest.json"
)"
.venv-chatbot-reference/bin/python tools/chatbot_prepare.py \
  --assets-dir artifacts/chatbot/qwen3-0.6b-base \
  --model-id Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --expected-tokenizer-fingerprint \
    6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237 \
  --input "$OASST2_CONVERSION_DIR/conversations.jsonl" \
  --conversion-manifest \
    "$OASST2_CONVERSION_DIR/conversion-manifest.json" \
  --source-uri "urn:sha256:$OASST2_CANONICAL_SHA256" \
  --source-version "sha256:$OASST2_CANONICAL_SHA256" \
  --source-license Apache-2.0 \
  --output-dir \
    artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-leakage-reviewed-prepared-v6 \
  --max-input-tokens 512
./scripts/run_with_chatbot_libtorch.sh --build-dir build-chatbot -- \
  python3 tools/chatbot_train.py \
  --dataset-dir \
    artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-leakage-reviewed-prepared-v6 \
  --run-dir artifacts/chatbot/runs/compact-ann-seed42 \
  --core-executable ./build-chatbot/chatbot_experiment \
  --config configs/chatbot/ann-baseline.json
```

The conversion manifest is the upstream link from that canonical hash to the
pinned compressed archive, converter, tokenizer, policy, counts, and lineage.
Preparation verifies that chain, then retains hashed copies as
`source-conversion-manifest.json` and `source-lineage.jsonl`; training validates
them and publication includes both as artifacts. Both `data/` and `artifacts/`
are ignored by Git; do not commit raw or derived data.

The config validator proves that its stored architecture, geometry, optimizer,
and runner argv agree before the core launches. The other three complete
config-driven commands, selection rules, and result gates are in
[`docs/CHATBOT_EXPERIMENT.md`](docs/CHATBOT_EXPERIMENT.md). The driver selects
the lowest validation loss including epoch zero, reloads that checkpoint into
a second process, requires the restored counters and validation metrics to
match, and only then touches the test split once.

Start interactive exact imported ANN inference:

```sh
./scripts/run_with_chatbot_libtorch.sh --build-dir build-chatbot -- \
  .venv-chatbot-reference/bin/python tools/chatbot_cli.py \
  --assets-dir artifacts/chatbot/qwen3-0.6b-base \
  --core-executable ./build-chatbot/chatbot_experiment \
  --qwen-archive artifacts/chatbot/qwen3-0.6b-base-dense-v1/qwen-dense.snnq \
  --qwen-archive-sha256 \
    333c8be1b3fde5013ee8d1dcb96f23051dd934f8e1f359aff329e7e5cf8ccac8 \
  --context-tokens 512 \
  --max-new-tokens 128 \
  --seed 42 \
  --temperature 0 \
  --ann --device cpu
```

`/reset` clears local multi-turn history and resets the persistent C++ core;
`/quit` shuts it down. Every prompt is rendered by the verified official Qwen
chat template. The pinned checkpoint is a pretrained Base model rather than an
instruction-tuned chat model, so untrained output is a plumbing demonstration.
The client reserves completion space inside the context window
and evicts only complete oldest user/assistant turns when necessary. Generation
is greedy when `--temperature 0`; sampled requests are explicitly seeded by
`--seed` and retain fixed `--temperature`, `--top-k`, and `--top-p` settings.
The bridge also passes the verified tokenizer's decodable vocabulary size, so
the core cannot emit one of Qwen's padded, non-tokenizer output rows.
Cross-device determinism still depends on the recorded Torch/hardware stack.
The executable and checkpoint are passed as an argument vector, never through
a shell. The wrapper reads the configured `Torch_DIR` and prepends that exact
library directory before launching Python and its persistent C++ child, so an
inherited incompatible LibTorch path cannot silently change the runtime.

Use `--snn` only as an uncalibrated hybrid smoke, or replace the archive options
with `--checkpoint PATH` for a matching trained compact model. Pass
`--qwen3-0.6b` as well when the checkpoint uses the exact Qwen geometry. The
repository does not contain a trained chatbot checkpoint. Interactive text
demonstrates the interface unless it is tied to a retained, evaluated run
artifact.

## Repository layout

- `experiments/`: executable experiment entry points
- `include/` and `src/`: reusable experiment support code
- `tests/`: dependency-free unit and integration tests
- `scripts/`: dataset and experiment utilities
- `docs/`: experiment protocol and result conventions
- `.github/`: CI, contribution templates, and dependency updates

## Reproducibility

Published results should record the experiment commit, `snnbase` commit, compiler, CMake cache
settings, command line, dataset identity, and raw metrics. See
[the experiment protocol](docs/EXPERIMENTS.md).

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change. Report vulnerabilities using
[SECURITY.md](SECURITY.md), not a public issue. This project is available under the
[MIT License](LICENSE).
