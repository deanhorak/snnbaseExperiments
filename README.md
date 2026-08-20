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

## Prerequisites

- CMake 3.20 or newer
- A C++20 compiler
- Ninja (when using the provided presets)
- LibTorch 2.3 or newer, with CUDA support for GPU training
- A sibling checkout at `../snnbase`, or an explicit `SNNBASE_SOURCE_DIR`
- `curl`, `gzip`, and `unzip` to use the dataset download helpers

## Build and test

```sh
git clone https://github.com/deanhorak/snnbase.git ../snnbase
cmake --preset default
cmake --build --preset default
ctest --preset default
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
built in CI; it is the required environment for the five-seed release run.

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
