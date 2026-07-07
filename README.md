# snnbase Experiments

Reproducible C++20 experiments built on
[`snnbase`](https://github.com/deanhorak/snnbase). The repository keeps experimental code,
configuration, validation, and result-reporting conventions separate from the core library.

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

`emnist_deep` adapts EMNIST splits to the library-owned
`snnbase::spiking_conv::Classifier`. Rate encoding, convolution, neuron
thresholds, surrogate-gradient training, and Adam now reside in `snnbase`.
The experiment retains only dataset loading, command-line orchestration, and
reporting. It reaches 99.05% on EMNIST-MNIST after a 30-epoch asymptote
sweep, and 99.00% on EMNIST Digits.

## Prerequisites

- CMake 3.20 or newer
- A C++20 compiler
- Ninja (when using the provided presets)
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
cmake -S . -B build -G Ninja -DSNNBASE_SOURCE_DIR=/path/to/snnbase
cmake --build build
ctest --test-dir build --output-on-failure
```

The project configures `snnbase` with 128-bit spike events, 96-bit payloads, 256 prototypes per
class, and weighted bitwise matching.

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
