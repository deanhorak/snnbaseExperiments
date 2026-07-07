# Experiment protocol

An experiment is complete when another developer can reproduce its inputs, configuration, and
metrics from documented commands.

Each result should record:

1. Experiment repository commit and `snnbase` commit.
2. Compiler, standard library, CMake version, and relevant hardware.
3. Dataset source, version, split, preprocessing, and sample limits.
4. All non-default CMake and command-line options.
5. Primary metrics, confusion data where applicable, and observed failures.

Large datasets, generated binaries, and build trees do not belong in Git. Store small deterministic
test fixtures in `tests/` only when their license permits redistribution.

## MNIST baseline

The harness pools pixels into the compile-time `SpikeEvent` payload grid. A cell emits a payload bit
when its mean intensity is nonzero and at least the image-wide mean multiplied by `--threshold`.
Training reinforces the neuron matching the known label. Prediction selects the highest prototype
similarity among ten class neurons.

The test suite uses generated 2×2 IDX fixtures to validate file parsing and the learning path.
When the MNIST files are installed, CMake also registers a full-dataset `mnist_accuracy` test with
an 85% regression floor. The optimized baseline measured 86.52% accuracy.
