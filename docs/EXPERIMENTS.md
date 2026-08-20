# Experiment protocol

Layer-by-layer diagrams for every classifier are collected in
[NETWORK_DIAGRAMS.md](NETWORK_DIAGRAMS.md), including both EMNIST modes, the
handcrafted CIFAR feature classifier, and both CIFAR spiking convolutional
networks.

The scaled perception-cognition-action experiment is documented in
[SPAUN_EXPERIMENT.md](SPAUN_EXPERIMENT.md), with its first deterministic result
in [results/SPAUN_BASELINE.md](results/SPAUN_BASELINE.md).

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

## CIFAR-10 baseline

The harness reads the official binary batches and adapts each 32x32 RGB image
to `snnbase::spiking_conv::ImageView`. The default research path uses RGB rate
encoding into `SpikeEvent` payload chunks, two same-padding convolution stages,
`Neuron`-controlled quantized firing rates, SEW-add residual merging,
surrogate-gradient backpropagation, Adam updates, and deterministic one-pixel
translation augmentation.

The test suite uses generated binary records to validate batch parsing,
limits, invalid labels, color/spatial encoding, and the learning path.

## CIFAR-10 temporal residual SNN

`cifar10_deep` adapts the same CIFAR-10 binary batches into
`snnbase::spiking_conv::SampleView` instances with 32x32x3 channel-major image
views. The library model preserves `[time, batch, channel, row, column]`
through every convolution, evolves learned LIF membrane state, applies exact
autograd through temporal batch normalization, and merges residual branches by
SEW-add at each timestep. Spatial downsampling, global pooling, and a population
readout replace the old two-convolution/flattened-head topology.

Training uses a stratified holdout from the training partition; the test set is
evaluated only after training. Result reports should record channel width,
blocks per stage, normalization, timesteps, seed, spike rate, checkpoint use,
epoch count, train/validation/test limits, and hardware. Multiple-seed reports
must include every individual result as well as mean and standard deviation.
