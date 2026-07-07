# Deep convolutional snnbase EMNIST results

Date: 2026-07-07

## Substrate

The classifier is implemented by `snnbase::spiking_conv::Classifier`:

- Input rate trains are packed into `snnbase::SpikeEvent` payloads.
- Shared convolutional kernels, biases, and dense weights are owned by the
  library model.
- Every convolutional location and class output is a `snnbase::Neuron`; its
  threshold controls quantized firing-rate generation.
- Forward activations are eight-timestep spike rates, not unconstrained ReLU
  activations.

Rate encoding, convolutional weight sharing, surrogate-gradient
backpropagation, and Adam are implemented in `snnbase`. The experiment layer
adapts EMNIST images and labels to non-owning library sample views.

## Network

```text
28x28 pixels
  -> 8-step snnbase SpikeEvent rate encoding
  -> 12 learned 5x5 convolution filters, stride 2
  -> quantized snnbase neuron spike rates (12x12x12)
  -> 24 learned 3x3 convolution filters, stride 2
  -> quantized snnbase neuron spike rates (24x5x5)
  -> class population readout and softmax
```

Training uses straight-through surrogate derivatives, cross-entropy, Adam,
mini-batches, and distributed ±1-pixel translation augmentation.

## Results

| Split | Prototype baseline | Structured prototype | Deep SNN | Epochs | Train time | Test time |
|---|---:|---:|---:|---:|---:|---:|
| EMNIST-MNIST | 88.94% | 92.53% | **99.05%** | 30 | 1286.612 s | 3.215 s |
| Digits | 90.22% | 93.00% | **99.00%** | 3 | 283.188 s | 9.807 s |
| Letters | 67.46% | 75.06% | **90.63%** | 5 | 283.045 s | 5.673 s |
| Balanced | 55.56% | 69.09% | **82.62%** | 3 | 176.679 s | 5.564 s |

The 10-class models contain 8,938 trainable `snnbase::Synapse` parameters and
2,338 `snnbase::Neuron` units. Letters uses 18,554 synapses; Balanced uses
31,175.

## EMNIST-MNIST epoch sweep

The EMNIST-MNIST run was repeated from scratch with increasing epoch counts to
measure the current model's training asymptote. The executable is deterministic
for this configuration, but it does not checkpoint/resume, so every row below
is a fresh train from epoch 1 through the requested epoch count.

| Epochs | Test accuracy | Test correct | Train time | Test loss | Gain vs prior |
|---:|---:|---:|---:|---:|---:|
| 5 | 98.44% | 9844/10000 | 214.224 s | 0.05220 | — |
| 10 | 98.72% | 9872/10000 | 427.663 s | 0.04186 | +0.28 pp |
| 15 | 98.74% | 9874/10000 | 642.077 s | 0.03951 | +0.02 pp |
| 20 | 98.91% | 9891/10000 | 854.427 s | 0.03299 | +0.17 pp |
| 25 | 99.02% | 9902/10000 | 1070.754 s | 0.03261 | +0.11 pp |
| 30 | 99.05% | 9905/10000 | 1286.612 s | 0.03041 | +0.03 pp |

The 25-to-30 epoch gain was only 0.03 percentage points, so this configuration
appears to asymptote around 99.0-99.1% on EMNIST-MNIST.

ByClass and ByMerge were not rerun with the deep model in this iteration:
their much larger partitions and output populations make them substantially
more expensive, and 98% is not a meaningful expectation for case-sensitive
glyphs that can be visually indistinguishable.
