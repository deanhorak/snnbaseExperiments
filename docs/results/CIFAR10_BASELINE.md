# CIFAR-10 baseline

## Run

- Date: 2026-07-10
- Experiment commit: `7288011` plus local CIFAR-10 harness changes
- `snnbase` commit: `aa80bd8` plus local `spiking_conv` updates
- Compiler: `c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`
- CMake: 3.31.4
- Build: `cmake --preset release && cmake --build --preset release`
- Dataset: CIFAR-10 binary batches, 50,000 train images and 10,000 test images
- Command: `./build-release/cifar10_experiment --epochs 3 --channels 8 --batch-size 32 --time-steps 4 --residual sew-add --no-normalize --save-checkpoint build-release/cifar10-width8-sewadd-e3.ckpt`
- Architecture: `snnbase::spiking_conv::Classifier`

## Configuration

- Spike event bits: 128
- Payload bits: 96
- Prototype queue depth: 256
- Input: 32x32x3 RGB
- Convolution width: 8 channels
- Kernels: 3x3, stride 1, same padding
- Residual: SEW-add from first to second convolution stage
- Channel normalization: disabled for the best measured run
- Epochs: 3
- Batch size: 32
- Timesteps: 4
- Learning rate: 0.001
- Neuron threshold: 0.5
- Parameters: 82,738
- `snnbase::Neuron` units: 16,394

## Result

```text
epoch=1 loss=1.68332 train_accuracy=40.29% seconds=418.360
epoch=2 loss=1.46201 train_accuracy=48.29% seconds=419.084
epoch=3 loss=1.41867 train_accuracy=50.07% seconds=421.938
correct=5108/10000
accuracy=51.08%
test_loss=1.37107
load_seconds=0.460
train_seconds=1259.382
test_seconds=76.620
```

Confusion matrix, rows are actual classes and columns are predicted classes:

```text
       0    1    2    3    4    5    6    7    8    9
0    591   56   77   15   24    5   12   23  149   48
1     37  724   13    9    7    4   10   12   60  124
2     86   16  498   50   96   68   64   58   40   24
3     29   44  163  311  106  126   94   71   19   37
4     50   21  246   56  360   44   83  116   16    8
5     17   26  172  164   90  330   62  100   25   14
6     12   21  133   95   90   37  534   44    8   26
7     27   17   70   70   86   66   25  578   15   46
8    159   96   23   12    2   11    8    4  653   32
9     50  235   20   21   11   10   28   29   67  529
```

This run improves over the previous retinal spike-feature baseline, which
measured 41.21%, and over the original single-event prototype baseline, which
measured 18.74%. It does not reach the requested 95%+ range. The current
updated `snnbase` path adds RGB input, same-padding convolution, residual
merging, surrogate-gradient training, normalization controls, and checkpointing,
but it remains a two-convolution CPU model. The measured trajectory suggests
that 95%+ CIFAR-10 still requires a deeper residual stack or a WideResNet-class
spiking architecture in `snnbase`, not just longer training of this topology.
