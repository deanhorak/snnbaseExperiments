# Temporal SNN architecture upgrade

## Environment

- Date: 2026-07-13
- Experiments base revision: `7288011e66d7964f6e3e870ea46e59d36bc21523`
  plus the changes in this worktree
- `snnbase` base revision: `aa80bd8129603d34f3c6f265063a689340fd30b5`
  plus the changes in its worktree
- Compiler: GCC 13.3.0; CMake 3.31.4; LibTorch 2.3.0
- CPU: Intel Xeon E5-2695 v2, 24 logical CPUs
- GPU: NVIDIA GeForce RTX 3050 8 GB
- Spike events: 128 bits, including a 96-bit payload

The existing dirty worktrees were preserved; the base revisions above do not
by themselves identify the uncommitted implementation.

## Architectural changes

- Added a library-owned temporal residual SNN that keeps explicit timestep
  tensors through all layers.
- Added learned per-channel LIF thresholds and leaks with hard forward spikes
  and sigmoid surrogate gradients.
- Replaced the two-convolution, flattened dense-head topology with a spiking
  stem, three downsampling SEW-add residual stages, global spatial pooling, and
  a population readout.
- Replaced hand-written normalization gradients on the accuracy path with
  LibTorch autograd and temporal batch normalization.
- Added AdamW, warmup plus cosine decay, label smoothing, crop/flip/Cutout,
  gradient clipping, spike-rate regularization, deterministic per-epoch seeds,
  and a stratified validation holdout.
- Checkpoints now contain model, optimizer, and completed-epoch state and can
  be resumed without restarting the data schedule.
- Added weighted excitatory/inhibitory synapses and timestep-synchronous LIF
  integration to the lightweight event network.
- Changed full structured-EMNIST experts from frozen memories to bounded,
  mistake-driven positive adaptation with hard-negative reinforcement.

## Full prototype experiments

Commands:

```sh
./build/emnist_battery --data-dir data/emnist --architecture structured
./build/mnist_experiment --data-dir data/mnist
```

All runs used the complete official train and test partitions.

| Experiment | Previous | New | Change |
|---|---:|---:|---:|
| EMNIST ByClass | 59.80% | 67.63% | +7.83 pp |
| EMNIST ByMerge | 65.21% | 71.82% | +6.61 pp |
| EMNIST Balanced | 55.56% | 67.38% | +11.82 pp |
| EMNIST Letters | 67.46% | 74.19% | +6.73 pp |
| EMNIST Digits | 90.22% | 92.87% | +2.65 pp |
| EMNIST-MNIST | 88.94% | 92.64% | +3.70 pp |
| MNIST single-event baseline | 86.52% | 86.52% | 0.00 pp |

The new structured-EMNIST regression floors are 66%, 70%, 66%, 73%, 92%,
and 92%, respectively.

## Temporal convolution experiments

The CIFAR-10 and deep-EMNIST results below use a 10% stratified validation
holdout from the training partition. The official test partition is evaluated
only after training.

### CIFAR-10

```sh
./build/cifar10_deep --data-dir data/cifar-10-batches-bin \
  --epochs 20 --batch-size 128 --width 32 --blocks 2 --time-steps 4 \
  --device cuda --seed 42
```

- Parameters: 700,148
- Training/validation/test samples: 45,000 / 5,000 / 10,000
- Final train accuracy: 87.58%
- Final validation accuracy: 86.18%
- Test accuracy: **86.12%** (8,612/10,000)
- Test loss: 0.44413
- Training/test spike rate: 0.1610 / 0.1558
- Training/test time: 2,461.4 s / 9.0 s
- Prior two-convolution baseline: 51.08%
- Improvement: **+35.04 percentage points**

The old baseline was rerun with its documented three-epoch command and exactly
reproduced 51.08% (5,108/10,000), with 1,253.7 s training time. This comparison
therefore uses results from the same machine and current worktrees rather than
only copying the historical report.

The hardest remaining pair is cat/dog: 105 cats were predicted as dogs and
107 dogs as cats. Automobile, ship, and truck recall were 95.1%, 93.2%, and
91.9%, respectively.

### Deep EMNIST-MNIST

```sh
./build/emnist_deep --data-dir data/emnist --split mnist \
  --epochs 5 --batch-size 128 --time-steps 4 --device cuda --seed 42
```

- Parameters: 176,740
- Training/validation/test samples: 54,000 / 6,000 / 10,000
- Final validation accuracy: 99.47%
- Test accuracy: **99.49%** (9,949/10,000)
- Test loss: 0.06266
- Training/test spike rate: 0.2015 / 0.2026
- Training/test time: 313.0 s / 3.5 s
- Previously documented deep result: 99.05% after 30 epochs
- Improvement: **+0.44 percentage points in one sixth as many epochs**

### Deep EMNIST Digits

```sh
./build/emnist_deep --data-dir data/emnist --split digits \
  --epochs 3 --batch-size 128 --time-steps 4 --device cuda --seed 42
```

- Parameters: 176,740
- Training/validation/test samples: 216,000 / 24,000 / 40,000
- Final validation accuracy: 99.55%
- Test accuracy: **99.55%** (39,821/40,000)
- Test loss: 0.05839
- Training/test spike rate: 0.1962 / 0.1899
- Training/test time: 747.7 s / 13.8 s
- Previously documented deep result: 99.00%
- Improvement: **+0.55 percentage points**

## Verification

- `snnbase` core, legacy convolution, and temporal tests: 3/3 passed.
- Experiment unit and smoke tests in temporal mode: 7/7 passed.
- Experiment core-only unit and smoke tests: 5/5 passed.
- Resumable checkpoint round-trip and continued-epoch test passed.
