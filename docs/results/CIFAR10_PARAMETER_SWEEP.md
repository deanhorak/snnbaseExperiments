# CIFAR-10 temporal SNN parameter sweep

## Protocol

The screening objective is validation accuracy, not test accuracy. Every
screening run used the first 10,000 CIFAR-10 training records, split
stratified into 9,000 training and 1,000 validation examples. All cases used
seed 42, eight epochs, two warmup epochs, augmentation, and cosine learning
rate decay. The official CIFAR-10 test partition was not loaded.

Unless a row says otherwise, the configuration was width 32, two blocks per
stage, four timesteps, direct-current encoding, threshold 1.0, leak 0.9,
surrogate slope 4, population 2, batch 128, learning rate 0.001, weight decay
0.0005, label smoothing 0.1, and Cutout 8.

## Architecture and neuron-dynamics screen

| Case | Best validation | Delta from baseline |
|---|---:|---:|
| Baseline | 64.74% | — |
| Threshold 0.75 | 62.94% | -1.80 pp |
| Threshold 1.25 | 64.64% | -0.10 pp |
| Leak 0.80 | 63.84% | -0.90 pp |
| Leak 0.95 | 63.64% | -1.10 pp |
| Surrogate slope 2 | 62.14% | -2.60 pp |
| Surrogate slope 8 | 61.34% | -3.40 pp |
| Deterministic rate encoding | 46.75% | -17.99 pp |
| Readout population 4 | 63.14% | -1.60 pp |
| Three blocks per stage | 63.74% | -1.00 pp |
| Width 48 | **67.83%** | **+3.09 pp** |
| Six timesteps | 64.74% | 0.00 pp |

Width was the only material improvement. Extra timesteps matched baseline
accuracy while increasing runtime from 198 to 314 seconds. Extra depth did
not improve early optimization at the same epoch budget, and the larger
readout did not improve feature quality.

## Width-48 optimizer and regularization screen

These cases used width 48 and batch 64. The width-48 baseline scored 67.83%.

| Case | Best validation | Delta from width-48 baseline |
|---|---:|---:|
| Learning rate 0.0005 | 65.93% | -1.90 pp |
| Learning rate 0.002 | 66.43% | -1.40 pp |
| Weight decay 0.0001 | 67.73% | -0.10 pp |
| Weight decay 0.001 | 67.23% | -0.60 pp |
| Label smoothing 0.05 | 66.23% | -1.60 pp |
| Label smoothing 0 | 66.63% | -1.20 pp |
| Cutout 0 | **68.53%** | **+0.70 pp** |
| Cutout 16 | 65.13% | -2.70 pp |

The original learning rate, weight decay, label smoothing, and neuron
dynamics remain selected. Width 48 is the clear capacity improvement. Cutout
0 and Cutout 8 advance to longer paired confirmation because disabling
regularization can improve a short proxy without improving a full schedule.

## Longer confirmation

The confirmation uses 25,000 training-partition records, a stratified
22,500/2,500 train/validation split, 15 epochs, and three warmup epochs.

| Case | Best validation | Best epoch | Runtime |
|---|---:|---:|---:|
| Width 48, Cutout 0, legacy black padding | **83.45%** | 15 | 1,493 s |
| Width 48, Cutout 8, zero-centered padding | 83.09% | 15 | 1,494 s |

These are not a clean Cutout pair. The executable was rebuilt with corrected
mean padding while the first process was already running, so its in-memory
model retained black padding and the second launch used zero-centered padding.
The 0.36-point difference is consequently confounded by Cutout and padding and
must not be attributed to either one. Cutout 0 remains the provisional choice
because it won the controlled short screen; the architecture screen below is
designed to isolate padding.

The no-Cutout curve was still rising at the end: validation accuracy improved
from 79.29% at epoch 10 to 83.45% at epoch 15. Its final training accuracy was
88.21%, and cosine decay had already reduced the learning rate to 2.69e-5.
This supports a longer final schedule, but it does not support extrapolating
to 95% from schedule length alone.

Even before the paired case completes, 83.45% is far enough below the 95%
goal that parameter tuning alone is not an adequate plan. The next controlled
screen therefore keeps the spiking SEW residual substrate and tests two
architecture/data-path corrections: zero-centered crop padding and independent
batch-normalization statistics at every simulation timestep (BNTT).

The BNTT implementation passed a one-epoch CPU forward/backward/validation
smoke run and a checkpoint save/reload/evaluation smoke run. Those checks prove
the path is executable and serializable; they are not accuracy measurements.

<!-- Paired confirmation, architectural screen, and final full-data results are
appended after completion. -->
