# Structured EMNIST network

Date: 2026-07-06

## Architecture

The structured network implements six changes to the original classifier:

1. Four 14×14 spatial tile channels instead of one pooled whole-image event.
2. Two intensity planes in each 96-bit tile event.
3. Sampled hard-negative reinforcement against the strongest incorrect class.
4. Four competitively selected expert neurons per class.
5. A hidden patch-feature channel built from `snnbase::Neuron` detectors for
   horizontal, vertical, diagonal, corner, and enclosure patterns.
6. Bounding-box centering and distributed ±1-pixel training augmentation.

Each expert bank retains at most 64 prototypes. Scores from the four spatial
channels and the lower-weight patch channel are normalized and aggregated
before class selection.

## Accuracy comparison

| Split | Baseline | Structured | Change |
|---|---:|---:|---:|
| ByClass | 59.80% | **65.08%** | +5.28 points |
| ByMerge | 65.21% | **70.24%** | +5.03 points |
| Balanced | 55.56% | **69.09%** | +13.53 points |
| Letters | 67.46% | **75.06%** | +7.60 points |
| Digits | 90.22% | **93.00%** | +2.78 points |
| EMNIST-MNIST | 88.94% | **92.53%** | +3.59 points |
| **Unweighted mean** | **71.20%** | **77.50%** | **+6.30 points** |

## Timing

| Split | Train samples | Test samples | Train time | Test time |
|---|---:|---:|---:|---:|
| ByClass | 697,932 | 116,323 | 116.273 s | 217.752 s |
| ByMerge | 697,932 | 116,323 | 97.991 s | 163.301 s |
| Balanced | 112,800 | 18,800 | 15.117 s | 26.165 s |
| Letters | 124,800 | 20,800 | 12.829 s | 16.474 s |
| Digits | 240,000 | 40,000 | 16.818 s | 12.388 s |
| EMNIST-MNIST | 60,000 | 10,000 | 4.215 s | 3.101 s |
| **Total** | **1,933,464** | **322,246** | **263.243 s** | **439.181 s** |

The measured training and testing phases total 702.424 seconds. The two
largest splits were run sequentially in 596.82 seconds wall time with a peak
resident set of 687,936 KiB.

The accuracy gain comes with substantially higher inference cost: each class
now evaluates four experts across five channels. A subsequent optimization
should cache encoded test features and investigate expert pruning or a
coarse-to-fine class shortlist.
