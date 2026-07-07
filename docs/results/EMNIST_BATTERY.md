# EMNIST battery

Date: 2026-07-06

The battery uses all six standard EMNIST splits from the official NIST binary
archive. Training and testing use the complete published partitions.

- Source: <https://www.nist.gov/itl/products-and-services/emnist-dataset>
- Downloaded archive SHA-256:
  `fb9bb67e33772a9cc0b895e4ecf36d2cf35be8b709693c3564cea2a019fcda8e`

## Configuration

- Event width: 128 bits
- Payload width: 96 bits
- Prototype queue depth: 256 per class
- Pixel threshold: 1.30
- Novelty threshold: 0.78
- Reward learning rate: 0.35
- Epochs: 1
- Decision method: weighted bitwise matching
- Build: Release, GCC 13.3.0

## Results

| Split | Classes | Train samples | Test samples | Train time | Test time | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| ByClass | 62 | 697,932 | 116,323 | 11.779 s | 42.843 s | 59.80% |
| ByMerge | 47 | 697,932 | 116,323 | 12.102 s | 32.911 s | 65.21% |
| Balanced | 47 | 112,800 | 18,800 | 1.876 s | 5.065 s | 55.56% |
| Letters | 26 | 124,800 | 20,800 | 2.136 s | 3.258 s | 67.46% |
| Digits | 10 | 240,000 | 40,000 | 4.183 s | 2.637 s | 90.22% |
| EMNIST-MNIST | 10 | 60,000 | 10,000 | 0.994 s | 0.656 s | 88.94% |
| **Total** | — | **1,933,464** | **322,246** | **33.070 s** | **87.370 s** | — |

Total process wall time, including data loading and allocation, was 122.42
seconds. Maximum resident memory was 673,152 KiB. The unweighted mean of the
six split accuracies is 71.20%; this is informational only because the splits
overlap and represent different classification tasks.

## Regression tests

When the local EMNIST files are present, CMake registers one accuracy test per
split:

```sh
ctest --test-dir build-release -L emnist --output-on-failure
```

The regression floors are deliberately below the measured values to tolerate
small platform differences while detecting material changes.

The default regression battery now uses the improved structured network.
See [EMNIST_STRUCTURED.md](EMNIST_STRUCTURED.md) for its architecture,
accuracy comparison, and timing.
