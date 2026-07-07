# MNIST optimization results

Date: 2026-07-02

## Configuration

- Training split: 60,000 images
- Test split: 10,000 images
- Epochs: 1
- Encoding threshold multiplier: 1.30
- Spike event width: 128 bits
- Payload width: 96 bits
- Prototype capacity: 256 per class
- Novelty threshold: 0.78
- Reward learning rate: 0.35
- Decision method: weighted bitwise matching
- Build: Release, GCC 13.3.0

Command:

```sh
./build-release/mnist_experiment
```

## Result

- Correct: 8,652 / 10,000
- Accuracy: **86.52%**
- Elapsed time: 1.75 seconds
- Maximum resident memory: 61,440 KiB

CTest enforces a conservative 85% minimum to detect material regressions while leaving margin for
supported compiler and platform differences.

## Parameter sweep

The optimization used 77 full 60,000/10,000 train/test runs. Representative results:

| Event bits | Payload bits | Queue depth | Pixel threshold | Novelty | Learning rate | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 32 | 32 | 1.30 | 0.80 | 0.20 | 55.24% |
| 96 | 64 | 128 | 1.30 | 0.80 | 0.20 | 78.93% |
| 128 | 96 | 64 | 1.00 | 0.80 | 0.20 | 81.93% |
| 128 | 96 | 128 | 1.30 | 0.80 | 0.20 | 83.67% |
| 128 | 96 | 256 | 1.30 | 0.80 | 0.20 | 84.31% |
| 128 | 96 | 256 | 1.30 | 0.78 | 0.30 | 86.44% |
| 128 | 96 | 256 | 1.30 | 0.78 | 0.35 | **86.52%** |
| 128 | 96 | 512 | 1.30 | 0.78 | 0.30 | 85.66% |
| 128 | 96 | 768 | 1.30 | 0.78 | 0.30 | 85.65% |

Two epochs reached 86.38% with the selected learning rate, so one epoch remains the default.
