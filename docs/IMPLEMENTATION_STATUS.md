# Reproducibility and implementation status

This file tracks the staged implementation of the snnbase/snnbaseExperiments
roadmap. It is intentionally kept separate from experiment result reports.

## Phase 0 snapshot

Snapshot date: 2026-08-20

| Repository | Branch | Committed revision | Working tree |
|---|---|---|---|
| `snnbaseExperiments` | `main` | `7288011` | Modified tracked files and untracked CIFAR/Spaun sources, tests, scripts, and reports |
| `snnbase` | `codex/prototype-memory-learning` | `aa80bd8` | Modified tracked files and untracked population, semantic, reward, and temporal sources |

The complete Phase 0 recovery snapshot is stored outside the repositories at:

`/tmp/snnbase-phase0.AQyH8C`

It contains status manifests, binary tracked diffs, and archives of untracked
files for both repositories. No source or data files were deleted or reset.

The implementation is now being developed on local feature branches:

| Repository | Branch | Current revision | Working tree |
|---|---|---|---|
| `snnbaseExperiments` | `codex/repro-v0.1` | published feature branch / PR #2 | Clean |
| `snnbase` | `codex/repro-v0.1-stabilize` | published feature branch `37abded` | Clean |

These revisions have been pushed but have not been merged to their default
branches.

## Gates

- [x] Record both repository SHAs and dirty state.
- [x] Preserve tracked and untracked worktree changes in a recoverable snapshot.
- [x] Make the library test suite pass in exact Debug and Release configurations.
- [x] Replace transitional assert-based test enforcement with an in-tree test framework.
- [x] Publish an immutable library revision consumed by experiments.
- [x] Add clean-checkout experiment CI configuration and run manifests.
- [x] Reproduce a full temporal result with five seeds from clean published revisions.
- [x] Add a native N-MNIST event-stream benchmark harness and parser tests.
- [x] Run and publish an artifact-backed N-MNIST benchmark on the real dataset.
- [x] Add checkpoint-level learned digit perception to the Spaun task loop.
- [x] Validate learned perception end-to-end on held-out raster/handwriting inputs.

## Validation completed

- `snnbase`: exact 128-bit-event/256-history Debug and Release builds pass all
  7 CTest targets; the installed CMake package also builds a separate
  `find_package(snnbase CONFIG)` consumer.
- The library test suite now uses test-only requirements that throw detailed
  failures independently of `NDEBUG`; Release validation no longer relies on
  forcing assertions back on through compiler flags.
- `snnbase`: sparse long-delay scheduling, inference-only neurons, learning
  metrics, and normalized spiking-convolution gradient tests are covered.
- `snnbaseExperiments`: a clean pair against the pinned library revision
  `37abded` builds 53 targets and passes all 6 `unit|smoke` tests, including
  the native N-MNIST parser and CIFAR-10 dataset smoke coverage.
- `snnbase` temporal backend: with the compatible local toolchain below, the
  library builds 8 targets and passes all 8 tests; the temporal experiments
  pair builds 57 targets and passes all 7 available `unit|smoke` tests.
- The Torch-enabled CMake target now propagates LibTorch's C++ ABI setting to
  the base and downstream targets, preventing static-link ABI failures in
  temporal experiment executables.
- Dataset download scripts validate archive/content checksums where an
  authoritative checksum is available and fail safely on corrupt downloads.
- Sweep, confirmation, architecture, and Spaun-memory scripts emit a JSON run
  manifest before each promoted command.
- `run_cifar10_temporal_seeds.sh` writes a manifest, log, best-validation
  checkpoint, final checkpoint, and checksums for every seed, then calculates
  the mean and sample standard deviation from the five test accuracies. A
  two-seed, 20-example CPU smoke execution verified the complete artifact path;
  it is not a reportable accuracy result.
- The full CIFAR-10 temporal SEW-ResNet protocol completed five 100-epoch CUDA
  runs: 90.582% +/- 0.120% test accuracy. Per-seed selected checkpoints,
  final checkpoints, manifests, logs, hashes, configuration, dataset digests,
  and the aggregate are recorded in
  `results/cifar10/temporal-sew-resnet-five-seed-2026-08-20/`; the complete
  large artifact directory remains locally retained and ignored.
- `nmnist_experiment` reads timestamped, polarity-coded N-MNIST recordings,
  encodes them into temporal event bins, and is covered by generated native
  binary-record tests. The complete official 60,000/10,000 split baseline
  (one epoch, ten bins) achieved 33.91%; its source revisions, archive hashes,
  command, and output fields are retained in
  `results/nmnist/full-baseline-2026-08-20/`. This is an event-stream baseline,
  not a competitive N-MNIST claim.
- The learned Spaun A1 checkpoint reached 98.36% on the held-out 10,000-image
  MNIST test split and correctly produced `7` for the task-loop stimulus
  `A1[7]?` after aspect-preserving centered glyph rasterization. Revisions,
  checkpoint digest, dataset checksums, and both gate outputs are retained in
  `results/spaun/learned-a1-2026-08-20/`.

## Temporal toolchain gate

The installed PyTorch 2.5.1+cu121 CMake package requires a matching CUDA 12.1
compiler and an older host compiler. The validated local recipe is:

```text
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
CMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc
CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
CUDAToolkit_ROOT=/usr/local/cuda-12.1
LD_LIBRARY_PATH unset when running tests and binaries
```

Using the default `/usr/bin/nvcc` (CUDA 12.0), the CUDA 12.9 symlink, GCC 13,
or an inherited `/opt/libtorch` library path is not supported: those combinations
either fail compiler identification or load an ABI-incompatible LibTorch at
runtime. A pinned recipe now exists at `containers/temporal.Dockerfile` with
the matching `cmake/toolchains/linux-cuda121-gcc11.cmake`; building and
exercising it in CI remains required before temporal five-seed results are
treated as release evidence. The digest-pinned image was built locally and
passed the `snnbase_temporal_tests` CTest target without GPU access.
The checked-in GitHub Actions workflow also defines a CPU-only
`temporal-container-test` job that repeats this build and CTest gate once the
pinned library revision is published.

## Current constraints

- Library edits require elevated workspace access to
  `/home/dean/repos/snnbase`; that access has now been granted for this run.
- Existing user changes must remain intact and must not be broadly staged or
  discarded.

The initial manifest writer is available at
`scripts/write_run_manifest.sh`. It records both repository revisions and dirty
flags, toolchain versions, the command, and SHA-256 hashes for supplied dataset
directories. Promoted runs must still add resolved configuration, seed, metrics,
checkpoint, and hardware fields.
