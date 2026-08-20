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
| `snnbaseExperiments` | `codex/repro-v0.1` | `142d0f4` | Clean |
| `snnbase` | `codex/repro-v0.1-stabilize` | `626babc` | Clean |

These revisions are local only. They have not been pushed or merged.

## Gates

- [x] Record both repository SHAs and dirty state.
- [x] Preserve tracked and untracked worktree changes in a recoverable snapshot.
- [x] Make the library test suite pass in exact Debug and Release configurations.
- [ ] Replace transitional assert-based test enforcement with a dedicated test framework.
- [ ] Publish an immutable library revision consumed by experiments.
- [x] Add clean-checkout experiment CI configuration and run manifests.
- [ ] Reproduce temporal results from clean tags with five seeds.
- [ ] Add a true event-stream benchmark.
- [ ] Integrate learned perception into the Spaun-inspired loop.

## Validation completed

- `snnbase`: exact 128-bit-event/256-history Debug and Release builds pass all
  7 CTest targets; the installed CMake package also builds a separate
  `find_package(snnbase CONFIG)` consumer.
- `snnbase`: sparse long-delay scheduling, inference-only neurons, learning
  metrics, and normalized spiking-convolution gradient tests are covered.
- `snnbaseExperiments`: a clean pair against library revision
  `626babc` builds 48 targets and passes all
  5 `unit|smoke` tests.
- `snnbase` temporal backend: with the compatible local toolchain below, the
  library builds 8 targets and passes all 8 tests; the temporal experiments
  pair builds 57 targets and passes all 7 available `unit|smoke` tests.
- Dataset download scripts validate archive/content checksums where an
  authoritative checksum is available and fail safely on corrupt downloads.
- Sweep, confirmation, architecture, and Spaun-memory scripts emit a JSON run
  manifest before each promoted command.

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
runtime. CI should promote this recipe into a pinned container/toolchain file
before temporal five-seed results are treated as release evidence.

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
