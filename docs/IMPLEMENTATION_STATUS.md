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
| `snnbase` | `codex/repro-v0.1-stabilize` | `ae7d8a7` | Clean |

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
  `ae7d8a79bdc2747965d0b82fbac9a6f371d7f68d` builds 48 targets and passes all
  5 `unit|smoke` tests.
- Dataset download scripts validate archive/content checksums where an
  authoritative checksum is available and fail safely on corrupt downloads.
- Sweep, confirmation, architecture, and Spaun-memory scripts emit a JSON run
  manifest before each promoted command.

## Current validation blockers

The temporal backend has not yet been promoted to a reproducible gate. On the
current host, configuring `snnbase` with the installed PyTorch 2.5.1+cu121
CMake package invokes `/usr/bin/nvcc` (CUDA 12.0) against the active CUDA 12.9
headers and GCC 13.3. CMake's CUDA compiler-identification test fails with
`_Float32`/`_Float64` declarations from the system headers. This must be
resolved by a pinned compatible LibTorch/CUDA/GCC container or toolchain file;
the temporal option therefore remains opt-in in CI.

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
