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

## Chatbot Phase 0 status

Snapshot date: 2026-09-01

The track is now split explicitly:

1. Exact Qwen ANN import/parity and hybrid SNN conversion. The deterministic
   converter maps the pinned Qwen3-0.6B-Base checkpoint into a versioned dense
   archive; the C++ loader validates provenance, geometry, tensor coverage, and
   hashes. ANN mode bypasses LIF for oracle comparison. SNN mode imports the
   same 310 dense tensors and initializes 112 experiment-owned threshold/leak
   parameters for two LIF sites in each of 28 layers.
2. Compact trainable ANN/SNN controls. These retain the tokenizer and data
   contracts but use a 256-wide, four-layer topology with random
   initialization. Current checkpoint signatures include spiking mode, so a
   compact ANN checkpoint cannot yet be transplanted into compact SNN mode;
   they are independently trainable controls.

Implemented plumbing and reference artifacts include strict UTF-8 conversation
validation, assistant-only labels, the cross-language
`sha256-seed-bucket-v1` split golden, content-hashed prepared shards, portable
SHA-256, the bounded persistent token protocol, a fail-closed run schema,
immutable Qwen pins and reference fixtures, Qwen inventory/conversion/import,
bounded `inspect`, the oracle gate, the LibTorch runner/checkpoint API, the
interactive official-template client, the bounded/token-aware OASST2 tree
converter with normalized-root deduplication and text-free lineage, and offline
contract/smoke tests.

The training driver now flushes at epoch boundaries, considers epoch-zero
validation, selects only on lowest validation loss, reloads the selected
checkpoint in a second core process, validates immutable metadata and restored
training counters, reproduces the selected validation aggregate, and then
evaluates the test shard once. The loader also recomputes every record's
deterministic split instead of trusting a self-consistent manifest. This
orchestration is covered by adversarial fake-core tests and was exercised in a
local ignored three-record CPU SNN wiring run using the real tokenizer and C++
core. That run is not an approved-dataset quality result. The training driver
now validates and consumes all four checked JSON configs, binds their canonical
runner argument lists, and records the consumed config hash in its run summary.
The publication assembler revalidates that binding and emits the separate
schema-checked durable run manifest. The executable reports the configure-time
experiment and library revisions/dirty flags, and publishable assembly requires
the current clean checkouts to match that embedded provenance.

A local ignored development archive was produced from the exact checkpoint:
1,192,143,104 bytes, SHA-256
`333c8be1b3fde5013ee8d1dcb96f23051dd934f8e1f359aff329e7e5cf8ccac8`,
with 310 dense tensors. The real three-case CPU float32 gate passed exact probe
IDs and ordered top-16 IDs with maximum absolute error
`9.5367431640625e-06` and maximum relative error
`8.471997478955767e-06` at `atol=rtol=1e-5`. It took 57.60 seconds and peak RSS
was 3,344,712 KiB. This was a local ignored development gate, not a promoted
durable result. The Release executable used `/usr/bin/c++` and the CPU
PyTorch/LibTorch 2.7.1+cpu installation under the conversion environment, with
its matching Torch `LD_LIBRARY_PATH`; it was not the older 2.3 dependency
matrix.

The pinned OASST2 revision
`179dd21fc55192153d94adb0e0ce8f69e222bf75` is now a selected development
candidate. A full ignored `quality05` conversion validated all 13,854 trees and
135,174 messages, then retained 12,427 exact-Qwen-compatible conversations:
9,990 train, 1,165 validation, and 1,272 test. All paths are at most 512 tokens
without truncation. Exact normalized-root-prompt deduplication removed 96
records; the canonical output SHA-256 is
`64c0fb332808ded357e6702b9639b284a0dd9393fa5014f17d69fa88f719a4e5`.
The raw/converted files remain ignored local artifacts. The conversion profile
is reproducible but not a universal safety filter, and fuzzy/semantic
near-duplicate review is still outstanding.

The same canonical output completed the real preparation path in 60.08 seconds
with 286,428 KiB peak RSS. The prepared dataset manifest SHA-256 is
`88f2298888c0bfb426dc4de84ad1dc89757de380d5773ab7d7d552a9396d416b`;
its 9,990/1,165/1,272 record shards were then reloaded and revalidated by the
training driver. The prepared manifest also hash-binds retained copies of the
conversion manifest and text-free lineage. This is data-contract evidence, not
a training or quality result.

The language backend is committed and published at the clean `snnbase` revision
`c282306ee3b80ccc5123fcb8b2da78ae51ed09fe`. Its decoder and installed-package
consumer tests passed with LibTorch 2.3.0 and 2.5.1, and the experiment
runner/smoke gates passed against the latter. The revision is published on the
`codex/spiking-chatbot` branch, and the workflow now pins it in a CPU
LibTorch 2.5.1 chatbot smoke job. That hosted job is a build/protocol gate, not
model-quality evidence. The existing general non-Torch CI job remains pinned to
`37abded48777968f112fcd5c66d357d0352d9ff2`; the explicit local
configure/smoke scripts provide the corresponding reproducible local path.

A conversation dataset candidate has been selected and mechanically validated,
but it has not received project-specific legal/policy approval and no full
training run has been performed. Therefore no reportable ANN or SNN held-out
assistant perplexity or generation-quality measurement is available. The
completed three-record SNN run is wiring evidence only. No latency, power, or
energy comparison has been made, and the exact hybrid SNN has not been
calibrated or evaluated. These remain result gates rather than implementation
claims. See
[`CHATBOT_EXPERIMENT.md`](CHATBOT_EXPERIMENT.md) and
[`results/CHATBOT_BASELINE.md`](results/CHATBOT_BASELINE.md).

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

## Manifest paths

Legacy non-chatbot experiments can use `scripts/write_run_manifest.sh` for a
basic command/revision/dataset record. Chatbot runs use
`tools/chatbot_publish.py`, whose schema-bound manifest additionally verifies
the resolved configuration, executable build provenance, prepared shards,
selected checkpoint, metrics stream, Qwen assets, and retained environment.
