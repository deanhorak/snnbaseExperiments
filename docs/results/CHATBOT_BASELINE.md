# Chatbot Phase 0 baseline status

Snapshot date: 2026-09-01

There is still no reportable chatbot training or held-out quality baseline.
There is now a successful local exact-import oracle gate, but its report and
build are ignored development artifacts rather than a promoted durable result.
This distinction prevents import parity from being mistaken for conversational
quality.

## Two-track result table

| Track and variant | Implementation status | Evidence available | Quality/efficiency status |
|---|---|---|---|
| Exact Qwen ANN import | Converter, 310-tensor loader, ANN bypass, bounded `inspect`, and oracle gate implemented | Local ignored three-case CPU float32 oracle passed exact probe IDs and ordered top-16 IDs; max abs `9.5367431640625e-06`, max rel `8.471997478955767e-06`, `atol=rtol=1e-5` | No promoted gate artifact, training, held-out quality, or efficiency result |
| Exact-dense hybrid SNN | Same Qwen dense archive can initialize SNN mode; 112 runtime LIF threshold/leak parameters are outside the Qwen checkpoint | Mapping/load contract and tests exist; ANN oracle is a prerequisite only | Not calibrated, trained, or evaluated; no parity or energy claim |
| Compact ANN | Prepare/train/validation-selection/reload/test plumbing implemented | Fixture/fake-core tests, tiny runner smoke, and an ignored pinned-OASST2 conversion | Development candidate only; no full run or held-out quality result |
| Compact hybrid SNN | Independently trainable compact SNN path implemented | Fixture/fake-core tests, tiny runner smoke, and the same candidate-data contract | Development candidate only; no full run; ANN-to-SNN checkpoint transplant not implemented; quality unavailable |

A local ignored three-conversation CPU SNN wiring run also completed the real
prepare, train, partial-accumulation flush, validation selection, fresh-process
reload, exact validation reproduction, and single test-evaluation path. Its
one-epoch validation loss changed from `15.5243883` to `14.6214914`; the
reloaded value was exactly `14.6214914`. This deliberately tiny synthetic-data
run is implementation evidence only, not a meaningful perplexity or chatbot
quality measurement.

The exact archive is 1,192,143,104 bytes with SHA-256
`333c8be1b3fde5013ee8d1dcb96f23051dd934f8e1f359aff329e7e5cf8ccac8`.
It contains 1,192,099,840 bytes across 310 BF16 dense tensors. The source
`model.safetensors` SHA-256 is
`cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`.

## Local candidate-data evidence

The ignored OASST2 development conversion used source revision
`179dd21fc55192153d94adb0e0ce8f69e222bf75`, compressed SHA-256
`7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b`,
and the declared Apache-2.0 license. The exact Qwen-token-aware `quality05`
profile emitted 12,427 records: 9,990 train, 1,165 validation, and 1,272 test.
All selected paths fit the 512-token limit without truncation. Normalized-root
deduplication removed 96 otherwise eligible records; fuzzy and semantic
near-duplicates have not yet been excluded.

The canonical conversations SHA-256 is
`64c0fb332808ded357e6702b9639b284a0dd9393fa5014f17d69fa88f719a4e5`;
the text-free lineage SHA-256 is
`2142c4e246075a3c3eab28f217cd693e14e69c3bb21d29ef95c3a7970ffa90bb`.
This is reproducibility and data-contract evidence only. The profile permits
missing human labels and is not a safety certification; the dataset license
declaration is not project-specific legal approval. Raw and derived data stay
outside Git.

## Local oracle evidence

The real gate compared the imported ANN against
`tests/fixtures/chatbot/qwen3_phase0_reference_logits.json` on CPU float32. All
three cases matched the exact requested probe IDs and ordered top-16 IDs, and
all compared logits satisfied the fixture tolerances. Measured resource data
was 57.60 seconds wall time and 3,344,712 KiB peak RSS.

The executable was a Release build using `/usr/bin/c++` and the CPU-only
PyTorch/LibTorch 2.7.1+cpu installation under the conversion environment. The
command explicitly used that installation's matching Torch
`LD_LIBRARY_PATH`. This is not a result from the older 2.3 dependency matrix.
The gate output, archive, build tree, and source assets remain ignored local
artifacts, so this result is labelled development evidence. Promotion requires
a durable report with the runner, reference, archive, commands, source
revisions, dependency inventory, and all hashes.

The gate proves only bounded final-position parity for the checked probes and
top-k values. It does not prove full-logit tensor identity or generation
equivalence. It says nothing about SNN-mode parity, training, assistant
perplexity, generated-answer quality, latency advantage, or energy.

## Implemented contract

- Exact official `Qwen/Qwen3-0.6B-Base` repository revision, weight,
  config/tokenizer hashes, and tokenizer fingerprint.
- Strict `snnbase-chatbot-jsonl-v1` roles and conversation validation.
- Assistant-content-plus-EOM loss mask and order-independent
  `sha256-seed-bucket-v1` split, including the shared
  `seed=42, conversation-17 -> bucket 7457` golden.
- Immutable prepared-token shards, a bounded persistent token protocol, and
  portable SHA-256 plus fail-closed run-manifest validation.
- Deterministic Qwen inventory/conversion, exact geometry validation, archive
  identity verification, bounded logit inspection, and an immutable-output
  oracle gate.
- A training driver that flushes at epoch boundaries, uses validation loss
  only (including epoch zero) for checkpoint selection, reloads the selected
  checkpoint in a second process, verifies metadata, and touches test once.
- An interactive client using the verified official Qwen chat template,
  persistent multi-turn state, bounded context/history, deterministic
  generation controls, `/reset`, and `/quit`.
- Offline Python contract tests and direct non-Torch C++ dataset/manifest
  builds, plus language-enabled local runner smoke tooling.

The checked JSON configs are executable contract artifacts. The training
driver validates them, launches the runner with their canonical argument
arrays, and records the exact consumed config; publication assembly rechecks
that binding. The compact checkpoint signature includes spiking mode, so
compact ANN and SNN checkpoints are not currently interchangeable.

## Gates before reportable training results

1. Complete owner/legal and policy approval of the pinned OASST2 candidate and
   retain the conversion/preparation provenance bundle.
2. Complete fuzzy and semantic near-duplicate analysis before promoting the
   already exact-normalized-deduplicated IDs/splits.
3. Publish and pin a clean language-enabled `snnbase` revision and execute the
   LibTorch chatbot CI smoke on a deterministic dependency stack.
4. Retain the fully resolved tokenizer/reference environment and hashes for
   every consumed model and tokenizer file.
5. Execute the compact ANN and SNN controls from their explicit configs, retain
   validation-selected checkpoints, and report the untouched test split.
6. Decide whether compact ANN-to-SNN tensor transplant is required; if so,
   implement and test a shape/provenance-aware transfer distinct from resumable
   checkpoint loading.
7. Calibrate or fine-tune the exact-dense hybrid SNN, then evaluate it against
   the exact ANN using the same immutable data and prompts. Do not assume ANN
   parity survives LIF activation.
8. Repeat promoted comparisons over preregistered seeds and retain individual
   values plus aggregate uncertainty.
9. Validate every result manifest against
   `schemas/chatbot-run-v1.schema.json`; set `publishable=true` only when all
   clean-source, provenance, environment, artifact, and metric gates pass.
10. Treat spike or operation counts as proxies only. Any power/energy claim
    requires direct measurement on named hardware.

Until these gates pass, interactive generations are qualitative demonstrations
and must not be used to tune against the held-out test split.
