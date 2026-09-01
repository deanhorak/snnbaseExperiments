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
| Compact ANN | Prepare/train/validation-selection/reload/test plumbing implemented | Fixture/fake-core tests and tiny runner smoke | No approved dataset or full run; quality unavailable |
| Compact hybrid SNN | Independently trainable compact SNN path implemented | Fixture/fake-core tests and tiny runner smoke | No approved dataset or full run; ANN-to-SNN checkpoint transplant not implemented; quality unavailable |

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

1. Select and document a legally usable conversation dataset, immutable
   source/version, license, filtering, policy review, and SHA-256.
2. Complete leakage and near-duplicate analysis before freezing IDs/splits.
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
