# Chatbot experiment contract

This document defines the Phase 0 conversational-language work as two distinct
tracks. Neither track currently establishes chatbot quality or an energy
advantage.

| Track | Checked configurations | Purpose | Current evidence |
|---|---|---|---|
| Exact Qwen import | [`qwen3-0.6b-ann-import.json`](../configs/chatbot/qwen3-0.6b-ann-import.json), [`qwen3-0.6b-hybrid-snn.json`](../configs/chatbot/qwen3-0.6b-hybrid-snn.json) | Prove faithful dense Qwen import in ANN mode, then use the same dense tensors with experiment-owned LIF dynamics | Conversion/import and the bounded oracle path exist. A local ignored three-case ANN gate passed; it is not a promoted result. The hybrid SNN has not been calibrated, trained, or evaluated. |
| Compact trainable controls | [`ann-baseline.json`](../configs/chatbot/ann-baseline.json), [`snn-baseline.json`](../configs/chatbot/snn-baseline.json) | Exercise seeded, provenance-controlled ANN/SNN training at feasible scale | Prepare/train/checkpoint-selection plumbing exists and is tested with fixtures plus a tiny local CPU wiring run. The leakage-remediated pinned OASST2 snapshot is approved for restricted internal research; no full quality run exists. |

The SNNs are activation-first hybrids. Attention, projections, normalization,
embedding, and readout remain dense; only attention-output and feed-forward
intermediate activations pass through LIF sites. “Hybrid SNN” does not mean a
fully asynchronous model and does not support a hardware-energy claim.

## Current implementation boundary

Implemented:

- strict conversation JSONL validation, assistant-only masking, deterministic
  splitting, immutable prepared shards, and fail-closed provenance checks;
- a bounded persistent C++ token protocol supporting metadata, train,
  evaluate, generate, final-position logit inspection, flush, reset, and
  shutdown operations;
- a deterministic Qwen dense archive converter and a loader that validates the
  archive, source checkpoint, config, tokenizer fingerprint, tensor names, and
  shapes before copying weights;
- a CPU float32 Qwen oracle gate over bounded final-position probes and ordered
  top-k logits;
- deterministic epoch orchestration, an epoch-zero validation candidate,
  validation-loss-only checkpoint selection, reload in a second process, and a
  single test evaluation of the selected checkpoint; and
- an interactive client using the verified official Qwen chat template.

Not established:

- separate legal/privacy approval before redistributing OASST2-derived text or
  using it outside the restricted internal-research scope;
- full ANN or SNN training, held-out perplexity, or generation quality;
- SNN calibration or fine-tuning after exact dense Qwen import;
- a promoted, durable exact-import oracle artifact; or
- comparative latency, power, or energy evidence.

The JSON configs are executable contracts. `chatbot_train.py --config` validates
each document against the checked schema, derives the canonical core argument
array without shell parsing, and records the exact config hash in the completed
run. The C++ core receives that resolved argument array; promoted manifests
independently verify it against the selected config.

## Immutable Qwen contract

Phase 0 uses the official text-only pretrained
[`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) assets:

| Field | Required value |
|---|---|
| Repository | `Qwen/Qwen3-0.6B-Base` |
| Revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| License | Apache-2.0 |
| `config.json` SHA-256 | `504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59` |
| `model.safetensors` SHA-256 | `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` |
| `tokenizer.json` SHA-256 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `tokenizer_config.json` SHA-256 | `3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5` |
| Tokenizer fingerprint | `6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237` |

These values were resolved from the official repository on 2026-09-01. A
mutable branch, abbreviated revision, mismatched tokenizer, or unverified
weight file fails closed.

Install the locked reference environment, download the exact revision, and
verify both tokenizer and bounded CPU reference fixtures with:

```bash
python3 -m venv .venv-chatbot-reference
.venv-chatbot-reference/bin/pip install -r requirements/qwen-reference.lock
SNNBASE_REFERENCE_PYTHON=.venv-chatbot-reference/bin/python \
  ./scripts/download_qwen_assets.sh \
  --model-id Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --expected-tokenizer-fingerprint \
    6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237 \
  --output-dir artifacts/chatbot/qwen3-0.6b-base \
  --include-weights
SNNBASE_REFERENCE_PYTHON=.venv-chatbot-reference/bin/python \
  ./scripts/check_qwen_reference.sh \
  --assets-dir artifacts/chatbot/qwen3-0.6b-base \
  --include-logits
```

The helper resolves `hf` or `huggingface-cli` from the selected reference
environment. Network access is used only by the explicit download step; both
fixture checks then force Hugging Face and Transformers offline modes.

The exact architecture is vocabulary 151,936, model width 1,024, 28 layers,
16 query heads, 8 key/value heads, explicit head width 128, feed-forward width
3,072, per-head query/key RMSNorm, tied embedding/readout, SiLU/SwiGLU,
RMS epsilon `1e-6`, RoPE base 1,000,000, and 32,768 trained positions. The
Phase 0 executable caps execution at 512 tokens by default. The explicit
128-wide head means the query projection is 2,048 wide; deriving head width as
`model_dimension / query_head_count` would be incorrect.

Tokenizer assertions are no automatic BOS, `<|endoftext|>` 151643,
`<|im_start|>` 151644, and `<|im_end|>` 151645. Generation stops on 151645 or
151643. The model has 151,936 padded embedding/readout rows while the verified
tokenizer exposes 151,669 decodable IDs. Every frontend generation request
therefore binds `sampling_vocabulary_size=151669`, and the core slices logits
to that range before greedy, top-k, or top-p selection. Code must apply the
pinned tokenizer artifact and chat template rather than reimplementing BPE or
template rules.

## Exact dense conversion and parity gate

The networked asset download is a separate step. With the verified snapshot at
`artifacts/chatbot/qwen3-0.6b-base`, first validate the complete tensor mapping
without writing payloads:

```bash
python3 tools/qwen_convert.py inventory \
  --assets-dir artifacts/chatbot/qwen3-0.6b-base \
  --model-id Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --expected-tokenizer-fingerprint \
    6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237 \
  --output artifacts/chatbot/qwen3-0.6b-inventory.json
```

Then write the deterministic archive to a new output directory:

```bash
python3 tools/qwen_convert.py convert \
  --assets-dir artifacts/chatbot/qwen3-0.6b-base \
  --model-id Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --expected-tokenizer-fingerprint \
    6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237 \
  --output-dir artifacts/chatbot/qwen3-0.6b-base-dense-v1
```

For this source and converter contract, `qwen-dense.snnq` is 1,192,143,104
bytes with SHA-256
`333c8be1b3fde5013ee8d1dcb96f23051dd934f8e1f359aff329e7e5cf8ccac8`.
It contains 310 BF16 dense tensors totaling 1,192,099,840 payload bytes. The
112 threshold/leak tensors at the two LIF sites in each of 28 layers are not
Qwen checkpoint tensors and are initialized from the runtime decoder config.

Run the real CPU float32 oracle gate against the checked reference fixture:

```bash
./scripts/run_with_chatbot_libtorch.sh --build-dir build-chatbot -- \
  python3 tools/qwen_oracle_gate.py \
  --runner ./build-chatbot/chatbot_experiment \
  --archive artifacts/chatbot/qwen3-0.6b-base-dense-v1/qwen-dense.snnq \
  --archive-sha256 \
    333c8be1b3fde5013ee8d1dcb96f23051dd934f8e1f359aff329e7e5cf8ccac8 \
  --reference tests/fixtures/chatbot/qwen3_phase0_reference_logits.json \
  --output artifacts/chatbot/qwen3-oracle-gate.json
```

The locally retained ignored development run passed all three cases: probe IDs
and ordered top-16 IDs matched, maximum absolute error was
`9.5367431640625e-06`, maximum relative error was
`8.471997478955767e-06`, and both tolerances were `1e-5`. It took 57.60 seconds
wall time with peak RSS 3,344,712 KiB. This supports bounded final-position ANN
logit parity on that machine. It is not full-tensor identity, a committed gate
artifact, conversational quality, or a promoted baseline.

The hybrid conversion loads the same 310 dense tensors under `--snn` and adds
the 112 runtime LIF parameters. ANN parity does not transfer through the LIF
nonlinearity; the hybrid must be calibrated or fine-tuned and evaluated as a
separate model.

## Dataset and assistant-loss contract

Source data is UTF-8 JSON Lines, one complete conversation per nonempty line:

```json
{"id":"example-0001","messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
```

Each object contains exactly `id` and `messages`. IDs are nonempty and unique.
Messages contain exactly `role` and `content`; roles are `system`, `user`, and
`assistant`. An optional single system message comes first. Remaining turns
alternate user/assistant, begin with user, and end with assistant. Parsing fails
closed on invalid UTF-8/JSON, unknown or duplicate fields, bad ordering,
duplicate IDs, NULs, trailing JSON, and configured resource-limit violations.

`sha256-seed-bucket-v1` hashes `seed as eight-byte big-endian || UTF-8 id`,
interprets the first eight digest bytes as big-endian, and reduces modulo
10,000. Buckets 0–999 are test, 1000–1999 validation, and 2000–9999 train. The
shared golden fixture includes seed 42 plus `conversation-17` mapping to 7457.
Grouping and near-duplicate checks are still required because ID hashing cannot
prevent semantic leakage.

Training applies the official chat template with
`add_generation_prompt=false`. Loss selects assistant content and its
`<|im_end|>` token only. System/user text, role headers, other control tokens,
and padding are excluded. The causal loss shifts by one token. Packing is off;
overlength conversations are rejected rather than truncated.

The source SHA-256 and byte count are accumulated from the exact binary stream
that is parsed and tokenized. Before the immutable shards are finalized, the
source pathname is rechecked against that consumed digest; a changed source
fails instead of attaching new provenance to old tokens.

The OASST2 `2023-11-05_oasst2_ready.trees.jsonl.gz` snapshot at revision
`179dd21fc55192153d94adb0e0ce8f69e222bf75` is approved for restricted internal
research under the [dataset approval record](OASST2_DATASET_APPROVAL.md). Its
declared license is Apache-2.0. This is not legal approval to redistribute the
text or deploy a public service, and neither available conversion profile is a
universal safety filter. No full training or held-out quality result exists.

The recommended development profile, `quality05`, accepts same-language paths
whose messages are reviewed, have a positive review count, are not deleted,
and are not synthetic. When present, `spam`, `lang_mismatch`, and `pii` must be
below 0.5; assistant `quality` and `helpfulness` must be at least 0.5, and
assistant `fails_task` must be below 0.5. Missing labels are allowed. Topic
labels and Detoxify scores are not a safety gate. The converter evaluates every
assistant-ended candidate with the exact pinned Qwen chat template, retains the
longest eligible endpoint at or below 512 tokens, and emits at most one path per
tree. It then deduplicates selected records by NFKC-normalized, case-folded,
Unicode-whitespace-collapsed root prompt, keeping the first eligible tree in
pinned source order. A subsequent fixed-threshold fuzzy and pinned multilingual
semantic audit found 168 high-confidence cross-split pairs. Its deterministic
remediation removed 137 train and 7 validation records while preserving every
test record. The remediated output was audited again; see the approval record
for the exact hashes, thresholds, limitations, and permitted claim.

For the pinned input, the pre-remediation `quality05` output is 12,427 records:
9,990 train, 1,165 validation, and 1,272 test. The approved output is 12,283
records: 9,853 train, 1,158 validation, and 1,272 test. The exact 512-token check is the
sequence-length gate; the default 32,768-byte aggregate-content limit is an
additional pre-tokenization resource bound (the longest accepted conversation
contains 3,357 content bytes). These are conversion counts, not a quality
result. The stricter `conservative-zero` profile is retained as an audit
alternative. After the same deduplication it keeps 1,700 records (1,407 train,
140 validation, and 153 test). It requires PII and the configured common
adverse labels, plus assistant `fails_task`, to be present and exactly zero;
this nearly eliminates multi-turn records and is not the recommended default.
It is not a universal safety guarantee either.

## Prepare and train

Place the exact pinned archive at
`data/source/oasst2/2023-11-05_oasst2_ready.trees.jsonl.gz`. With the verified
Qwen assets and reference environment already present, the following converter
command is entirely offline; the HTTPS URI is recorded provenance and is not
fetched:

```bash
.venv-chatbot-reference/bin/python tools/oasst2_convert.py \
  --profile quality05 \
  --assets-dir artifacts/chatbot/qwen3-0.6b-base \
  --model-id Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --expected-tokenizer-fingerprint \
    6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237 \
  --input data/source/oasst2/2023-11-05_oasst2_ready.trees.jsonl.gz \
  --expected-source-sha256 \
    7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b \
  --expected-source-bytes 54370156 \
  --source-uri \
    https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/179dd21fc55192153d94adb0e0ce8f69e222bf75/2023-11-05_oasst2_ready.trees.jsonl.gz \
  --source-version 179dd21fc55192153d94adb0e0ce8f69e222bf75 \
  --source-license Apache-2.0 \
  --license-reviewed --policy-reviewed \
  --output-dir \
    artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-v1 \
  --max-input-tokens 512
```

The two review flags are required attestations that the declared license
metadata and exact selection policy were examined. They do not assert legal
approval, universal safety, or model quality. The converter publishes a
write-once bundle containing `conversations.jsonl`, `lineage.jsonl`, and
`conversion-manifest.json`; the manifest records the raw compressed and
decompressed hashes, converter and tokenizer identity, policy, filter and split
counts, output hashes, and source-message lineage.

Create a separate leakage environment from
[`chatbot-leakage.lock`](../requirements/chatbot-leakage.lock), download
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, and audit that immutable baseline:

```bash
python3 -m venv .venv-chatbot-leakage
.venv-chatbot-leakage/bin/pip install \
  -r requirements/chatbot-leakage.lock
.venv-chatbot-leakage/bin/python -c \
  "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', revision='e8f8c211226b894fcb81acc59f3b34ba3efd5f42', local_dir='artifacts/chatbot/leakage-models/paraphrase-multilingual-MiniLM-L12-v2-e8f8c211', allow_patterns=['*.json','*.safetensors','*.model','*.txt','1_Pooling/*'])"
.venv-chatbot-leakage/bin/python tools/chatbot_leakage_audit.py \
  --conversion-manifest \
    artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-v1/conversion-manifest.json \
  --model-dir artifacts/chatbot/leakage-models/paraphrase-multilingual-MiniLM-L12-v2-e8f8c211 \
  --model-id sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --model-revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42 \
  --device cuda \
  --output \
    artifacts/chatbot/leakage-audits/oasst2-quality05-v5/leakage-audit-v1.json \
  --exclusions-output \
    artifacts/chatbot/leakage-audits/oasst2-quality05-v5/leakage-exclusions-v1.json
```

Rerun the converter command above with its `--output-dir` changed to
`artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-leakage-reviewed-v6`
and add:

```bash
  --leakage-audit \
    artifacts/chatbot/leakage-audits/oasst2-quality05-v5/leakage-audit-v1.json \
  --expected-leakage-audit-sha256 \
    a5fc18c713e6e35e6c553b63e975de024a5b1500c98dc1294fe2b0f40c5f9d87 \
  --leakage-exclusions \
    artifacts/chatbot/leakage-audits/oasst2-quality05-v5/leakage-exclusions-v1.json \
  --expected-leakage-exclusions-sha256 \
    e8a297c92b3c5af0712b61f83a78d261433e68afca03020028665791efc154f0
```

The converter rejects any mismatch with the exact audited pre-remediation byte
stream or any exclusion ID that is not encountered. The approved result and
scope restrictions are recorded in
[`OASST2_DATASET_APPROVAL.md`](OASST2_DATASET_APPROVAL.md).

Prepare immutable token shards from the canonical conversation output. The
content-addressed URI/version below comes directly from the conversion
manifest, so the prepared dataset manifest's source SHA-256 joins unambiguously
to the pinned archive and conversion policy:

```bash
OASST2_CONVERSION_DIR=artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-leakage-reviewed-v6
OASST2_CANONICAL_SHA256="$(
  python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["output"]["sha256"])' \
    "$OASST2_CONVERSION_DIR/conversion-manifest.json"
)"
.venv-chatbot-reference/bin/python tools/chatbot_prepare.py \
  --assets-dir artifacts/chatbot/qwen3-0.6b-base \
  --model-id Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --expected-tokenizer-fingerprint \
    6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237 \
  --input "$OASST2_CONVERSION_DIR/conversations.jsonl" \
  --conversion-manifest \
    "$OASST2_CONVERSION_DIR/conversion-manifest.json" \
  --source-uri "urn:sha256:$OASST2_CANONICAL_SHA256" \
  --source-version "sha256:$OASST2_CANONICAL_SHA256" \
  --source-license Apache-2.0 \
  --output-dir \
    artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-leakage-reviewed-prepared-v6 \
  --max-input-tokens 512
```

Raw `data/` and derived `artifacts/` are ignored by Git and must not be
committed. The prepared directory retains hash-verified copies named
`source-conversion-manifest.json` and `source-lineage.jsonl`; the trainer
revalidates them and the publisher records them as separate artifacts.
Conversion and preparation are data-contract gates, not training or quality
results.

All four resolved configs are strict executable inputs. Each command validates
the config schema and its architecture/geometry/training-to-runner binding
before launching the persistent core.

Compact ANN:

```bash
./scripts/run_with_chatbot_libtorch.sh --build-dir build-chatbot -- \
  python3 tools/chatbot_train.py \
  --dataset-dir \
    artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-leakage-reviewed-prepared-v6 \
  --run-dir artifacts/chatbot/runs/compact-ann-seed42 \
  --core-executable ./build-chatbot/chatbot_experiment \
  --config configs/chatbot/ann-baseline.json
```

Compact hybrid SNN:

```bash
./scripts/run_with_chatbot_libtorch.sh --build-dir build-chatbot -- \
  python3 tools/chatbot_train.py \
  --dataset-dir \
    artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-leakage-reviewed-prepared-v6 \
  --run-dir artifacts/chatbot/runs/compact-snn-seed42 \
  --core-executable ./build-chatbot/chatbot_experiment \
  --config configs/chatbot/snn-baseline.json
```

Exact imported ANN:

```bash
./scripts/run_with_chatbot_libtorch.sh --build-dir build-chatbot -- \
  python3 tools/chatbot_train.py \
  --dataset-dir \
    artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-leakage-reviewed-prepared-v6 \
  --run-dir artifacts/chatbot/runs/qwen3-0.6b-ann-seed42 \
  --core-executable ./build-chatbot/chatbot_experiment \
  --config configs/chatbot/qwen3-0.6b-ann-import.json
```

Exact-dense hybrid SNN:

```bash
./scripts/run_with_chatbot_libtorch.sh --build-dir build-chatbot -- \
  python3 tools/chatbot_train.py \
  --dataset-dir \
    artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-leakage-reviewed-prepared-v6 \
  --run-dir artifacts/chatbot/runs/qwen3-0.6b-hybrid-snn-seed42 \
  --core-executable ./build-chatbot/chatbot_experiment \
  --config configs/chatbot/qwen3-0.6b-hybrid-snn.json
```

The compact checkpoint contract includes spiking mode, so it correctly rejects
direct loading of a compact ANN checkpoint into the compact SNN. Until a
tensor-only transplant is implemented, these are independently initialized
controls. Neither full-size Qwen training path has been run; resource
feasibility and optimization quality are unknown.

At each epoch boundary the driver flushes accumulated gradients, compares
validation loss against the epoch-zero candidate, and preserves only strict
improvements. It then stops the training process, reloads the selected
checkpoint into a second process, validates immutable metadata and exact
training counters, reproduces the selected validation aggregate within a
deterministic tolerance, and evaluates the test shard once only after those
checks pass. It independently recomputes every record's deterministic split
when loading prepared shards. Run directories and outputs are immutable.

## Interactive inference

Exact imported ANN:

```bash
./scripts/run_with_chatbot_libtorch.sh --build-dir build-chatbot -- \
  .venv-chatbot-reference/bin/python tools/chatbot_cli.py \
  --assets-dir artifacts/chatbot/qwen3-0.6b-base \
  --core-executable ./build-chatbot/chatbot_experiment \
  --qwen-archive artifacts/chatbot/qwen3-0.6b-base-dense-v1/qwen-dense.snnq \
  --qwen-archive-sha256 \
    333c8be1b3fde5013ee8d1dcb96f23051dd934f8e1f359aff329e7e5cf8ccac8 \
  --ann --device cpu \
  --context-tokens 512 --max-new-tokens 128 \
  --seed 42 --temperature 0
```

The Phase 0 pin is a pretrained **Base** model, not an instruction-tuned chat
checkpoint. The official template makes the interface reproducible, but raw
interactive generations are plumbing demonstrations until the model is
fine-tuned and evaluated on an approved conversation corpus.

Use `--snn` for an uncalibrated hybrid smoke only. For a trained compact model,
replace the Qwen archive arguments with `--checkpoint PATH`, retain the
512-token context, and pass the checkpoint's `--ann` or `--snn` mode. For an
exact-geometry Qwen checkpoint, also pass `--qwen3-0.6b`.

`/reset` clears multi-turn history and resets the persistent core; `/quit`
shuts it down. The client reserves completion space, evicts only whole oldest
turns, applies configured byte/message limits, invokes no shell, and decodes the
assistant tokens. Greedy generation is selected by `--temperature 0`; sampled
requests require fixed seed, temperature, top-k, and top-p, though cross-device
bitwise determinism is not promised. The wrapper binds both Python and its C++
child to the `Torch_DIR` recorded in the selected CMake build cache.

## Metrics, manifests, and publication

Training reports target-token-weighted assistant loss, perplexity, accuracy,
and mean spike rate for each phase. Generated quality, throughput, memory, and
operation counts require separate measured protocols; they are not emitted by
the current training summary. Perplexity is
`exp(total assistant NLL / supervised assistant tokens)`, not an average of
per-record perplexities.

Every promoted result must conform to
[`chatbot-run-v1.schema.json`](../schemas/chatbot-run-v1.schema.json) and retain
full experiment and `snnbase` revisions, dirty state, command, resolved config,
dataset and split provenance, environment/hardware, and hashes for executable,
all consumed model/tokenizer files, checkpoints, logs, and metrics. Operation
counts are not energy. Only direct power/energy measurement on named target
hardware can support an energy claim.

After a config-driven run completes, assemble and independently validate its
immutable publication manifest. The environment input is an exact JSON object
with `toolchain` fields `compiler`, `cmake`, `torch`, and `cuda`, plus `hardware`
fields `cpu` and `gpu`; use an explicit value such as `"none"` rather than an
ambiguous empty value in a publishable CPU run.

```bash
python3 tools/chatbot_publish.py assemble \
  --run-dir artifacts/chatbot/runs/compact-ann-seed42 \
  --config configs/chatbot/ann-baseline.json \
  --dataset-source \
    artifacts/chatbot/oasst2-2023-11-05-quality05-qwen3-512-leakage-reviewed-v6/conversations.jsonl \
  --experiments-repo . \
  --snnbase-repo ../snnbase \
  --environment artifacts/chatbot/environment.json \
  --qwen-assets-dir artifacts/chatbot/qwen3-0.6b-base \
  --run-id compact-ann-seed42 \
  --dataset-name oasst2-2023-11-05-quality05-development-candidate \
  --output artifacts/chatbot/runs/compact-ann-seed42/publication-manifest.json
python3 tools/chatbot_publish.py validate \
  --manifest \
    artifacts/chatbot/runs/compact-ann-seed42/publication-manifest.json
```

Add `--publishable` to `assemble` and `--require-publishable` to `validate` only
for a retained approved-dataset run from clean repository revisions. Assembly
rechecks the consumed config and command, source/shard hashes and split
contract, executable, selected checkpoint, one-time test metrics, official
Qwen/tokenizer assets, environment, and the configure-time Git revisions and
dirty flags embedded in the executable. Publishable mode also requires the
current clean checkouts to match those embedded build revisions exactly, so an
older binary cannot be relabeled with newer source. Output creation is
exclusive so an existing manifest is never silently overwritten.

The offline implementation gates remain:

```bash
./scripts/check_chatbot_contracts.sh
./scripts/configure_chatbot.sh \
  --snnbase-source /path/to/snnbase \
  --snnbase-revision c282306ee3b80ccc5123fcb8b2da78ae51ed09fe \
  --torch-prefix /path/to/libtorch/share/cmake \
  --build-dir build-chatbot \
  --build-type Release
./scripts/run_chatbot_smoke.sh --build-dir build-chatbot
```

The default toolchain selects the validated CUDA 12.1/GCC 11/SM86 matrix. Add
`--toolchain none` when `--torch-prefix` names a CPU-only LibTorch build. The
local ANN parity evidence used the latter kind of Release build with
PyTorch/LibTorch 2.7.1+cpu; it is deliberately not described as evidence from
the CUDA 12.1 matrix.

The first reportable-result boundary is tracked in
[`results/CHATBOT_BASELINE.md`](results/CHATBOT_BASELINE.md).
