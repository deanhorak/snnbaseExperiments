# Abstract temporal spiking language model

See [the measured Qwen conversion results](results/TEMPORAL_LLM_RESULTS.md) for
the validated checkpoint, development comparisons and remaining limitations.

`tools/temporal_llm.py` runs a full-vocabulary causal language model implemented
by `snnbase::language::Decoder`. The upstream tokenizer encodes the prompt; the
C++ library performs every model forward pass and selects generated token IDs;
the tokenizer decodes those IDs into text. Python does not run a Transformers
model or use a lookup table to choose answers.

The initial pretrained profile is **Qwen/Qwen3-0.6B-Base** (the upstream name uses
`0.6B`, not `0.6N`). Its verified dense parameters initialize the temporal
decoder. The `tiny` profile uses the same complete tokenizer vocabulary with a
small random decoder to exercise learning and serialization quickly. Random
initialization has no pretrained language ability.

The original small reservoir experiment remains a useful separate baseline.
It learns a restricted output vocabulary from its demonstration corpus. The
LLM path samples every valid upstream tokenizer ID and excludes only padded
model rows that have no tokenizer representation.

## What is spiking

The temporal variant is selected with the C++ `--temporal-spiking` option.
Activation values are represented by signed events over a configurable temporal
window at the decoder's attention-output and feed-forward activation sites.
Channels share a clock whose pulse thresholds progressively decrease; positive
and negative event channels represent opposing abstract populations. This
clocked pulse code is an engineering approximation, not evidence of emergent
biological synchronization. Calibration measures per-channel activation scales
using training/calibration text. The Qwen profile reserves 16.16 times the
observed absolute maximum (sixteenfold headroom plus 1% endpoint slack), while
the tiny profile keeps the library's 4.04× default. Greater headroom reduces
clipping on unseen activations at the cost of coarser finite-step precision. The
optional temporal decay carries abstract activity between causal token
positions; it does not model ion channels, neurotransmitters or molecular
membrane mechanisms. A zero decay is the default for measuring how closely the
converted pretrained model retains its original behavior.

This is a **hybrid spiking decoder**. Embeddings, projection matrices, grouped
query attention, softmax, normalization and vocabulary readout still use dense
tensor operations. The inherited Qwen matrices contain signed effective
connections; they are not a proof of separate biological excitatory/inhibitory
cell classes. Attention couples token representations, but that alone does not
establish biological oscillatory synchronization. More spike events or a large
microstep count do not by themselves imply better accuracy or efficiency.

The ANN control uses the same dense source parameters without temporal encoding.
Compare held-out likelihood, output tokens, activity and measured latency before
claiming successful conversion. There is no claim here that copying Qwen
parameters produces an equally capable or faster spiking model automatically.

## Build and run locally

The validated local checkout already contains the calibrated Qwen checkpoint at
`artifacts/temporal_llm/qwen3-0.6b-calibrated-h16.pt` and its `.temporal.json` sidecar.
These large artifacts are ignored by Git. Start a CPU continuation immediately:

```bash
python3 /home/dean/repos/snnbaseExperiments/tools/temporal_llm.py \
  --checkpoint /home/dean/repos/snnbaseExperiments/artifacts/temporal_llm/qwen3-0.6b-calibrated-h16.pt \
  --weights-only --device cpu --prompt "The capital of France is" --max-new-tokens 24
```

The recorded run produced `Paris. The capital of Spain is Madrid.` and continued
with other capitals. It used the source Qwen weights plus calibration only.
This is a development smoke result, not instruction tuning or a general quality
benchmark. A fresh clone requires the tokenizer/source archive or a copied
checkpoint and sidecar before running inference.

From `/home/dean/repos/snnbaseExperiments`, the validated local build uses
`/opt/libtorch` 2.3.0 and the GCC 11/CUDA 12.1 toolchain. CUDA-enabled LibTorch can
run these CPU commands without an available GPU. Python needs `tokenizers`
(validated locally with 0.22.2); the frontend does not import Python Torch.

```bash
env -u LD_LIBRARY_PATH cmake -S . -B build-temporal-llm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/libtorch/share/cmake \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-cuda121-gcc11.cmake \
  -DSNNBASE_EXPERIMENTS_ENABLE_TEMPORAL=OFF \
  -DSNNBASE_EXPERIMENTS_ENABLE_CHATBOT=ON \
  -DSNNBASE_EXPERIMENTS_ENABLE_GUI=OFF \
  -DSNNBASE_ENABLE_CUDA_LANGUAGE_KERNELS=ON \
  -DSNNBASE_SOURCE_DIR=/home/dean/repos/snnbase \
  -DSNNBASE_EXPECTED_SNNBASE_REVISION=10ba9df7cd1819c198124666d11c4ae2b3499ebf \
  -DSNNBASE_REQUIRE_CLEAN_SNNBASE=ON
cmake --build build-temporal-llm --target chatbot_experiment -j 2
python3 tools/temporal_llm.py --profile qwen3-0.6b --plan
```

The frontend automatically uses `scripts/run_with_chatbot_libtorch.sh` when the
executable's build directory records its Torch installation. This avoids
loading ABI-incompatible libraries from an unrelated Python Torch environment.
Set `--core-executable /absolute/path/chatbot_experiment` for another build.
For a genuinely CPU-only LibTorch installation, configure a separate build with
that installation's CMake prefix and omit the CUDA toolchain option.

The reproducible library dependency is commit
`10ba9df7cd1819c198124666d11c4ae2b3499ebf` on `ReturmToBasics`. The configure
command requires that exact clean `snnbase` checkout; select that commit before
building if the library has subsequently advanced. Historical measurement
reports retain the executable digests from the runs that produced them.

All execution is offline. The default profile expects these existing artifacts:

* `artifacts/chatbot/qwen3-0.6b-base/`: pinned tokenizer/config/weights snapshot.
* `artifacts/chatbot/qwen3-0.6b-base-dense-v1/qwen-dense.snnq`: verified converted
  BF16 source archive; the runner currently loads its parameters as FP32.

`--assets-dir` can relocate the tokenizer, but its fingerprint must still match
the selected profile. The frontend never downloads a model implicitly.

## Calibration and a bounded comparison

Supply separate calibration and evaluation files. A `.txt` file is one document
per nonblank line. A `.jsonl` file contains one of these forms per line:

```json
{"text":"A complete passage whose next tokens will all be predicted."}
{"prompt":"The capital of France is", "response":" Paris."}
```

For a prompt/response record, only targets touching the response are scored.
The complete string is tokenized together so BPE boundary merges are preserved.
Whitespace is meaningful inside JSON strings: include the leading space in a
response when appropriate. Long records are rejected instead of silently
truncated. Split long documents into suitably bounded records beforehand.

```bash
python3 tools/temporal_llm.py --profile qwen3-0.6b \
  --calibration-corpus configs/temporal_llm/calibration.jsonl \
  --eval-corpus configs/temporal_llm/evaluation.jsonl \
  --diagnostics-corpus configs/temporal_llm/evaluation.jsonl \
  --max-records 32 --compare-ann \
  --save-checkpoint artifacts/temporal_llm/qwen3-0.6b-recalibrated.pt \
  --report artifacts/temporal_llm/calibration-report.json
```

These supplied files contain 32 authored calibration passages and 16 separate
authored evaluation passages for a reproducible, bounded language check. They
are synthetic smoke data, not a benchmark establishing general language ability.
The default calibration headroom was selected using development comparisons;
results used to select that setting are development evidence, not an independent
final test, even when the calibration and evaluation passages are disjoint.
The Qwen profile's sixteenfold headroom was selected from bounded 4×, 8×, 16×,
and 32× development comparisons. Override it with
`--calibration-headroom` when evaluating a different model or corpus.
`--max-records` defaults to eight records **per corpus**. Calibration
uses only its selected calibration records, then freezes the resulting scales
for evaluation and generation. The ANN comparison runs afterward in a separate
process to avoid holding two full models simultaneously. It reports
token-weighted NLL/perplexity and agreement for one final next-token prediction
per selected evaluation record; that agreement is not an all-token metric.

Evaluation rejects token-identical records seen in calibration/training. The
checkpoint sidecar preserves these record identities across resumed runs. This
check detects exact overlap; it does not detect paraphrases or shared source
documents. Choose genuinely independent data splits for substantive evaluation.

`--diagnostics-corpus` records raw and derived metrics for every layer's
attention and feed-forward temporal encoder: saturation, silent nonzero input,
absolute reconstruction error, and maximum input-to-scale ratio. It may reuse
calibration records because it does not score language quality. Diagnostics
are opt-in and excluded from generation benchmarks because their reductions
and device transfers deliberately add overhead.

## Text generation

```bash
python3 tools/temporal_llm.py \
  --checkpoint artifacts/temporal_llm/qwen3-0.6b-calibrated-h16.pt \
  --prompt "The capital of France is" --max-new-tokens 24

python3 tools/temporal_llm.py \
  --checkpoint artifacts/temporal_llm/qwen3-0.6b-calibrated-h16.pt --chat
```

`--prompt` performs raw continuation. `--chat` uses a simple `User:`/`Assistant:`
conversation prefix, retaining complete turns that fit the context. `/reset`
clears history and `/quit` exits. Qwen Base is a pretrained completion model;
this interface does not turn it into an instruction-tuned assistant. Compare
natural text continuation before judging instruction following.

`--temperature 0` uses greedy decoding. `--temperature`, `--top-k`, `--top-p`
and `--seed` control sampling. Prompt prefill uses a KV cache in chunks of
`--prefill-chunk-size` tokens (default 128); generation then feeds one new token
at a time. `--prefill-chunk-size 0` requests a whole-prompt prefill. Prompt length
plus requested completion must fit the selected context.

For repeated generation measurements, keep the model loaded and send the same
prompt through one process:

```bash
python3 tools/temporal_llm.py \
  --checkpoint artifacts/temporal_llm/qwen3-0.6b-calibrated-h16.pt \
  --weights-only --device cuda --prompt "The capital of France is" \
  --max-new-tokens 32 --benchmark-warmups 2 --benchmark-repeats 5 \
  --report artifacts/temporal_llm/qwen3-0.6b-cuda-benchmark.json
```

`--benchmark-repeats` accepts 1–100 measured requests; zero, the default, keeps
ordinary single-prompt generation. `--benchmark-warmups` accepts 0–20 requests
and defaults to two when benchmarking. Benchmarking requires `--prompt`.
Each request resets the KV/neuron state and reuses the same seed and generation
settings. Warmups and measured samples retain every output token ID, decoded
text, token count, core latency, time to first token and frontend wall time in
the JSON report. The summary contains medians from measured samples only,
their total generated tokens and whether their output token IDs agree. The
first measured continuation is printed, followed by the summary on stderr.

Core timing includes completed prompt prefill, decoding and spike-rate
collection, with CUDA synchronized before its timer starts. Frontend wall
timing also includes tokenization, protocol transport and decoding. Neither
includes model/process startup; the report's total benchmark wall time includes
warmups and bookkeeping and is not used for the summary medians. Generation
tokens/second includes prefill and uses the actual number of emitted tokens,
so EOS or differing completions can change the amount of work. Use matching
prompt, context, sampling, thread and token-count settings when comparing runs.

Run CPU and CUDA benchmarks separately. For the current checkpoint, whose
dense weights were imported and calibrated without fine-tuning, obtain the
source ANN control in another process using `--profile qwen3-0.6b --ann` in
place of `--checkpoint ... --weights-only`, retaining the other settings.
Check model identity and actual generated token counts in both reports.
Checkpoint signatures deliberately prevent switching a saved SNN checkpoint
to ANN mode. A source-archive ANN control would not isolate conversion effects
after SNN fine-tuning because its dense weights would differ. These benchmark
options measure the implementation; they make no speed or energy claim.

The report's `metadata.temporal_backend` distinguishes compiled fused CUDA
support from eligibility for this model's generation requests. Fused inference
requires the optional library CUDA kernels, a CUDA device, temporal spiking
enabled and zero cross-token decay. It runs only without gradients; training
retains the existing ATen surrogate path. The metadata reports eligibility,
not per-request profiling evidence of kernel execution. These execution
capabilities are not part of checkpoint signatures.

A direct `--profile qwen3-0.6b --prompt ...` run imports the source weights and
uses initially configured temporal scales. Use calibration and held-out
measurement before treating its text quality as representative. Add `--ann`
when explicitly requesting the source ANN control.

## Bounded learning and checkpoints

```bash
python3 tools/temporal_llm.py --profile tiny \
  --calibration-corpus configs/temporal_llm/calibration.jsonl \
  --train-corpus configs/temporal_llm/calibration.jsonl \
  --eval-corpus configs/temporal_llm/evaluation.jsonl \
  --train-steps 16 --max-records 8 --learning-rate 0.001 \
  --save-checkpoint artifacts/temporal_llm/tiny-trained.pt \
  --report artifacts/temporal_llm/tiny-training.json

python3 tools/temporal_llm.py \
  --checkpoint artifacts/temporal_llm/tiny-trained.pt \
  --train-corpus configs/temporal_llm/calibration.jsonl --train-steps 8 \
  --save-checkpoint artifacts/temporal_llm/tiny-resumed.pt
```

These commands reuse the supplied calibration passages for the small model's
training and use `evaluation.jsonl` for development measurements. They are a
training/checkpoint mechanics smoke, not a language-quality benchmark. The
separate `holdout.jsonl` passages are reserved for the frozen, untuned quality
check and are not used by this training example.

Training performs causal cross-entropy updates with AdamW and surrogate
gradients through temporal event encoding. It is not local STDP or evidence
of biological credit assignment. `--train-steps` counts microbatches and cycles
through the selected training records. `--gradient-accumulation` controls
microbatches per optimizer update; the final partial accumulation is flushed
before saving. The default learning rate is `1e-5`; small random models often
need a larger rate. Full-model Qwen fine-tuning can require considerably more
memory and time than inference.

Calibration/training requires `--save-checkpoint`. The backend checkpoint
contains model parameters, calibration state, optimizer state and counters;
`FILE.pt.temporal.json` binds it by SHA-256 to its profile, neuron settings,
training settings and learning-record identities. Keep both files together.
Resume automatically restores these settings and rejects conflicting geometry
or optimizer arguments. A new save destination is required unless explicitly
resuming and saving the same checkpoint path.

Full training resume is also bound to the saved device type because checkpoints
include optimizer and device RNG state. For inference/evaluation on a different
device, explicitly select `--weights-only`; model parameters and calibration
are loaded while optimizer state, counters and RNG state are left fresh:

```bash
python3 tools/temporal_llm.py \
  --checkpoint artifacts/temporal_llm/qwen3-0.6b-calibrated-h16.pt \
  --weights-only --device cuda --chat
```

This requires a working CUDA-enabled build and GPU access in the launching
terminal. The saved 0.6B checkpoint was subsequently validated on the local
NVIDIA GeForce RTX 3050 with 8 GiB VRAM: weights-only loading completed on
`cuda` and the four-token continuation was ` Paris. The capital`. That short
first-load smoke run is evidence of functional CUDA execution, not a controlled
throughput comparison. The frontend restricts
`--weights-only` to inference/evaluation and rejects training, calibration or
checkpoint saving in that mode. Profile and checkpoint SHA checks still apply.

The bounded frontend does not select the best epoch or train a 0.6B model to
convergence. Its purpose is to establish a working conversion/learning workflow
and expose measurable behavior before investing in longer runs.

## Scaling beyond 0.6B

Profiles under `configs/temporal_llm/` specify vocabulary size, hidden width,
depth, query/KV heads, explicit head dimension, FFN width, context, tied/untied
embedding readout, RMSNorm/RoPE settings and temporal steps/decay. Copy the tiny
profile to a custom JSON path and set arbitrary compatible dimensions for a
new randomly initialized architecture:

```bash
python3 tools/temporal_llm.py --profile configs/temporal_llm/my-larger-model.json --plan
```

`--plan` performs integer arithmetic and reads small JSON files only. It does
not initialize Torch, load a tokenizer, allocate weights or open the source
archive. For the 0.6B profile it reports 596,049,920 dense parameters, about
2.38 GB of FP32 weights and at least 9.54 GB for weights, gradients and AdamW
moments, before activations and workspace. These are decimal byte estimates,
not promises about peak RAM. Full-context attention has a quadratic score
allocation; prefill chunking and last-position vocabulary readout reduce
avoidable inference allocations but do not make training memory constant.

For a supported larger pretrained model, first convert and verify its complete
tensor inventory with the repository's Qwen asset/conversion workflow. Change
profile initialization from `random` to:

```json
{
  "type": "qwen_archive",
  "path": "artifacts/my-model/qwen-dense.snnq",
  "sha256": "WHOLE_ARCHIVE_SHA256",
  "identity_path": "artifacts/my-model/identity.json"
}
```

The identity file is an explicit flat JSON object containing `model_id`,
immutable `revision`, `source_checkpoint_sha256`, `config_sha256`, and
`tokenizer_fingerprint_sha256`. Supply the verified values from the conversion
manifest; for sharded sources use the converter's aggregate checkpoint identity.
`--qwen-identity` can override the identity path. Profile/tokenizer/archive
geometry and identities must agree. There is no automatic reshaping, truncation
or reuse of 0.6B weights for a larger architecture.

Support for configurable dense Qwen-compatible dimensions does not imply
support for MoE routing, quantized inference, tensor parallelism, distributed
training, offloading or every future Qwen architecture. Those require separate
implementations and quality/performance validation.

## Verification

```bash
python3 -m unittest discover -s tests/python -p test_temporal_llm.py -v
```

The frontend tests cover resource arithmetic against the real source inventory,
arbitrary larger geometry, tokenizer/model ID boundaries, corpus masking and
separation, calibrated/trained checkpoint resume, and protocol command binding.
C++ library/runner tests establish causal full/cached execution, temporal
encoding and serialization; a real-weight calibration/generation run is a
separate integration measurement. Record its exact settings and results rather
than inferring pretrained quality from the small protocol tests.
