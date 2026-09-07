# Temporal spiking Qwen experiment — 7 September 2026

Implemented in both repositories on `ReturmToBasics`. The measurements below
were recorded during development, before the final implementation commits.

The experiment loads the pinned Qwen3-0.6B-Base weights into `snnbase::language::Decoder`, generates from the full tokenizer vocabulary, calibrates hard signed temporal events, and supports checkpointed causal training. Model dimensions and output-head tying are configurable; verified sharded BF16 archives and an alternate untied architecture are covered by tests.

This is a hybrid spiking LLM. Attention, normalization, residual paths and matrix projections remain dense. The temporal code uses synchronized binary event windows with decreasing thresholds and optional previous-token activity. It does not implement local STDP, dendritic coincidence, biological polarity constraints or emergent synchronization in the LLM decoder.

## Ready to run

The compiled executable is `/home/dean/repos/snnbaseExperiments/build-temporal-llm/chatbot_experiment`. The calibrated checkpoint and matching sidecar are under `/home/dean/repos/snnbaseExperiments/artifacts/temporal_llm/`.

CPU command validated with the saved model:

```bash
python3 /home/dean/repos/snnbaseExperiments/tools/temporal_llm.py \
  --checkpoint /home/dean/repos/snnbaseExperiments/artifacts/temporal_llm/qwen3-0.6b-calibrated.pt \
  --weights-only --device cpu --prompt "The capital of France is" --max-new-tokens 24
```

To use the RTX 3050 from your normal terminal:

```bash
python3 /home/dean/repos/snnbaseExperiments/tools/temporal_llm.py \
  --checkpoint /home/dean/repos/snnbaseExperiments/artifacts/temporal_llm/qwen3-0.6b-calibrated.pt \
  --weights-only --device cuda --chat
```

`--weights-only` loads model parameters and calibration on the selected device, without restoring CPU optimizer/RNG state. Exact training resume remains available without this flag. The original entry point also accepts `tools/temporal_chat.py --llm`; its reservoir baseline remains available.

The RTX 3050 is present with 8 GiB VRAM and driver 580.173.02. CUDA execution could not be tested through this task's sandbox because NVIDIA device nodes were unavailable. CPU execution and weights-only checkpoint loading are validated. No driver installation was needed.

## Measured language behavior

The source is [Qwen/Qwen3-0.6B-Base at the pinned revision](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd). It has 596,049,920 imported dense parameters; runtime neuron parameters bring the total to 596,279,296. The model vocabulary has 151,936 rows and sampling is restricted to the tokenizer's 151,665 decodable tokens.

Calibration used 32 authored passages containing 730 tokens. Evaluation used 16 different authored passages with 345 predicted targets. **These development passages informed calibration headroom selection; the measurements are not an independent final benchmark.** Qwen weights were imported and calibrated, with no Qwen fine-tuning.

| Variant | Token NLL | Perplexity | Token accuracy |
|---|---:|---:|---:|
| Dense Qwen reference | 2.358701 | 10.5772 | 51.30% |
| Initial narrow range, 16 ticks | 2.858804 | 17.4407 | 42.03% |
| Final 4.04× range, 16 ticks | 2.402092 | 11.0463 | 48.41% |
| 4.04× range, 20 ticks | 2.401190 | 11.0363 | 48.41% |

Sixteen ticks remain the default because the additional four ticks produced little improvement. Final mean hard-event activity was 0.3494 on the development text. Calibration is frozen during inference; the forward reconstruction contains emitted events only, with no analog remainder at the encoder sites.

Prompt: `The capital of France is`

Generated continuation:

> Paris. The capital of Spain is Madrid. The capital of Italy is Rome. The capital of Germany is Berlin. The

One CPU generation measurement produced 24 tokens in 6.29 seconds (3.82 tokens/second). This is a single runtime observation, not a controlled speed or energy benchmark. The base model has not been instruction-tuned by this experiment; general assistant quality is not established.

## Correctness and training

- The Python suite ran 125 tests successfully, with 2 optional tests skipped.
- Both library language test targets passed, including legacy behavior and the new temporal decoder tests.
- Runner and Qwen import C++ test targets passed, including calibration, training, exact resume, weights-only loading, strict identities and a different untied architecture imported from shards.
- Temporal tests check hard-event reconstruction, silence without an analog bypass, calibration headroom, causal gradients, future-token isolation, independent stream caches, full/chunk/token equivalence, compact state storage and checkpoint versions.
- Inference omits surrogate-gradient calculations while matching the hard forward output. Training retains the surrogate path.
- A small model using the full Qwen vocabulary trained and resumed successfully. Its language quality remains poor after a few updates; those runs validate training mechanics.

## Scaling and biological limits

Generation uses chunked prefill and projects only the final position to vocabulary logits. At context 512, this changes the logits allocation from 311,164,928 bytes to 607,744 bytes. Configuration-only `--plan` estimates weights, optimizer state, KV cache and attention workspace before allocating a larger model.

The 0.6B model needs approximately 2.22 GiB for FP32 dense weights and 112 MiB for its batch-one, context-512 KV cache, before workspace. Full FP32 AdamW needs at least 8.88 GiB for weights, gradients and moments alone, exceeding the RTX 3050's 8 GiB. Use the small profile for the training experiment on this card; this implementation does not provide weight quantization, offload, tensor parallelism or parameter-efficient fine-tuning.

Larger dense Qwen geometries, untied heads and sharded conversion are implemented, but a larger pretrained model has not been validated. Unsupported MoE, sliding attention and scaled RoPE are rejected explicitly.

Optional cross-token neuronal recurrence is covered by correctness tests. The final Qwen conversion uses `temporal_decay=0` to preserve pretrained behavior; nonzero recurrence needs its own quality evaluation and training. Dense attention supplies contextual interaction, and shared event windows supply engineered timing. This is a conversion and training foundation, not a complete abstraction of all biological learning and synchronization.

Implementation and reproduction instructions are in `/home/dean/repos/snnbase/docs/TEMPORAL_LANGUAGE.md` and `/home/dean/repos/snnbaseExperiments/docs/TEMPORAL_LLM.md`. Source digests, exact model identity and measured metrics are included in the accompanying JSON report.

## Independent frozen-checkpoint check

After selecting and freezing the settings above, a separate set of 16 short
passages was authored in `configs/temporal_llm/holdout.jsonl`. These introduce
topics such as chess, ceramics, weaving and postal sorting. The checkpoint was
not recalibrated or trained, and no parameters or prompts were tuned using
these results. The original development measurements above remain historical
evidence of configuration selection.

Both models evaluated the same 417 causal targets with four CPU threads. Their
ordered token-record hashes and source-archive identities match. Every holdout
record hash differs from both the earlier calibration and development sets.

| Frozen variant | Token NLL | Perplexity | Token accuracy |
|---|---:|---:|---:|
| Dense Qwen source | 3.011883 | 20.3256 | 41.73% |
| Temporal conversion, 4.04× range, 16 ticks | 3.169769 | 23.8020 | 40.05% |

The conversion increases NLL by **0.157886** and perplexity by **17.10%**, while
reducing token accuracy by **1.68 percentage points**. Mean hard-event activity
was 0.35008. This separate check shows a measurable conversion cost beyond the
earlier development set. Sixteen authored passages remain a small quality check,
not a broad language-model benchmark.

Three non-punctuation prefixes were selected before inspecting the holdout
results. Each model generated at most 12 new tokens with greedy decoding. All
outputs are included below, including incorrect continuations.

| Prefix | Dense source continuation | Temporal continuation |
|---|---|---|
| `A chessboard has sixty-four squares arranged in` | ` two rows and thirty-two columns. How many squares of the` | ` a $4\times 4$ grid. How many` |
| `A potter fires clay in a` | ` horizontal direction. The clay travels 100 m before` | ` horizontal direction with a speed of 10 m/s.` |
| `A woven fabric is made by interlacing` | ` two fabrics of the same width. The width of the woven` | ` two layers of cloth. The first layer is 10` |

Both the dense Base model and the temporal conversion fail these common-knowledge
continuations. The temporal chess answer also contradicts the stated square
count. These results reinforce that this is a completion/conversion experiment,
not a reliable factual or instruction-following assistant. The successful
capital-city example should not be interpreted as general competence.

Checkpoint SHA-256, unchanged after the independent runs:
`30ab2ccfbb25e77cb60ca8ff1f281bce2beeefc1ca7b1cd425659d221ad514c3`.
Holdout corpus SHA-256:
`f146c12118ed5842540a27df634827142f00c506675a21e5c6229f58ff9ff14f`.
The numerical checks used executable SHA-256
`8367cdee7470bdb6d2d0d1515833147778c03cfdabbe9883d50cf1cb499fc4ce`.
Subsequent shared checkpoint-validation changes do not change valid-model
forward calculations.

Complete local evidence is stored under `artifacts/temporal_llm/` as
`independent-holdout-spiking.json`, `independent-holdout-ann.json`,
`independent-holdout-generations.json` and `independent-holdout-summary.json`.
