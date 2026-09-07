# Temporal spiking Qwen experiment — 7 September 2026

Implemented in both repositories on `ReturmToBasics`. Historical measurements
are labeled below. The selected 16× development and holdout runs used clean
builds at snnbase `5783b00` and snnbaseExperiments `4d23ec6`.

The experiment loads the pinned Qwen3-0.6B-Base weights into `snnbase::language::Decoder`, generates from the full tokenizer vocabulary, calibrates hard signed temporal events, and supports checkpointed causal training. Model dimensions and output-head tying are configurable; verified sharded BF16 archives and an alternate untied architecture are covered by tests.

This is a hybrid spiking LLM. Attention, normalization, residual paths and matrix projections remain dense. The temporal code uses synchronized binary event windows with decreasing thresholds and optional previous-token activity. It does not implement local STDP, dendritic coincidence, biological polarity constraints or emergent synchronization in the LLM decoder.

## Ready to run

The compiled executable is `/home/dean/repos/snnbaseExperiments/build-temporal-llm/chatbot_experiment`. The calibrated checkpoint and matching sidecar are under `/home/dean/repos/snnbaseExperiments/artifacts/temporal_llm/`.

CPU command validated with the saved model:

```bash
python3 /home/dean/repos/snnbaseExperiments/tools/temporal_llm.py \
  --checkpoint /home/dean/repos/snnbaseExperiments/artifacts/temporal_llm/qwen3-0.6b-calibrated-h16.pt \
  --weights-only --device cpu --prompt "The capital of France is" --max-new-tokens 24
```

To use the RTX 3050 from your normal terminal:

```bash
python3 /home/dean/repos/snnbaseExperiments/tools/temporal_llm.py \
  --checkpoint /home/dean/repos/snnbaseExperiments/artifacts/temporal_llm/qwen3-0.6b-calibrated-h16.pt \
  --weights-only --device cuda --chat
```

`--weights-only` loads model parameters and calibration on the selected device, without restoring CPU optimizer/RNG state. Exact training resume remains available without this flag. The original entry point also accepts `tools/temporal_chat.py --llm`; its reservoir baseline remains available.

The RTX 3050 is present with 8 GiB VRAM and driver 580.173.02. The selected
checkpoint generated ` Paris. The capital of Germany is Berlin...` from the
prompt above. Five retained 32-token measurements after two warmups had a
median 9.265 tokens/second, 3.454-second core latency, and 109.1 ms to first
token. All five token sequences were identical. This is a bounded runtime
measurement rather than a universal speed or energy result.

## Measured language behavior

The source is [Qwen/Qwen3-0.6B-Base at the pinned revision](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd). It has 596,049,920 imported dense parameters; runtime neuron parameters bring the total to 596,279,296. The model vocabulary has 151,936 rows and sampling is restricted to the tokenizer's 151,665 decodable tokens.

Calibration used 32 authored passages containing 730 tokens. Evaluation used 16 different authored passages with 345 predicted targets. **These development passages informed calibration headroom selection; the measurements are not an independent final benchmark.** Qwen weights were imported and calibrated, with no Qwen fine-tuning.

| Variant | Token NLL | Perplexity | Token accuracy |
|---|---:|---:|---:|
| Dense Qwen reference, CUDA | 2.358726 | 10.5775 | 51.30% |
| Initial narrow range, 16 ticks | 2.858804 | 17.4407 | 42.03% |
| 4.04× range, 16 ticks | 2.402151 | 11.0469 | 48.41% |
| 8.08× range, 16 ticks | 2.370564 | 10.7034 | 49.86% |
| **Selected 16.16× range, 16 ticks** | **2.362769** | **10.6203** | **51.30%** |
| 32.32× range, 16 ticks | 2.364560 | 10.6394 | 51.01% |

Sixteen ticks remain the default because an earlier 20-tick comparison produced
little improvement. The selected range reduced the temporal-minus-dense NLL gap
from 0.043425 to 0.004043 and matched dense token accuracy and all 16 selected
final-position predictions. Mean hard-event activity was 0.2878. Calibration is
frozen during inference; the forward reconstruction contains emitted events
only, with no analog remainder at the encoder sites.

Layer telemetry explained the range choice. With 4× headroom, development
feed-forward sites clipped 3,700 of 31,051,776 values and reached 14.95 times
the calibrated scale. At 16×, clipping fell to 173 values (0.000557%) and the
aggregate absolute reconstruction error was 0.2077% of absolute input. The 32×
candidate clipped only 23 values, but its coarser pulse resolution raised that
error to 0.3818% and slightly worsened NLL.

Prompt: `The capital of France is`

Generated continuation:

> Paris. The capital of Spain is Madrid. The capital of Italy is Rome. The capital of Germany is Berlin. The

One CPU generation measurement produced 24 tokens in 6.29 seconds (3.82 tokens/second). This is a single runtime observation, not a controlled speed or energy benchmark. The base model has not been instruction-tuned by this experiment; general assistant quality is not established.

A subsequent controlled RTX 3050 comparison used the same frozen checkpoint,
prompt and CUDA environment. Each binary ran two excluded warmups followed by
five retained 32-token generations in one loaded process. All original and
fused runs produced identical token IDs. The fused CUDA encoder also reproduced
the earlier 4× checkpoint's original CUDA development metrics exactly: NLL
2.402142, perplexity 11.046817, token accuracy 48.41% and mean event activity
0.349393.

| Warm CUDA measurement | Original eager encoder | Fused encoder | Change |
|---|---:|---:|---:|
| Median generation rate | 2.229 tokens/s | 9.580 tokens/s | 4.30× |
| Median total latency, 32 tokens | 14.356 s | 3.340 s | −76.7% |
| Median time to first token | 449.3 ms | 105.2 ms | −76.6% |

The benchmark measures this model, prompt and RTX 3050 rather than universal
throughput. CUDA parity tests additionally cover pulse boundaries, four dtypes,
strided tensors, gradient/fallback paths, non-default streams and cached decoder
execution.

## Correctness and training

- The updated Python suite ran 128 tests: 126 passed and 2 optional tests skipped.
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

After selecting and freezing the settings above, the existing separate set of
16 short passages in `configs/temporal_llm/holdout.jsonl` was evaluated once
with the new 16× checkpoint. These introduce topics such as chess, ceramics,
weaving and postal sorting. The checkpoint was not recalibrated or trained, and
no parameters or prompts were tuned using these results.

The historical dense source evaluation and the new CUDA temporal evaluation
cover the same 417 causal targets. Their ordered token-record hashes and
source-archive identities match. Every holdout record hash differs from both
the earlier calibration and development sets.

| Frozen variant | Token NLL | Perplexity | Token accuracy |
|---|---:|---:|---:|
| Dense Qwen source | 3.011883 | 20.3256 | 41.73% |
| Earlier temporal conversion, 4.04× range, 16 ticks | 3.169769 | 23.8020 | 40.05% |
| **Temporal conversion, 16.16× range, 16 ticks** | **3.048154** | **21.0764** | **42.69%** |

The selected conversion increases NLL by **0.036271** and perplexity by
**3.69%**; token accuracy is **0.96 percentage points higher** on this small
sample. It removes 77.0% of the earlier NLL gap. Mean hard-event activity was
0.28843. Holdout telemetry found 179 clipped feed-forward values among
37,244,928 (0.000481%). This separate check still shows a measurable
probabilistic conversion cost. Sixteen authored passages remain a small quality
check, not a broad language-model benchmark.

The generations below are the earlier 4× checkpoint's historical outputs; the
holdout was not reused for selecting or tuning the new checkpoint.

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
