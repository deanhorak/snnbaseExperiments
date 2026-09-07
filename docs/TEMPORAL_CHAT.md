# CPU temporal spiking token experiment

`temporal_chat` is a small trained text-generation experiment built on
`snnbase::population::Network`. A pinned Qwen3 tokenizer converts a user prompt
to token IDs. Those IDs drive temporal spike patterns in a recurrent network;
its learned readout predicts output tokens that are fed back through the same
spiking circuit and decoded into text. It uses no pretrained model weights,
LibTorch, attention layer, language-model API, response lookup, or retrieval.

This is a **six-intent toy proof of concept**. Its 36 training prompts and 12
held-out prompts ask about the colors of tomato/sky/grass and the sounds of
cat/dog/bird. Held-out prompts have new wording for the **same six subjects and
known answers**. They do not test new factual knowledge, unrestricted dialogue,
instruction following, or general language-model quality. There are only 12
output token IDs in the supplied corpus. Unsupported prompts still generate
from that limited output vocabulary; there is no trained uncertainty detector.
The optional chat loop runs independent single-turn prompts and resets neural
state between turns.

## Build and run

From the repository root, with the updated sibling `../snnbase` checkout:

```sh
cmake -S . -B build-temporal-chat -DCMAKE_BUILD_TYPE=Release \
  -DSNNBASE_EXPERIMENTS_ENABLE_TEMPORAL=OFF \
  -DSNNBASE_EXPERIMENTS_ENABLE_CHATBOT=OFF \
  -DSNNBASE_EXPERIMENTS_ENABLE_GUI=OFF
cmake --build build-temporal-chat --target temporal_chat temporal_chat_tests -j 8
python3 -m pip install -r experiments/temporal_chat/requirements.txt
python3 tools/temporal_chat.py --prompt "What is the color of grass?" \
  --report artifacts/temporal-chat/run.json
python3 tools/temporal_chat.py --chat
ctest --test-dir build-temporal-chat -R '^temporal_chat_' --output-on-failure
```

The new `SNNBASE_EXPERIMENTS_ENABLE_TEMPORAL_CHAT` option is enabled by default
and links only the CPU library. The separate older LibTorch chatbot and visual
experiments remain available with their existing flags and commands. Use
`--core /path/to/temporal_chat` for another build directory. Each invocation
trains once from the tiny corpus, freezes learning, then handles prompts with
the in-memory model. Checkpoint persistence is not implemented.

The default tokenizer directory is
`artifacts/chatbot/qwen3-0.6b-base`, reusing an existing local snapshot. If those
files are absent, download just the following four upstream files into a
directory of your choice, then pass `--tokenizer-dir /that/directory`:

- [tokenizer.json](https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/tokenizer.json)
- [tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/tokenizer_config.json)
- [vocab.json](https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/vocab.json)
- [merges.txt](https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/merges.txt)

The source is `Qwen/Qwen3-0.6B-Base` at immutable revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, distributed under
[Apache-2.0](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/LICENSE).
The frontend reuses `tools/qwen_contract.py` to verify the pinned SHA-256
fingerprint and individual tokenizer hashes before loading the local
`tokenizer.json` through Hugging Face's `tokenizers` library. It does not open
weight files, including any present in the existing snapshot, and never
downloads assets during execution or tests. Exact package versions are
recorded in the report; the requirements file pins the tested version.

Prompts use the experiment's own `User: {text}\nAssistant:` format, encoded
without automatic special tokens. This is not Qwen chat-template parity. The
response's independently encoded IDs end with upstream `<|im_end|>` ID 151645.
Special conversation delimiters embedded in user content are rejected. Only
generated output IDs, with special tokens omitted, are passed to the decoder.

## Network and learning

The network contains 353 neurons: 128 input, 192 excitatory reservoir, 32
inhibitory reservoir, and one shared pacemaker neuron. There are 5,216 sparse
synapses. A stable seeded hash maps each upstream token to 16 input neurons,
which emit a phase-coded pattern over six discrete steps. A projection carries
those spikes into one dendritic branch; delayed excitatory recurrent activity
and pacemaker spikes converge onto a second branch. Branches integrate signed
coincident input and emit abstract events at threshold. Somatic inhibitory
feedback, refractory periods, and spike-triggered threshold adaptation limit
activity. Shared pacemaker spikes provide a neuronal synchronization pathway;
this experiment does not measure biological phase locking or prove that
pacemaker input improves language performance.

Training has two explicitly separate stages:

1. **Local STDP warmup.** One pass over training prompts and teacher responses
   updates input-to-excitatory synapses using the library's delay-aligned pair
   STDP (`learning_rate=0.00005`, potentiation 1.0, depression 1.05). The
   projection preserves excitatory polarity. No held-out tokens participate.
   `set_projection_learning_enabled(..., false)` then freezes these weights
   while preserving signal transmission. All neural state resets before each
   example.
2. **Supervised local readout.** Fast and slow traces of excitatory/inhibitory
   reservoir spikes form 448 features plus a bias. **There are no direct input
   token or input-neuron features in the readout.** A 12-class categorical
   softmax produces token probabilities. The experiment updates each readout
   connection with `eta * presynaptic_trace * (teacher_target - probability)`.
   This is an engineered error-modulated local rule; categorical softmax and
   the externally supplied correct token are not asserted to be biological
   mechanisms. There is no backpropagation through the reservoir. Teacher
   forcing supplies previous response tokens during training; each target's
   features contain only its preceding tokens.

Because the reservoir is frozen and deterministic during stage 2, its training
spike features are computed once and replayed for 120 shuffled readout epochs.
This avoids rerunning the network for every epoch. `freeze()` prevents further
readout training. Inference resets only transient state, drives the entire
prompt through the network, chooses the most probable token, feeds that token
back into the network, and repeats until stop or the token limit. The readout
weights stay fixed. Token softmax and readout updates remain dense operations;
no fully event-driven, neuromorphic energy, or throughput claim is made.

## Evaluation and controls

The report includes teacher-forced cross-entropy and accuracy on assistant
tokens including stop, and separate autoregressive exact-match accuracy for
complete held-out answers including stop. Zero-readout/uniform and
training-only smoothed unigram baselines are evaluated on the same held-out
targets. The output vocabulary is built from training targets only. Vocabulary
membership is checked for held-out targets; unseen target tokens cause an
explicit error rather than a silent remapping. There is no held-out checkpoint
selection or early stopping. Architecture and training settings are fixed for
the reported run; this tiny held-out set has been inspected during development
and is a regression set, not an untouched research benchmark.

```sh
python3 tools/temporal_chat.py --evaluate-only --report artifacts/temporal-chat/default.json
python3 tools/temporal_chat.py --evaluate-only --no-stdp \
  --report artifacts/temporal-chat/no-stdp.json
python3 tools/temporal_chat.py --evaluate-only --lesion-reservoir \
  --report artifacts/temporal-chat/lesion.json
python3 tools/temporal_chat.py --evaluate-only --seed 7 \
  --report artifacts/temporal-chat/seed7.json
```

`--no-stdp` leaves input weights at their seeded initialization while training
the same supervised readout. `--lesion-reservoir` replaces all reservoir spike
features with zero in both training and inference, retaining only the learned
bias as a control. It does not claim to identify which recurrent connection or
biological feature contributes most. See the recorded results below for the
actual differences, rather than assuming STDP is necessary for this task.

## Bounds and tests

The C++ core uses a versioned ASCII integer protocol on stdin and emits JSON
on stdout. A startup header gives seed, epochs, stop ID, train/held-out counts,
STDP flag and lesion flag; bounded `PAIR` records follow. After one training
response, `GENERATE max_new_tokens prompt_length ids...` requests are accepted
until `QUIT`. Both sides validate lengths and IDs. The core rejects duplicate
training/held-out prompts, overlap between splits, misplaced stop IDs, invalid
integers, extra fields, oversize lines, and excessive training work. Backend
errors go to stderr with exit code 2. The frontend validates output vocabulary,
held-out target IDs, and generation stop consistency.

Limits: 128 training examples, 32 held-out examples, 128 prompt tokens, 32
response tokens, 512 output classes, 2,048 supervised training tokens, 300
epochs, 64 generated tokens, and 1.5 billion estimated readout coordinate
updates. Input lines are capped at 65,536 bytes. The Python wrapper caps corpus
files and response lines at 1 MiB and waits at most 120 seconds for a response.
These are proof-of-concept resource limits, not a scalable corpus pipeline.

Tests exercise history dependence with identical final input tokens,
deterministic reset, actual firing, learning improvement, two different
prompt-conditioned autoregressive replies, frozen weights, seeded training,
malformed protocol, split overlap, and early stop rejection. A real-tokenizer
test runs only when the verified local tokenizer and Python dependency are
available; it never substitutes a fake tokenizer for that integration test or
downloads missing assets. Reports preserve corpus and core hashes, tokenizer
provenance, settings, all held-out generated/expected IDs and text.

The small `corpus.json` was authored for this experiment and is released under
the repository's license. It contains no scraped conversations or external
training corpus.

## Recorded local run

On 2026-09-07, the CPU Release build (GCC 13.3, `tokenizers==0.22.2`, 120
readout epochs) produced the following results on the 44 assistant target
tokens in the 12 held-out prompts. All runs used the same final executable
SHA-256 `336e7bd2214e7aadb28b46ce6283f0119d87356f854ab2da77a50a459f2e0607`.

| Configuration | Seed | Held-out token loss | Token accuracy | Exact replies |
| --- | ---: | ---: | ---: | ---: |
| STDP warmup + supervised readout | 42 | 0.004712 | 100% | 12/12 |
| No STDP + supervised readout | 42 | 0.063700 | 97.73% | 11/12 |
| STDP warmup + supervised readout | 7 | 0.004469 | 100% | 12/12 |
| No STDP + supervised readout | 7 | 0.008381 | 100% | 12/12 |
| STDP warmup + supervised readout | 91 | 0.005510 | 100% | 12/12 |
| No STDP + supervised readout | 91 | 0.015031 | 100% | 12/12 |
| Reservoir-feature lesion + supervised bias | 42 | 2.131020 | 27.27% | 0/12 |
| Untrained zero readout (uniform) | 42 | 2.484907 | 27.27% | — |
| Training-only smoothed unigram | 42 | 2.113743 | 27.27% | — |

The default prompt `What is the color of grass?` generated `green.`. STDP
changed input synapses (sum of absolute changes 3.292160 for seed 42) and
reduced loss in these three paired runs. The no-STDP controls also learned
almost every reply: this does **not** establish that STDP is necessary for
text generation, or that its small-corpus benefit transfers to natural
language modelling. Zeroing all reservoir features removed prompt and
sequence information and prevented correct complete replies. This supports
dependence on neural activity, without isolating a particular dendritic,
inhibitory, or synchronization mechanism.

Both CTest targets passed, including the real-tokenizer integration and seven
Python validation/protocol/integration cases. These are local development
results on the bundled six-intent regression corpus, not a benchmark of
general conversational ability.
