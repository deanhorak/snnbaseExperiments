# Spaun behavioral benchmark status

Date: 2026-07-18

Status: **partial evidence; behavioral equivalence is not supported**

This page records the durable A1 result, the completed A2--A5 protocol runs,
and the architecture used by the protocol runner. The memory sweep and its
three-seed confirmation are recorded separately in
[SPAUN_MEMORY_PARAMETER_SWEEP.md](SPAUN_MEMORY_PARAMETER_SWEEP.md).

This record compares the current snnbase experiment with the behavioral
protocols reported for Spaun. It does not treat the eight canonical examples as
an accuracy dataset, and it does not apply a universal 95% target to all tasks.
The original publication specifies a different comparison for each task.

Primary sources:

- [Eliasmith et al., “A Large-Scale Model of the Functioning Brain,” Science
  338, 2012](https://compneuro.uwaterloo.ca/files/publications/eliasmith.2012.pdf)
- [Original Spaun supplemental
  material](https://cs.uwaterloo.ca/~jhoey/teaching/cogsci600/papers/Spaun_Supplemental_Material_manualcitations.pdf)
- [Computational Neuroscience Research Group publication
  record](https://compneuro.uwaterloo.ca/publications/eliasmith2012.html)
- [Official Spaun 2.0 source](https://github.com/xchoo/spaun2.0)
- [Official Spaun videos](https://xchoo.github.io/spaun2.0/videos.html)

Across tasks, the source model used 28×28 visual inputs, presented each symbol
for 150 ms followed by a 150 ms blank, kept the model fixed between tasks, and
used the arm as its only behavioral output. Those are published protocol
constraints, not accuracy thresholds.

## Current outcome

| Question | Current answer |
|---|---|
| Does the current canonical A0--A7 smoke battery pass every fixed example? | No: 6/8 at the behavioral defaults; A3/A5 errors are expected from the tuned serial-position profile |
| Is there registered held-out/protocol evidence? | Yes: registered A1 and full executable A2--A5 schedules |
| Does A1 meet the project gate? | Yes: 98.35% exceeds the 94% project minimum |
| Which behavioral gates pass in the consolidated report? | A1--A5 pass; A0, A6, and A7 are missing; none fail |
| Has behavior been attributed to the spiking pathways with protocol ablations? | No |
| Is behavioral equivalence supported? | No |
| Is full Spaun equivalence supported? | No |

The consolidated seed-42 benchmark therefore correctly reports:

```text
gates passed=5 failed=0 missing=3
behavioral_equivalence_supported=false
spiking_mechanism_supported=false
full_equivalence_supported=false
```

## Registered A1 result

The A1 benchmark trains `snnbase::spiking_conv::Classifier` on the official
60,000-image MNIST training set and evaluates the untouched 10,000-image
official test set. The recorded checkpoint uses this structure:

```text
28x28 grayscale
  -> 12-channel 5x5 convolution, stride 2, spiking units
  -> 24-channel 3x3 convolution, stride 2, spiking units
  -> dense 10-class readout
  -> 8 spike-rate timesteps
```

Recorded evaluation:

| Field | Value |
|---|---:|
| Training samples | 60,000 |
| Held-out test samples | 10,000 |
| Trainable parameters | 8,938 |
| Correct | 9,835 |
| Accuracy | **98.35%** |
| Test loss | 0.05300 |
| Project gate | at least 94% |
| Bootstrap samples | 3,000 |
| Project-run 95% interval | 98.10%--98.59% |

An evaluate-only verification of `spaun-a1-e5.ckpt` produced:

```text
benchmark=Spaun-A1-registered-MNIST train_samples=60000 test_samples=10000
parameters=8938 conv1=12x5x5/2 conv2=24x3x3/2 time_steps=8
test_correct=9835/10000 test_accuracy=0.9835 test_loss=0.05300
```

SHA-256 provenance:

| Artifact | SHA-256 |
|---|---|
| MNIST train images | `ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db` |
| MNIST train labels | `65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5` |
| MNIST test images | `0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7` |
| MNIST test labels | `ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2` |
| A1 checkpoint | `7953d733a96bdf16bcec4793aa42d212a2ada8c4802dc203b3db998573149615` |

This is stronger than the **94% Spaun A1 model result** reported in the primary
paper and close to the paper's approximate 98% human comparator. It establishes
the project's A1 acceptance gate only. It does not establish whole-model
equivalence, and the paper does not state an exact A1 sample count or split that
would make this a perfectly matched replication.

The trained 28×28 classifier is also not yet the front end used by
`spaun_experiment` or `spaun_gui`; those use the native spike-causal 5×7 visual
matcher. Consequently, this result must not be described as end-to-end
perception-to-arm performance by the integrated model.

For durable provenance, a published result should register an immutable dataset
identifier, split identifier, checkpoint identifier/hash, and the assertion
that reported images were held out. The runner refuses an A1 registration when
any of those fields is absent.

## Consolidated A1--A5 protocol result

The final report used seed 42 and 3,000 bootstrap samples. A0, A6, and A7 were
selected as well, but remain explicitly unavailable rather than being scored
with invented schedules.

| Task | Protocol result | Gate |
|---|---|---|
| A1 | 9,835/10,000 = **98.35%**, 95% CI [98.10%, 98.59%] | Passed >=94% |
| A2 | Best-arm rates by 20-trial block: 85%, 35%, 35%; final-five rates: 100%, 100%, 100% | Passed the recorded reversal-adaptation review |
| A3 | Exact sequences 35.00%; item profile ranges from perfect endpoints to lower middle positions as length increases | Passed the per-length primacy/recency review |
| A4 | 25/25 exact; **420 ms/item**, 95% CI [420, 420] | Passed overlap with Spaun's 419 +/- 10 ms/item band |
| A5 | 90/140 = 64.29%; P/K profiles show similar primacy and recency | Passed the recorded profile review |

The A3/A5 exact rates are not failed 95% targets. Raising memory retention to
the accuracy-first region produced 85--91% A3 exact and 94--97% A5 exact, but
made the short-list serial-position curves nearly perfect and failed the A3
behavioral gate. The selected default instead passed both profile gates on
seeds 42, 43, and 44. See the parameter-sweep record for the full tradeoff.

The instantiated integrated model reports **7,018 LIF neurons and 1,489,322
stored synaptic coefficients**. These are scaled implementation counts, not a
claim to match Spaun's approximately 2.5 million neurons.

## Four kinds of benchmark information

The implementation uses separate types so that a convenient local threshold
cannot be misreported as something the paper required.

| Kind | Meaning | Example |
|---|---|---|
| Published numeric reference | A number reported for Spaun or a human comparator | A1 Spaun accuracy = 0.94 |
| Published protocol parameter | A condition needed to reproduce the experiment | A3 uses 40 runs at each list length 4--7 |
| Published qualitative claim | A shape or behavior without a numeric cutoff | A3 exhibits primacy and recency |
| Project-defined gate | This project's explicit pass/fail decision | A1 accuracy must be at least 0.94 |

There is no published universal “95%+” goal for the task battery. In particular,
Spaun's raw A7 accuracy was 30/40 = 75%, with a reported 95% interval of
60%--88%. Its separately reported 88% value is chance-adjusted and must not be
substituted for raw accuracy.

## Published task protocols and current gaps

| Task | Published comparison | Current evidence | What remains unresolved |
|---|---|---|---|
| A0 copy drawing | 20 additional held-out examples; recognizable reproduction of input style; no numeric threshold | Canonical input state is recalled from its recurrent slot and gated directly through the spiking output route; vector-stroke and GUI geometry tests exist | Present held-out handwriting, preserve its style, replace host-selected gating/stroke planning, and conduct blinded qualitative review |
| A1 recognition | Held-out handwritten digits; Spaun 94%, human comparison about 98% | Registered MNIST test: 9,835/10,000 = 98.35% | Integrate learned 28×28 latent spikes into the full model and match otherwise unstated source split details |
| A2 reinforcement learning | One 60-trial trajectory; 20-trial blocks; best-arm probabilities 0.72 vs 0.12; five-trial moving choice probability | Full run completed; the runner supplies all three blocks externally and `RewardModulatedSelector` adapts from sampled reward; the final five choices select the best arm in every block | Compare the entire moving-choice trajectory more directly; expose one external decision/reward exchange at a time |
| A3 serial working memory | Lengths 4--7, 40 runs each, 22 position means; 17/22 human means inside Spaun 95% intervals | Full run and three-seed sweep completed; direct recurrent-state output shows robust primacy/recency | A source-compatible human-data comparison is still needed; host selects output gates |
| A4 counting | Counts 1--5, five simulated individuals; Spaun response-time slope 419 +/- 10 ms/item, human 344 +/- 135 | Full 25-trial run: 25/25 exact and 420 ms/item from the delayed successor chain | Host clock/stage selection and multi-digit formatting remain; validate the onset landmark independently |
| A5 question answering | Seven-item lists; 10 simulated individuals × 7 positions × P/K = 140; similar P/K primacy/recency profiles | Full 140-trial run passes the recorded P/K and serial-position review; P queries route directly | Implement neural query/gate selection, especially K queries, and compare with human data when available |
| A6 rapid variable creation | Infer a suffix/removal rule and answer a new example; response period begins 150 ms after examples | Canonical transformation smoke example | Use held-out transformations/operands, neural rule creation, and source-compatible timing; no aggregate source threshold exists |
| A7 fluid reasoning | 40 runs; raw 30/40 = 0.75, 95% interval 0.60--0.88; adjusted 0.88; human 0.89 | Canonical progression smoke example | Run a matched 40-item set with a neural solver and compare raw accuracy/interval without conflating adjusted accuracy |

The A2 probability blocks are environmental inputs, not an answer oracle. The
selector receives a cue, emits an action, and receives only the reward sampled
for that action; its reward-modulated projection carries adaptation across the
60 decisions. The public model call nevertheless contains the whole
choice/reward loop instead of exposing one embodied exchange at a time.

For A4, response onset is measured from the blank after the final query symbol
to the first pen-positioning frame. Each requested increment must traverse one
delayed semantic-successor projection. Host code controls how many stages are
clocked and selects the corresponding stage, but it does not add the operands.
Consequently, the configured projection delay—not a synthetic host timing
formula—is the source of the per-item slope.

Confidence intervals intended to reproduce the paper use 3,000 bootstrap
resamples. Schedules whose exact stimuli were not published are marked as
project-generated rather than “fully specified by publication.”

## Project acceptance gates

These gates are transparent project decisions. Only their reference values or
protocol motivation come from the source.

| Task | Required project gate | Current status |
|---|---|---|
| A0 | Qualitative review of recognizable held-out copies | Missing |
| A1 | Held-out recognition accuracy >= 0.94 | **Passed: 0.9835** |
| A2 | Qualitative blockwise adaptation review | **Passed** |
| A3 | Qualitative serial-position/profile agreement | **Passed** on the full run and seeds 42--44 |
| A4 | Estimated interval overlaps 409--429 ms/item | **Passed: 420 ms/item** |
| A5 | Qualitative P/K profile and primacy/recency agreement | **Passed**; P is direct, K remains host-computed |
| A6 | Generalize an inferred rule to held-out operands | Missing |
| A7 | Raw estimate/interval agrees with the reported 0.60--0.88 raw band | Missing |

A canonical correct answer cannot fill one of these rows unless it was produced
under the corresponding protocol and metric. A quick/subsampled run remains a
smoke test even when its score is high.

## Causal substrate status

The current implementation has real lesion points for:

- retina to visual matching;
- visual matching to symbol encoding;
- semantic encoding to working memory;
- working-memory recurrence;
- working memory to transformation, including direct recall and counting
  entry/output gates;
- reward plasticity;
- transformation/action to decoding;
- decoding to motor;
- motor to arm.

Component tests show that disabling these projections clears queued drive and
removes the corresponding recall, action learning, motor digit, or drawing.
Behavioral correctness now requires observable arm ink for every nonempty
expected response; a motor-to-arm lesion preserves the decoded cognitive
answer for diagnostics but marks the trial incorrect.
That is necessary but not sufficient. The behavioral runner does not yet
produce matched intact-versus-lesioned protocol distributions, so it leaves
`spiking_mechanism_supported` false.

A0, A1, and A3 no longer use `Model::solve()` to construct response digits:
their selected recurrent item states route directly through the spiking
semantic transformation, action, decoding, and motor populations. A5 position
queries use the same direct route after host code selects the queried slot. A4
seeds and advances a delayed spiking successor chain and routes its selected
stage through that output path. A2 uses the reward-plastic selector with
externally supplied contingencies.

The remaining boundary is narrower but still scientifically important. Host
code parses the recognized stream, decodes state for control, selects recall
and counting gates, clocks the A4 chain, formats multi-digit counts, owns the
60-step A2 interaction loop and final utility readout, and constructs plans for
A5 kind queries, A6, and A7. Stroke planning and inverse kinematics are also
host-side. The runner does not yet produce matched intact-versus-lesioned
behavioral distributions, so direct-route component tests alone cannot set
`spiking_mechanism_supported=true` or establish Spaun equivalence.

## Reproduction commands

Train and evaluate A1:

```bash
./build/spaun_vision_benchmark \
  --data-dir data/mnist \
  --epochs 5 \
  --batch-size 64 \
  --time-steps 8 \
  --learning-rate 0.001 \
  --seed 42 \
  --save-checkpoint spaun-a1-e5.ckpt \
  --minimum-accuracy 0.94
```

Evaluate the saved artifact without fitting:

```bash
./build/spaun_vision_benchmark \
  --data-dir data/mnist \
  --evaluate-only \
  --load-checkpoint spaun-a1-e5.ckpt \
  --minimum-accuracy 0.94
```

Register that result in an A1-only behavioral report:

```bash
./build/spaun_benchmark \
  --task A1 \
  --bootstrap 3000 \
  --a1-correct 9835 \
  --a1-total 10000 \
  --a1-dataset MNIST-IDX \
  --a1-split official-test-10000 \
  --a1-artifact spaun-a1-e5.ckpt \
  --a1-held-out \
  --json spaun-a1-report.json
```

Reproduce the consolidated report without promoting it to an equivalence
claim (the identifiers should be replaced by full immutable hashes in an
archived report):

```bash
./build/spaun_benchmark \
  --task all \
  --seed 42 \
  --bootstrap 3000 \
  --a1-correct 9835 \
  --a1-total 10000 \
  --a1-dataset MNIST-IDX-sha256-ba891046e6505d7a \
  --a1-split official-test-10000-sha256-0fa7898d509279e4 \
  --a1-artifact sha256-7953d733a96bdf16 \
  --a1-held-out \
  --json spaun-behavioral-final.json
```

`--require-equivalence` is intentionally expected to fail until every required
behavioral gate has protocol evidence and the causal ablation battery supports
the spiking mechanism.
