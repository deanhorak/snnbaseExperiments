# Experiment network structures

This document diagrams every experiment network structure implemented by the
experiments project. Dimensions are the defaults in the experiment entry
points. `C` means class count, `T` means simulation timesteps, and `P` means
readout population per class.

## Experiment map

```mermaid
flowchart LR
  M[mnist_experiment] --> MP[MNIST prototype SNN]
  EB[emnist_battery<br/>baseline] --> EP[EMNIST prototype SNN]
  ES[emnist_battery<br/>--structured] --> EE[EMNIST structured experts]
  ED[emnist_deep] --> ER[EMNIST temporal SEW-ResNet]
  CF[src/cifar10.cpp<br/>support classifier] --> FB[CIFAR feature-bank classifier]
  CS[cifar10_experiment] --> SC[CIFAR shallow spiking CNN]
  CD[cifar10_deep] --> CR[CIFAR temporal SEW-ResNet]
  SP[spaun_experiment / spaun_gui] --> SPAUN[Scaled Spaun migration]
```

The classifier in `src/cifar10.cpp` is included because it is a complete
experiment-support implementation, although the current `cifar10_experiment`
entry point instantiates `snnbase::spiking_conv::Classifier` instead.

## Shared spiking building blocks

### Prototype neuron

The MNIST and non-deep EMNIST models use a bank of `snnbase::Neuron` objects.
Each neuron stores bounded spike-event prototypes and scores a new event by
similarity to its stored history. Reinforcement, novelty admission, replay,
and STDP relevance update that history.

```mermaid
flowchart LR
  E[SpikeEvent bit vector] --> S[Compare with stored prototypes]
  H[(Prototype history)] --> S
  S --> A[Maximum prototype similarity]
  A --> O[Neuron score]
  R[Reward or punishment] --> U[Novelty / STDP / replay update]
  U --> H
```

### Temporal spiking convolution

The deep EMNIST and deep CIFAR models use the same temporal convolution unit.
The convolution computes synaptic current; batch normalization conditions that
current; and a learnable leaky integrate-and-fire neuron produces spikes at
every timestep. Therefore these convolution stages are spiking layers, not
ordinary ANN convolution blocks.

```mermaid
flowchart LR
  X[Spike tensor<br/>T x B x Cin x H x W] --> C[Conv2d]
  C --> BN[BatchNorm2d<br/>shared or one per timestep]
  BN --> L[LIF sequence<br/>learnable threshold and leak]
  L --> Y[Spike tensor<br/>T x B x Cout x H2 x W2]
```

### SEW residual block

```mermaid
flowchart LR
  X[Input spikes] --> C1[3x3 Conv + BN + LIF]
  C1 --> C2[3x3 Conv + BN + LIF]
  X --> K{Shape changes?}
  K -->|No| I[Identity spikes]
  K -->|Yes| P[1x1 projection<br/>Conv + BN + LIF]
  C2 --> ADD[SEW elementwise add]
  I --> ADD
  P --> ADD
  ADD --> O[Output spike tensor]
```

The SEW addition is performed independently at each timestep. Coincident
spikes can produce a value of two; the result is intentionally not clipped
before entering the next spiking convolution.

### Native Spaun visual front end

```mermaid
flowchart LR
  IMG[5x7 binary retina] --> OPP[Opponent code<br/>35 on + 35 off LIF]
  OPP -->|dense, delay 1| MATCH[16 fixed glyph-match LIF]
  MATCH -->|diagonal, delay 1| ENC[16 symbol-encoding LIF]
  ENC --> EVT[Decode the sole emitted<br/>encoding-neuron index]
```

The caller supplies a retina rather than a symbol label. Host code detects a
new presentation and decodes an emitted index, but all matching occurs in the
two explicit spiking projections. Retina-to-visual and visual-to-encoding are
independent causal lesion points.

### Recurrent semantic workspace

```mermaid
flowchart TB
  VOC[Fixed 32-D vocabulary<br/>D0-D9, N10-N18,<br/>A0-A7, P/K, POS1-POS9] --> SIN[Semantic encoding<br/>32 positive + 32 negative LIF]
  SIN --> TASK[Task buffer<br/>64 recurrent LIF]
  SIN --> QUERY[Query buffer<br/>64 recurrent LIF]
  SIN --> PLAN[Plan buffer<br/>64 recurrent LIF]
  SIN --> SLOTS[9 groups x 9 item slots<br/>each 64 recurrent LIF]
  SLOTS --> CLEAN[Digit cleanup from<br/>current neural activity]
  CLEAN -. host rules for<br/>A5-K/A6/A7 .-> PLAN
  SLOTS -->|gated direct recall<br/>A0/A1/A3/A5-P| PERM[+1 semantic transport<br/>transformation, 64 LIF]
  SLOTS -->|gated current state| C0[Counting stage 0<br/>64 recurrent LIF]
  C0 -->|learned successor<br/>delayed| CN[Counting stages 1-9<br/>64 recurrent LIF each]
  C0 -->|selected-stage gate| PERM
  CN -->|selected-stage gate| PERM
  PLAN -->|selected route| PERM
  PERM --> INV[-1 permutation<br/>action, 64 LIF]
  INV --> DEC[Decoding<br/>64 LIF]
  DEC --> MOTOR[Motor<br/>64 LIF]
  MOTOR --> OUT[Digit cleanup from<br/>motor activity only]
```

Every represented dimension uses an opponent pair; the default has one neuron
per sign. Separate recurrent gains for the first, middle, and final list slots,
plus seeded slot noise, provide a mechanism for primacy and recency. They have
not yet been shown to reproduce Spaun's published serial-position curves.

The solid selected-route edges are gated spiking projections. Host control
chooses which gate to open, but A0/A1/A3 and A5 position responses are not
decoded and rebuilt as host plans. A5 kind queries, A6, and A7 still use the
dotted host transform before their planned digits enter the output pipeline.

### Delayed spiking counting chain

```mermaid
flowchart LR
  SLOT[Current A4 start-item<br/>recurrent spikes] -->|identity gate| C0[Stage 0<br/>start value]
  C0 -->|fitted semantic successor<br/>delay 42 ticks by default| C1[Stage 1<br/>start + 1]
  C1 -->|same delayed transform| CX[Stages 2-8]
  CX -->|same delayed transform| C9[Stage 9<br/>start + 9]
  C0 --> SEL[Host-selected stage gate]
  C1 --> SEL
  CX --> SEL
  C9 --> SEL
  SEL --> TX[Semantic transform]
  TX --> ACT[Semantic action]
  ACT --> DEC[Decode and motor route]
```

The successor matrix is fitted from semantic-pointer pairs `0 -> 1` through
`17 -> 18`. A4 seeds stage zero from the existing working-memory spike, clocks
one delayed projection per requested increment, and routes the selected stage
through the normal spiking output path. At a 10 ms tick, the default 42-tick
delay contributes 420 ms per item. Host code still supplies the increment
count, clocks and selects the stage, and formats values 10--18 as two decimal
digits; it does not compute the sum itself.

### Reward-modulated action selector

```mermaid
flowchart LR
  BLOCKS[Externally supplied<br/>three probability blocks] --> ENV[External reward sample]
  CUE[Cue population<br/>1 LIF] -->|3 mutable utility weights<br/>delay 1| ACT[Action population<br/>3 noisy LIF]
  ACT --> COUNT[Spike-count winner<br/>activity tie break]
  COUNT --> CHOICE[Selected arm]
  CHOICE --> ENV
  ENV -->|reward minus baseline| ELIG[Selected-channel<br/>eligibility update]
  ELIG -->|bounded weight change| ACT
```

This compact selector uses the native population runtime's
`reward_modulated_hebbian` projection. It replaces the scaffold's host utility
array. The environment owns the supplied contingency and reward sample; the
spiking selector owns choice and utility adaptation. One model call still
orchestrates the 60 choice/reward exchanges and reads the final utility winner,
and this is not an anatomical reproduction of Spaun's basal ganglia and
thalamus.

## 1. MNIST prototype SNN

Entry point: `experiments/mnist/main.cpp`

```mermaid
flowchart LR
  I[28x28 grayscale image] --> G[Adaptive pooled grid]
  G --> T[Compare each cell mean<br/>with 1.30 x image mean]
  T --> E[One SpikeEvent bit vector]
  E --> N0[Class neuron 0]
  E --> N1[Class neuron 1]
  E --> NX[...]
  E --> N9[Class neuron 9]
  N0 --> A[argmax similarity]
  N1 --> A
  NX --> A
  N9 --> A
  A --> O[Predicted digit]
```

Structure:

- One event encoder compresses the image into at most the available spike-event
  payload bits.
- There are exactly ten independent prototype neurons, one per digit.
- Supervised training reinforces only the neuron associated with the label.
- Prediction evaluates all ten neurons and selects the largest similarity.
- This is an event/prototype SNN; it has no convolution or dense ANN layer.

## 2. EMNIST baseline prototype SNN

Entry point: `experiments/emnist/main.cpp` without `--structured`

```mermaid
flowchart LR
  I[28x28 EMNIST image] --> E[MNIST pooled-grid<br/>SpikeEvent encoder]
  E --> B[One prototype neuron per class]
  B --> A[argmax class score]
  A --> O[Label plus split offset]
```

The structure is the MNIST prototype network generalized to the selected
split. `C` is 10 for digits/mnist, 26 for letters, 47 for balanced/bymerge,
or 62 for byclass. The letters split adds label offset one after classification.

## 3. EMNIST structured expert SNN

Entry point: `experiments/emnist/main.cpp --structured`

```mermaid
flowchart TB
  I[28x28 EMNIST image] --> CTR[Center ink bounding box]
  CTR --> Q[Four 14x14 quadrant encoders]
  CTR --> PG[4x4 grid of 7x7 patches]
  PG --> D[Six prototype shape detectors per patch<br/>horizontal, vertical, diagonals, corners, box]
  D --> F[One structural-feature SpikeEvent]
  Q --> X[For every class: four experts]
  F --> X
  X --> B[Each expert has five prototype banks<br/>4 quadrant banks + 1 feature bank]
  B --> S[Mean of four quadrant scores<br/>plus 0.5 x feature score]
  S --> BE[Best expert per class]
  BE --> BC[Best class]
  BC --> O[Predicted label]
```

Per selected class, the network contains:

- four experts;
- five `snnbase::Neuron` prototype banks per expert;
- up to 64 prototypes per bank;
- four quadrant event channels plus one learned structural-feature channel.

The patch feature layer contains six fixed prototype detectors. Training uses
small deterministic image shifts, assigns examples to available/best experts,
reinforces a misclassified correct expert, and punishes the competing expert.

## 4. EMNIST deep temporal SEW-ResNet

Entry point: `experiments/emnist/deep_main.cpp`

Default: `T=4`, `P=2`, two residual blocks per stage.

```mermaid
flowchart LR
  I[1x28x28 normalized image] --> R[Repeat direct current<br/>across T=4 steps]
  R --> ST[Stem<br/>3x3, 1 to 16<br/>BN + LIF]
  ST --> S1[Stage 1<br/>2 SEW blocks<br/>16x28x28]
  S1 --> S2[Stage 2<br/>2 SEW blocks<br/>32x14x14<br/>first block stride 2]
  S2 --> S3[Stage 3<br/>2 SEW blocks<br/>64x7x7<br/>first block stride 2]
  S3 --> GAP[Spatial global average<br/>at every timestep]
  GAP --> FC[Linear 64 to C x 2]
  FC --> PM[Mean population dimension]
  PM --> TM[Leaky temporal membrane readout]
  TM --> O[C class logits]
```

Every stem and residual convolution is followed by an LIF spike generator.
Only the final linear population readout and leaky logit accumulator are
non-spiking arithmetic. The number of output classes and label offset come
from the selected EMNIST split.

## 5. CIFAR-10 handcrafted feature-bank classifier

Implementation: `src/cifar10.cpp`

```mermaid
flowchart TB
  I[3x32x32 CIFAR image] --> F1[Color intensity event]
  I --> F2[Opponent-color event]
  I --> F3[Sobel-orientation event]
  I --> F4[Center-surround event]
  I --> F5[Color-category event]
  I --> F6[Quadrant-texture event]
  F1 --> NB[Binary feature likelihoods per class]
  F2 --> NB
  F3 --> NB
  F4 --> NB
  F5 --> NB
  F6 --> NB
  NB --> LP[Class prior + summed Bernoulli log likelihood]
  LP --> A[argmax over 10 classes]
```

This classifier creates six spike-event feature banks and learns per-class
active-bit counts. Its decision rule is a smoothed Bernoulli naive-Bayes score.
It operates on spike-event features but does not contain simulated LIF layers;
it is best understood as an event-coded statistical baseline.

## 6. CIFAR-10 shallow spiking convolution network

Entry point: `experiments/cifar10/main.cpp`

Default: 16 channels, `T=8`, threshold 0.5, channel normalization enabled,
and SEW-add residual merge.

```mermaid
flowchart LR
  I[3x32x32 image] --> RC[Rate representation<br/>T=8 discrete levels]
  RC --> C1[Conv 3x3 same<br/>3 to 16]
  C1 --> N1[Per-channel normalization]
  N1 --> L1[Spiking neurons<br/>16x32x32]
  L1 --> C2[Conv 3x3 same<br/>16 to 16]
  C2 --> N2[Per-channel normalization]
  N2 --> L2[Spiking neurons<br/>16x32x32]
  L1 --> ADD[SEW-add skip]
  L2 --> ADD
  ADD --> FLAT[Flatten 16x32x32]
  FLAT --> D[Dense to 10 class units]
  D --> SM[Softmax probabilities]
  SM --> O[Predicted class]
```

Both convolution layers use `snnbase::Neuron` thresholded spike/rate units.
The residual mode is configurable as none, averaged add, SEW add, or SEW
multiply. Unlike the deep temporal model, this implementation explicitly
stores scalar convolution weights and manually computes surrogate gradients
and Adam updates.

## 7. CIFAR-10 deep temporal SEW-ResNet

Entry point: `experiments/cifar10/deep_main.cpp`

Default: width 32, two blocks per stage, `T=4`, `P=2`. The selected sweep
configuration changes width to 48 but leaves the topology unchanged.

```mermaid
flowchart LR
  I[3x32x32 normalized image] --> AUG[Random crop / flip<br/>optional Cutout]
  AUG --> ENC[Direct-current repeat<br/>T=4 spike simulation steps]
  ENC --> ST[Stem<br/>3x3, 3 to W<br/>BN + LIF]
  ST --> S1[Stage 1<br/>2 SEW blocks<br/>W x 32x32]
  S1 --> S2[Stage 2<br/>2 SEW blocks<br/>2W x 16x16<br/>stride-2 projection]
  S2 --> S3[Stage 3<br/>2 SEW blocks<br/>4W x 8x8<br/>stride-2 projection]
  S3 --> GAP[Spatial global average<br/>per timestep]
  GAP --> FC[Linear 4W to 10 x P]
  FC --> POP[Population mean<br/>P=2]
  POP --> MEM[Leaky temporal membrane accumulation]
  MEM --> O[10 class logits]
```

For the default `W=32`, the stage widths are 32, 64, and 128. For the
accuracy-sweep `W=48` model, they are 48, 96, and 192. Each downsampling block
uses a spiking 1x1 projection on the skip branch. Batch normalization can be
shared across time or replaced with independent per-timestep BNTT statistics.

### Complete deep-model dataflow

```mermaid
flowchart TB
  IMG[Image batch BxCxHxW] --> SEQ[Temporal encoding TxBxCxHxW]
  SEQ --> SPIKES[Stem and six SEW residual blocks]
  SPIKES --> POOL[TxBx4W global-average features]
  POOL --> READ[TxBx10xP linear currents]
  READ --> PAVG[Mean P neurons per class]
  PAVG --> LOOP[For t=0..T-1:<br/>membrane = leak x membrane + current_t]
  LOOP --> LOGITS[Bx10 logits]
  LOGITS --> LOSS[Label-smoothed cross entropy]
  SPIKES --> RATE[Mean network spike rate]
  RATE --> REG[Spike-rate target penalty]
  REG --> LOSS
```

The classifier intentionally reads membrane/current values rather than forcing
the ten output units to spike. This preserves gradient information at the
decision boundary while all spatial feature extraction remains on the spiking
substrate.

## 8. Scaled Spaun perception-cognition-action migration

Entry points: `experiments/spaun/main.cpp` and
`experiments/spaun/gui_main.cpp`

```mermaid
flowchart TB
  I[Visual A0-A7 5x7 stream] --> RET[Opponent retina<br/>70 LIF]
  RET --> VIS[Fixed visual match<br/>16 LIF]
  VIS --> ENC[Symbol encoding<br/>16 LIF]
  ENC --> ROUTE[Event-driven token routing]
  ROUTE --> WM[Recurrent semantic workspace<br/>81 item slots]
  WM -->|gated recalled states<br/>A0/A1/A3/A5-P| PIPE[Transformation -> action -><br/>decoding -> motor spikes]
  WM -->|A4 start state| COUNT[10 recurrent count stages<br/>9 delayed successor projections]
  COUNT -->|selected-stage gate| PIPE
  WM -. cleanup read .-> HOST[Remaining host rules<br/>A5-K/A6/A7]
  HOST --> PLAN[Recurrent semantic plan]
  PLAN --> PIPE
  PIPE --> STROKE[Host stroke planner and IK]
  STROKE --> LEGACY[Legacy nine-module telemetry SNN<br/>576 LIF]
  LEGACY --> P[Two-joint physical plant<br/>pen trajectory]

  BLOCKS[Externally supplied<br/>A2 probability blocks] --> ENV[External bandit environment]
  BANDIT[A2 cue<br/>1 LIF] -->|reward-plastic weights| ACTION[A2 actions<br/>3 LIF]
  ACTION --> ENV
  ENV -->|sampled reward| ACTION
  ACTION -. final utility readout .-> PLAN
```

`spaun_experiment` reports **7,018 LIF neurons and 1,489,322 stored
coefficients** at the current defaults: 102/1,136 visual, 6,336/1,486,848
semantic, 4/3 selector, and 576/1,335 legacy telemetry.
Do not reuse the old 576-neuron/1,335-synapse baseline as a current total: it
describes only the legacy graph and predates the semantic and counting
populations.

Task cues and operands are recovered from retina-driven spikes and retained in
recurrent semantic populations. A0/A1/A3 and A5 position-query digits traverse
explicit selected-slot gates without being rebuilt as host plans. A4 advances
the recalled start state through a delayed, fitted semantic-successor chain.
A2 contingencies come from the external environment while choice utilities
adapt in reward-plastic synapses.

Host orchestration remains decisive: it parses recognized tokens, selects
recall gates, clocks and selects the A4 stage, formats multi-digit values,
reads the final A2 utility winner, and computes A5 kind-query, A6, and A7 plans.
The GUI is observational and does not feed its decoded telemetry back into the
model. Behavioral scoring requires a decoded match plus observable arm ink,
but does not yet perform blinded handwriting recognition. The component
circuits also advance on separate clocks bridged by host calls; GUI time comes
from the legacy graph. The architecture therefore remains a scaled migration,
not end-to-end Spaun-equivalent neural cognition.

The separate A1 benchmark uses a 28×28 two-convolution spiking classifier
(12-channel 5×5/stride-2, then 24-channel 3×3/stride-2, eight timesteps). Its
registered held-out MNIST result is 9,835/10,000, or 98.35%. It is benchmark
evidence for A1, not the visual front end currently wired into this diagram.
See [the Spaun architecture narrative](SPAUN_EXPERIMENT.md) and
[the behavioral benchmark record](results/SPAUN_BEHAVIORAL_BENCHMARK.md) for
evidence limitations.

## Structural comparison

| Structure | Spatial feature extraction | Temporal simulation | Learned representation | Classifier |
|---|---|---:|---|---|
| MNIST prototype | Pooled event grid | No | Per-class spike prototypes | Maximum similarity |
| EMNIST baseline | Pooled event grid | No | Per-class spike prototypes | Maximum similarity |
| EMNIST structured | Quadrants + patch detectors | No | Class/expert prototype banks | Best expert then class |
| EMNIST deep | Spiking SEW-ResNet | 4 steps | Surrogate-trained LIF features | Population membrane readout |
| CIFAR feature bank | Six handcrafted event banks | No | Per-class bit frequencies | Bernoulli log likelihood |
| CIFAR shallow | Two spiking convolutions | 8 levels/steps | Surrogate-trained spike rates | Dense softmax |
| CIFAR deep | Spiking SEW-ResNet | 4 steps | Surrogate-trained LIF features | Population membrane readout |
| Scaled Spaun | Native 5×7 opponent retina; separate 28×28 A1 benchmark | Host-coupled 10 ms component clocks; delayed counting; 8-step A1 classifier | Recurrent semantic LIF slots + successor stages + reward-plastic selector; host control and A5-K/A6/A7 rules remain | Motor semantic cleanup then two-joint arm trajectory |
