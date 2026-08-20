# Spaun memory parameter sweep

Date: 2026-07-18

This sweep tunes the recurrent semantic memory used by A3 serial recall and A5
question answering. It deliberately optimizes the published behavioral
signature, not a universal accuracy target: Spaun was compared with
serial-position curves showing primacy and recency, while only A1 has a
published recognition-accuracy reference.

## Sweep design

The sweep varied memory noise and the recurrent gain for middle list
positions. Primacy and recency gains remained `1.05`. Each confirmation used
the full A3 schedule (40 trials at each length 4--7), the full A5 schedule (10
participants x 7 positions x P/K), seeds 42--44, and 3,000 bootstrap samples.

Representative points:

| Setting | Noise | Middle gain | Accuracy result | Behavioral result |
|---|---:|---:|---|---|
| Accuracy-first region | 0.00945 | 1.02005 | A3 exact 85--91%; A5 exact 94--97% across seeds | Failed the per-length A3 primacy/recency gate because short lists were at ceiling |
| Transition point | 0.01015 | 1.02000 | Seed-42 A3 exact 46.25%; A5 exact 75.00% | Passed both gates for seed 42, but not seeds 43--44 |
| Behavioral default | **0.01050** | **1.02000** | Lower exact accuracy, as expected from a serial-position effect | Passed both A3 and A5 gates on all three seeds |

The accuracy-first point is useful when exact list reproduction is the product
objective. It is not the correct default for a Spaun behavioral comparison:
ceiling performance removes the error profile that the experiment is meant to
reproduce.

## Three-seed confirmation of the selected default

| Seed | A3 exact (95% CI) | A3 item accuracy | A3 endpoint advantage | A5 exact (95% CI) | A5 endpoint advantage |
|---:|---:|---:|---:|---:|---:|
| 42 | 35.00% [28.13%, 42.50%] | 69.77% | +0.3882 | 64.29% [56.43%, 72.14%] | +0.50 |
| 43 | 31.88% [24.38%, 38.75%] | 70.11% | +0.3941 | 62.86% [54.98%, 70.71%] | +0.52 |
| 44 | 36.25% [28.75%, 43.75%] | 68.30% | +0.4053 | 66.43% [58.57%, 74.29%] | +0.47 |

Aggregate A3 exact accuracy was 165/480 = 34.38%; mean item accuracy was
69.39%. Aggregate A5 exact accuracy was 271/420 = 64.52%. These aggregate
values are diagnostics, not published Spaun acceptance thresholds. The gates
are the serial-position shapes: every A3 list length must show both endpoint
advantages, and the A5 P/K profiles must remain similar while showing primacy
and recency. Those are explicit project operationalizations because the source
does not publish numeric cutoffs for either shape.

## Reproduction

Run the recorded grid with:

```bash
./scripts/run_spaun_memory_sweep.sh ./build/spaun_benchmark
```

Or reproduce one confirmation directly:

```bash
./build/spaun_benchmark \
  --task A3,A5 \
  --memory-noise 0.0105 \
  --memory-recurrence 1.020 \
  --primacy-recurrence 1.05 \
  --recency-recurrence 1.05 \
  --bootstrap 3000 \
  --seed 42 \
  --json spaun-memory-seed42.json
```

The selected values are now the `spaun::Config` defaults.
