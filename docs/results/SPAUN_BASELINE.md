# snnbase scaled Spaun baseline

> **Historical baseline (superseded).** This record predates the native visual
> front end, recurrent semantic workspace, reward-modulated selector, direct
> recalled-state gates, and delayed semantic-successor counting chain. Its
> graph counts, spike totals, timings, and host-solver description are retained
> for provenance and do not describe the current executable. Use a newly
> generated benchmark report for current results.

Date: 2026-07-18  
Configuration: release build, seed 42, 64 neurons per module, 10 ms tick,
150 ms stimulus, 150 ms blank, five arm-integration ticks per target.

Command:

```bash
/usr/bin/time -f 'wall_seconds=%e max_rss_kb=%M' \
  /tmp/snnbase-spaun/spaun_experiment \
  --task all \
  --minimum-accuracy 1.0 \
  --trace-json /tmp/spaun-trace.json
```

Graph:

- 576 `snnbase` LIF neurons;
- 1,335 explicit delayed synapses;
- nine functional populations;
- visual and working-memory prototype neurons outside the LIF graph.

## Results

| Task | Expected | Output | Correct | Spikes | Simulated time | Frames |
|---|---:|---:|---:|---:|---:|---:|
| A0 Copy drawing | 2 | 2 | Yes | 45,939 | 5.08 s | 508 |
| A1 Recognition | 7 | 7 | Yes | 39,983 | 4.45 s | 445 |
| A2 Reinforcement learning | 2 | 2 | Yes | 43,681 | 4.78 s | 478 |
| A3 Serial working memory | 015873 | 015873 | Yes | 213,016 | 22.73 s | 2,273 |
| A4 Counting | 8 | 8 | Yes | 59,625 | 6.66 s | 666 |
| A5 Question answering | 1 | 1 | Yes | 73,678 | 8.52 s | 852 |
| A6 Rapid variable creation | 74 | 74 | Yes | 124,780 | 14.39 s | 1,439 |
| A7 Fluid reasoning | 555 | 555 | Yes | 177,181 | 20.08 s | 2,008 |

Canonical task score: **8/8 (100%)**.  
Measured wall time: **0.39 seconds**.  
Measured peak RSS: **34,712 KiB**.

The longer motor phase is intentional: ordered vector strokes are interpolated,
the physical arm converges with the pen raised before each stroke, and every
endpoint is completed before the pen lifts. This replaced the original raster
row scan, which lowered the pen while the arm was still outside the drawing
surface and could not draw vertical or singleton glyph segments.

Multi-digit responses are presented sequentially: every completed digit is held
for 350 ms, the live surface is cleared with the pen raised, and the next digit
is drawn full-size. The final digit remains visible. The A6 `74` trace, for
example, contains one erase transition between the `7` and `4` rather than a
cumulative two-digit overlay.

This is a deterministic implementation-correctness result, not a scientific
accuracy comparison with human participants or original Spaun. Canonical
answers currently use exact glyphs and host-side task transformations, so 100%
must not be compared with Spaun's reported MNIST, memory, or reasoning results.

## Verification

- `spaun_tests`: passed;
- `spaun_task_battery`: passed;
- `spaun_gui --self-test`: passed;
- native FreeGLUT window creation/render/automatic close: passed for five frames.
- geometry regression for all digits, continuous ink, stroke boundaries, and
  sequential multi-digit clearing: passed;
- visual inspection of completed A0 `2` and A6 `74` drawings: passed.
