# Reproducibility and implementation status

This file tracks the staged implementation of the snnbase/snnbaseExperiments
roadmap. It is intentionally kept separate from experiment result reports.

## Phase 0 snapshot

Snapshot date: 2026-08-20

| Repository | Branch | Committed revision | Working tree |
|---|---|---|---|
| `snnbaseExperiments` | `main` | `7288011` | Modified tracked files and untracked CIFAR/Spaun sources, tests, scripts, and reports |
| `snnbase` | `codex/prototype-memory-learning` | `aa80bd8` | Modified tracked files and untracked population, semantic, reward, and temporal sources |

The complete Phase 0 recovery snapshot is stored outside the repositories at:

`/tmp/snnbase-phase0.AQyH8C`

It contains status manifests, binary tracked diffs, and archives of untracked
files for both repositories. No source or data files were deleted or reset.

## Gates

- [x] Record both repository SHAs and dirty state.
- [x] Preserve tracked and untracked worktree changes in a recoverable snapshot.
- [ ] Make the library test suite authoritative in Debug and Release.
- [ ] Publish an immutable library revision consumed by experiments.
- [ ] Add clean-checkout experiment CI and run manifests.
- [ ] Reproduce temporal results from clean tags with five seeds.
- [ ] Add a true event-stream benchmark.
- [ ] Integrate learned perception into the Spaun-inspired loop.

## Current constraints

- Library edits require elevated workspace access to
  `/home/dean/repos/snnbase`; that access has now been granted for this run.
- Existing user changes must remain intact and must not be broadly staged or
  discarded.

The initial manifest writer is available at
`scripts/write_run_manifest.sh`. It records both repository revisions and dirty
flags, toolchain versions, the command, and SHA-256 hashes for supplied dataset
directories. Promoted runs must still add resolved configuration, seed, metrics,
checkpoint, and hardware fields.
