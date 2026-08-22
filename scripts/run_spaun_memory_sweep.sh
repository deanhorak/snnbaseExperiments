#!/usr/bin/env bash

set -euo pipefail

binary="${1:-./build/spaun_benchmark}"
output_dir="${2:-artifacts/spaun-memory-sweep}"
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${output_dir}"

# name, recurrent-memory noise, middle-position recurrent gain
cases=(
  "accuracy_first 0.00945 1.02005"
  "transition 0.01015 1.02000"
  "behavioral_default 0.01050 1.02000"
)

for specification in "${cases[@]}"; do
  read -r name noise recurrence <<<"${specification}"
  for seed in 42 43 44; do
    report="${output_dir}/${name}-seed${seed}.json"
    echo "case=${name} seed=${seed} noise=${noise} recurrence=${recurrence}"
    "${script_root}/write_run_manifest.sh" \
      "${output_dir}/${name}-seed${seed}.manifest.json" -- \
      "${binary}" --task A3,A5 --seed "${seed}" --bootstrap 3000 \
      --memory-noise "${noise}" --memory-recurrence "${recurrence}" \
      --primacy-recurrence 1.05 --recency-recurrence 1.05 --json "${report}"
    "${binary}" \
      --task A3,A5 \
      --seed "${seed}" \
      --bootstrap 3000 \
      --memory-noise "${noise}" \
      --memory-recurrence "${recurrence}" \
      --primacy-recurrence 1.05 \
      --recency-recurrence 1.05 \
      --json "${report}"
  done
done
