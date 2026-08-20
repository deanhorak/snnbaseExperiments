#!/usr/bin/env bash

set -euo pipefail

binary="${1:-./build/cifar10_deep}"
data_dir="${2:-data/cifar-10-batches-bin}"
output_dir="${3:-/tmp/cifar10-sweep-screen}"
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${output_dir}"

common=(
  --data-dir "${data_dir}"
  --train-limit 10000
  --epochs 8
  --warmup-epochs 2
  --time-steps 4
  --width 32
  --blocks 2
  --batch-size 128
  --threshold 1.0
  --leak 0.9
  --surrogate-slope 4
  --readout-population 2
  --device cuda
  --seed 42
  --validation-only
)

run_case() {
  local name="$1"
  shift
  local log="${output_dir}/${name}.log"
  echo "case=${name}"
  "${script_root}/write_run_manifest.sh" \
    "${output_dir}/${name}.manifest.json" \
    --dataset-dir "${data_dir}" -- "${binary}" "${common[@]}" "$@"
  "${binary}" "${common[@]}" "$@" >"${log}" 2>&1
  sed -n '/^dataset=/p;/^best_validation_epoch=/p;/^train_seconds=/p' "${log}"
}

run_case threshold_075 --threshold 0.75
run_case threshold_125 --threshold 1.25
run_case leak_080 --leak 0.80
run_case leak_095 --leak 0.95
run_case slope_2 --surrogate-slope 2
run_case slope_8 --surrogate-slope 8
run_case rate_encoding --encoding rate
run_case population_4 --readout-population 4
run_case depth_3 --blocks 3 --batch-size 96
run_case width_48 --width 48 --batch-size 64
run_case timesteps_6 --time-steps 6 --batch-size 96
