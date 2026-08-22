#!/usr/bin/env bash

set -euo pipefail

binary="${1:-./build/cifar10_deep}"
data_dir="${2:-data/cifar-10-batches-bin}"
output_dir="${3:-/tmp/cifar10-width48-hparams}"
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${output_dir}"

common=(
  --data-dir "${data_dir}"
  --train-limit 10000
  --epochs 8
  --warmup-epochs 2
  --time-steps 4
  --width 48
  --blocks 2
  --batch-size 64
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

run_case learning_rate_0005 --learning-rate 0.0005
run_case learning_rate_002 --learning-rate 0.002
run_case weight_decay_0001 --weight-decay 0.0001
run_case weight_decay_001 --weight-decay 0.001
run_case smoothing_005 --label-smoothing 0.05
run_case smoothing_0 --label-smoothing 0
run_case cutout_0 --cutout-size 0
run_case cutout_16 --cutout-size 16
