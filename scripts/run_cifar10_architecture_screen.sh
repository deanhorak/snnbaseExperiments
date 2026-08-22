#!/usr/bin/env bash
set -euo pipefail

executable="${1:-./build/experiments/cifar10/cifar10_deep}"
data_dir="${2:-data/cifar-10-batches-bin}"
output_dir="${3:-artifacts/cifar10-architecture-screen}"
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
  --learning-rate 0.001
  --weight-decay 0.0005
  --label-smoothing 0.1
  --cutout-size 0
  --device cuda
  --seed 42
  --validation-only
)

run_case() {
  local name="$1"
  shift
  echo "case=${name}"
  "${script_root}/write_run_manifest.sh" \
    "${output_dir}/${name}.manifest.json" \
    --dataset-dir "${data_dir}" -- "${executable}" "${common[@]}" "$@"
  "${executable}" "${common[@]}" "$@" 2>&1 |
    tee "${output_dir}/${name}.log"
}

run_case centered_shared
run_case black_shared --black-padding
run_case centered_bntt --bntt
run_case black_bntt --bntt --black-padding
