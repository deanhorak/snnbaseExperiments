#!/usr/bin/env bash

# Run a fixed, artifact-backed CIFAR-10 temporal-SNN protocol across five
# independent seeds. Override the SNNBASE_TEMPORAL_* variables for a smaller
# local gate; retain the emitted manifests, logs, and checkpoints for every
# scientifically reported run.
set -euo pipefail

binary="${1:-./build/cifar10_deep}"
data_dir="${2:-data/cifar-10-batches-bin}"
output_dir="${3:-artifacts/cifar10-temporal-five-seed}"
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

seeds="${SNNBASE_TEMPORAL_SEEDS:-42 43 44 45 46}"
epochs="${SNNBASE_TEMPORAL_EPOCHS:-100}"
train_limit="${SNNBASE_TEMPORAL_TRAIN_LIMIT:-0}"
test_limit="${SNNBASE_TEMPORAL_TEST_LIMIT:-0}"
device="${SNNBASE_TEMPORAL_DEVICE:-cuda}"
width="${SNNBASE_TEMPORAL_WIDTH:-32}"
blocks="${SNNBASE_TEMPORAL_BLOCKS:-2}"
batch_size="${SNNBASE_TEMPORAL_BATCH_SIZE:-128}"
time_steps="${SNNBASE_TEMPORAL_TIME_STEPS:-4}"

[[ -x "${binary}" ]] || {
  echo "temporal executable is not runnable: ${binary}" >&2
  exit 2
}
[[ -d "${data_dir}" ]] || {
  echo "CIFAR-10 directory does not exist: ${data_dir}" >&2
  exit 2
}

mkdir -p "${output_dir}"
summary="${output_dir}/summary.csv"
printf '%s\n' 'seed,best_validation_epoch,best_validation_accuracy_percent,test_accuracy_percent,checkpoint_sha256,log_sha256' >"${summary}"

common=(
  --data-dir "${data_dir}"
  --epochs "${epochs}"
  --warmup-epochs 5
  --time-steps "${time_steps}"
  --width "${width}"
  --blocks "${blocks}"
  --batch-size "${batch_size}"
  --threshold 1.0
  --leak 0.9
  --surrogate-slope 4.0
  --readout-population 2
  --learning-rate 0.001
  --weight-decay 0.0005
  --label-smoothing 0.1
  --cutout-size 8
  --device "${device}"
)

if [[ "${train_limit}" != 0 ]]; then
  common+=(--train-limit "${train_limit}")
fi
if [[ "${test_limit}" != 0 ]]; then
  common+=(--test-limit "${test_limit}")
fi

for seed in ${seeds}; do
  run_dir="${output_dir}/seed-${seed}"
  mkdir -p "${run_dir}"
  best_checkpoint="${run_dir}/best-validation.pt"
  final_checkpoint="${run_dir}/final.pt"
  log="${run_dir}/run.log"
  manifest="${run_dir}/run.manifest.json"
  command=("${binary}" "${common[@]}" --seed "${seed}"
           --save-best-checkpoint "${best_checkpoint}"
           --save-checkpoint "${final_checkpoint}")

  "${script_root}/write_run_manifest.sh" "${manifest}" \
    --dataset-dir "${data_dir}" -- "${command[@]}"
  "${command[@]}" >"${log}" 2>&1

  [[ -s "${best_checkpoint}" ]] || {
    echo "missing best-validation checkpoint for seed ${seed}" >&2
    exit 1
  }
  [[ -s "${final_checkpoint}" ]] || {
    echo "missing final checkpoint for seed ${seed}" >&2
    exit 1
  }

  best_epoch="$(awk -F'[ =]' '/^best_validation_epoch=/ {print $2; exit}' "${log}")"
  best_accuracy="$(awk -F'[ =%]+' '/^best_validation_epoch=/ {print $4; exit}' "${log}")"
  test_accuracy="$(awk -F'[ =%]+' '/^test_correct=/ {print $4; exit}' "${log}")"
  [[ -n "${best_epoch}" && -n "${best_accuracy}" && -n "${test_accuracy}" ]] || {
    echo "could not parse complete metrics for seed ${seed}" >&2
    exit 1
  }
  printf '%s,%s,%s,%s,%s,%s\n' \
    "${seed}" "${best_epoch}" "${best_accuracy}" "${test_accuracy}" \
    "$(sha256sum "${best_checkpoint}" | awk '{print $1}')" \
    "$(sha256sum "${log}" | awk '{print $1}')" >>"${summary}"
done

awk -F, '
  NR == 1 { next }
  { sum += $4; sumsq += $4 * $4; count += 1 }
  END {
    if (count == 0) exit 1
    mean = sum / count
    variance = count > 1 ? (sumsq - sum * sum / count) / (count - 1) : 0
    printf "completed_seeds=%d mean_test_accuracy_percent=%.4f sample_stddev_percent=%.4f\n", \
      count, mean, sqrt(variance)
  }
' "${summary}" | tee "${output_dir}/aggregate.txt"

echo "artifacts=${output_dir}"
