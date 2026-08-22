#!/usr/bin/env bash

# Install the two official N-MNIST archives after downloading them manually
# from https://www.garrickorchard.com/datasets/n-mnist. The provider exposes
# several hosted mirrors, so this script intentionally verifies local archives
# instead of hard-coding a brittle mirror URL.
set -euo pipefail

usage() {
  echo "Usage: $0 TRAIN.zip TEST.zip [DESTINATION]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage
train_archive="$1"
test_archive="$2"
destination="${3:-data/nmnist}"
train_md5="20959b8e626244a1b502305a9e6e2031"
test_md5="69ca8762b2fe404d9b9bad1103e97832"

for archive in "${train_archive}" "${test_archive}"; do
  [[ -f "${archive}" ]] || {
    echo "archive does not exist: ${archive}" >&2
    exit 1
  }
  unzip -tq "${archive}" >/dev/null
done

[[ "$(md5sum "${train_archive}" | awk '{print $1}')" == "${train_md5}" ]] || {
  echo "Train.zip checksum mismatch" >&2
  exit 1
}
[[ "$(md5sum "${test_archive}" | awk '{print $1}')" == "${test_md5}" ]] || {
  echo "Test.zip checksum mismatch" >&2
  exit 1
}

mkdir -p "${destination}"
unzip -nq "${train_archive}" -d "${destination}"
unzip -nq "${test_archive}" -d "${destination}"

for split in Train Test; do
  for label in {0..9}; do
    find "${destination}/${split}/${label}" -maxdepth 1 -type f -name '*.bin' -print -quit |
      grep -q . || {
        echo "N-MNIST extraction is incomplete: ${destination}/${split}/${label}" >&2
        exit 1
      }
  done
done

echo "installed validated N-MNIST data in ${destination}"
