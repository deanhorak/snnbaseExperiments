#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/cifar-10-batches-bin"
ARCHIVE="${ROOT_DIR}/data/cifar-10-binary.tar.gz"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
EXPECTED_MD5="c58f30108f718f92721af3b95e74349a"

verify_md5() {
  local file="$1"
  command -v md5sum >/dev/null 2>&1 || {
    echo "md5sum is required to verify CIFAR-10" >&2
    exit 1
  }
  printf '%s  %s\n' "${EXPECTED_MD5}" "${file}" | md5sum --check --status
}

mkdir -p "${ROOT_DIR}/data"
if [[ -f "${ARCHIVE}" ]] && {
  ! tar -tzf "${ARCHIVE}" >/dev/null 2>&1 || ! verify_md5 "${ARCHIVE}"
}; then
  mv "${ARCHIVE}" "${ARCHIVE}.invalid"
fi
if [[ ! -f "${ARCHIVE}" ]]; then
  curl --fail --location --retry 3 --retry-delay 2 \
    --output "${ARCHIVE}.partial" "${URL}"
  verify_md5 "${ARCHIVE}.partial"
  mv "${ARCHIVE}.partial" "${ARCHIVE}"
fi

verify_md5 "${ARCHIVE}"
tar -xzf "${ARCHIVE}" -C "${ROOT_DIR}/data"

for file in \
  data_batch_1.bin data_batch_2.bin data_batch_3.bin \
  data_batch_4.bin data_batch_5.bin test_batch.bin; do
  if [[ ! -f "${DATA_DIR}/${file}" ]]; then
    echo "CIFAR-10 extraction did not produce ${file}" >&2
    exit 1
  fi
done

if [[ ! -f "${DATA_DIR}/data_batch_1.bin" || ! -f "${DATA_DIR}/test_batch.bin" ]]; then
  echo "CIFAR-10 extraction did not produce expected .bin files" >&2
  exit 1
fi

echo "CIFAR-10 binary files are ready in ${DATA_DIR}"
