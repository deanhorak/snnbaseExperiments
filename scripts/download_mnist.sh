#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-$project_root/data/mnist}"
base_url="https://storage.googleapis.com/cvdf-datasets/mnist"
files=(
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
)
declare -A compressed_md5=(
  [train-images-idx3-ubyte]=f68b3c2dcbeaaa9fbdd348bbdeb94873
  [train-labels-idx1-ubyte]=d53e105ee54ea40749a09fcbcd1e9432
  [t10k-images-idx3-ubyte]=9fb629c4189551a2d022fa330f9573f3
  [t10k-labels-idx1-ubyte]=ec29112dd5afa0611ce80d1b7f02629c
)
declare -A raw_md5=(
  [train-images-idx3-ubyte]=6bbc9ace898e44ae57da46a324031adb
  [train-labels-idx1-ubyte]=a25bea736e30d166cdddb491f175f624
  [t10k-images-idx3-ubyte]=2646ac647ad5339dbf082846283269ea
  [t10k-labels-idx1-ubyte]=27ae3e4e09519cfbb04c329615203637
)

verify_md5() {
  local expected="$1"
  local file="$2"
  command -v md5sum >/dev/null 2>&1 || {
    echo "md5sum is required to verify MNIST" >&2
    exit 1
  }
  printf '%s  %s\n' "$expected" "$file" | md5sum --check --status
}

mkdir -p "$destination"
for file in "${files[@]}"; do
  archive="$destination/$file.gz"
  raw="$destination/$file"
  if [[ -f "$raw" ]]; then
    verify_md5 "${raw_md5[$file]}" "$raw" || {
      echo "checksum mismatch for existing MNIST file: $raw" >&2
      exit 1
    }
    continue
  fi
  if [[ -f "$archive" ]] && ! verify_md5 "${compressed_md5[$file]}" "$archive"; then
    mv "$archive" "$archive.invalid"
  fi
  if [[ ! -f "$archive" ]]; then
    curl --fail --location --retry 3 --retry-delay 2 \
      --output "$archive.partial" "$base_url/$file.gz"
    verify_md5 "${compressed_md5[$file]}" "$archive.partial"
    mv "$archive.partial" "$archive"
  fi
  verify_md5 "${compressed_md5[$file]}" "$archive"
  gzip --decompress --stdout "$archive" > "$raw.partial"
  verify_md5 "${raw_md5[$file]}" "$raw.partial"
  mv "$raw.partial" "$raw"
done

echo "MNIST IDX files are available in $destination"
