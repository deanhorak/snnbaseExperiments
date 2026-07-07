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

mkdir -p "$destination"
for file in "${files[@]}"; do
  archive="$destination/$file.gz"
  if [[ ! -f "$destination/$file" ]]; then
    curl --fail --location --retry 3 --output "$archive" "$base_url/$file.gz"
    gzip --decompress --force "$archive"
  fi
done

echo "MNIST IDX files are available in $destination"
