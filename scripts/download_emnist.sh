#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-$project_root/data/emnist}"
archive="$destination/gzip.zip"
source_url="https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

mkdir -p "$destination"
if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 3 --output "$archive" "$source_url"
fi

unzip -o -j "$archive" 'gzip/*.gz' -d "$destination"
for file in "$destination"/*.gz; do
  gzip --decompress --force "$file"
done

echo "EMNIST IDX files are available in $destination"
