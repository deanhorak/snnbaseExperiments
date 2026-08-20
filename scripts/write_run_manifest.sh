#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: write_run_manifest.sh OUTPUT.json [--dataset-dir PATH ...] -- COMMAND [ARG ...]

The command is recorded but not executed. Set SNNBASE_SOURCE_DIR to override the
default sibling snnbase checkout. Set SNNBASE_MANIFEST_DATA_DIRS to a
colon-separated list of additional dataset directories.
EOF
  exit 2
}

[[ $# -ge 3 ]] || usage
output="$1"
shift

command_args=()
dataset_dirs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir)
      [[ $# -ge 2 ]] || usage
      dataset_dirs+=("$2")
      shift 2
      ;;
    --)
      shift
      command_args=("$@")
      break
      ;;
    *)
      usage
      ;;
  esac
done

[[ ${#command_args[@]} -gt 0 ]] || usage
command -v jq >/dev/null 2>&1 || {
  echo "write_run_manifest.sh requires jq" >&2
  exit 1
}

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
library_root="${SNNBASE_SOURCE_DIR:-${project_root}/../snnbase}"
IFS=: read -r -a environment_dataset_dirs <<< "${SNNBASE_MANIFEST_DATA_DIRS:-}"
dataset_dirs+=("${environment_dataset_dirs[@]}")

git_revision() {
  local repository="$1"
  git -C "$repository" rev-parse HEAD 2>/dev/null || printf 'unknown'
}

git_dirty() {
  local repository="$1"
  if [[ -n "$(git -C "$repository" status --porcelain --untracked-files=all 2>/dev/null)" ]]; then
    printf 'true'
  else
    printf 'false'
  fi
}

dataset_json='[]'
for directory in "${dataset_dirs[@]}"; do
  [[ -n "$directory" && -d "$directory" ]] || continue
  while IFS= read -r -d '' file; do
    checksum="$(sha256sum "$file" | awk '{print $1}')"
    dataset_json="$(jq --arg path "$file" --arg sha256 "$checksum" \
      '. + [{path: $path, sha256: $sha256}]' <<<"$dataset_json")"
  done < <(find "$directory" -type f -print0 | sort -z)
done

mkdir -p "$(dirname -- "$output")"
jq -n \
  --arg created_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg experiments_path "$project_root" \
  --arg experiments_revision "$(git_revision "$project_root")" \
  --argjson experiments_dirty "$(git_dirty "$project_root")" \
  --arg library_path "$library_root" \
  --arg library_revision "$(git_revision "$library_root")" \
  --argjson library_dirty "$(git_dirty "$library_root")" \
  --arg cmake "$(/usr/bin/cmake --version | head -1)" \
  --arg compiler "$(c++ --version | head -1)" \
  --argjson command "$(printf '%s\n' "${command_args[@]}" | jq -R -s 'split("\n")[:-1]')" \
  --argjson datasets "$dataset_json" \
  '{schema_version: 1,
    created_utc: $created_utc,
    repositories: {
      experiments: {path: $experiments_path, revision: $experiments_revision, dirty: $experiments_dirty},
      library: {path: $library_path, revision: $library_revision, dirty: $library_dirty}
    },
    toolchain: {cmake: $cmake, compiler: $compiler},
    command: $command,
    datasets: $datasets}' > "$output"

echo "wrote run manifest: $output"
