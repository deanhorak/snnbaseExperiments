#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: download_qwen_assets.sh --model-id OWNER/REPO --revision COMMIT \
  --expected-tokenizer-fingerprint SHA256 --output-dir PATH [--include-weights]

Downloads the tokenizer/config/license files. Pass --include-weights for the
exact model.safetensors required by conversion. Phase-0 pin:
  model:       Qwen/Qwen3-0.6B-Base
  revision:    da87bfb608c14b7cf20ba1ce41287e8de496c0cd
  fingerprint: 6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237

Set SNNBASE_REFERENCE_PYTHON to the isolated reference Python interpreter.
Set SNNBASE_HF_CLI to an explicit hf/huggingface-cli executable if needed.
EOF
  exit 2
}

model_id=""
revision=""
expected_fingerprint=""
output_dir=""
include_weights=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id)
      [[ $# -ge 2 ]] || usage
      model_id="$2"
      shift 2
      ;;
    --revision)
      [[ $# -ge 2 ]] || usage
      revision="$2"
      shift 2
      ;;
    --expected-tokenizer-fingerprint)
      [[ $# -ge 2 ]] || usage
      expected_fingerprint="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || usage
      output_dir="$2"
      shift 2
      ;;
    --include-weights)
      include_weights=true
      shift
      ;;
    *) usage ;;
  esac
done

[[ "$model_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || usage
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$expected_fingerprint" =~ ^[0-9a-f]{64}$ ]] || usage
[[ -n "$output_dir" ]] || usage

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
reference_python="${SNNBASE_REFERENCE_PYTHON:-python3}"

hf_cli="${SNNBASE_HF_CLI:-}"
if [[ -n "$hf_cli" ]]; then
  hf_cli="$(command -v -- "$hf_cli")" || {
    echo "SNNBASE_HF_CLI is not executable: $SNNBASE_HF_CLI" >&2
    exit 1
  }
else
  reference_scripts="$("$reference_python" -c \
    'import sysconfig; print(sysconfig.get_path("scripts"))')"
  for candidate in "$reference_scripts/hf" \
                   "$reference_scripts/huggingface-cli"; do
    if [[ -x "$candidate" ]]; then
      hf_cli="$candidate"
      break
    fi
  done
  if [[ -z "$hf_cli" ]]; then
    echo "the reference environment has no hf or huggingface-cli executable" >&2
    exit 1
  fi
fi

asset_files=(
  config.json generation_config.json merges.txt tokenizer.json
  tokenizer_config.json vocab.json LICENSE README.md
)
if [[ "$include_weights" == true ]]; then
  asset_files+=(model.safetensors)
fi

"$hf_cli" download "$model_id" "${asset_files[@]}" \
  --revision "$revision" \
  --local-dir "$output_dir" \
  --max-workers 4

"$reference_python" "$project_root/tools/qwen_assets.py" create \
  --assets-dir "$output_dir" \
  --model-id "$model_id" \
  --revision "$revision" \
  --expected-tokenizer-fingerprint "$expected_fingerprint"

if [[ "$include_weights" == true ]]; then
  "$reference_python" "$project_root/tools/qwen_assets.py" verify \
    --assets-dir "$output_dir" \
    --model-id "$model_id" \
    --revision "$revision" \
    --expected-tokenizer-fingerprint "$expected_fingerprint" \
    --require-weights
fi
