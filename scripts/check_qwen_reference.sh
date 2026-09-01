#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: check_qwen_reference.sh --assets-dir PATH [--include-logits]

Verifies a previously downloaded local Qwen Phase 0 tokenizer snapshot and
regenerates the tokenizer golden fixture with network access disabled. Install
requirements/qwen-reference.lock in an isolated environment first and point
SNNBASE_REFERENCE_PYTHON at that environment's Python interpreter.

Pass --include-logits to require the pinned weight file and regenerate/compare
the bounded CPU float32 reference-logit fixture as well.
EOF
  exit 2
}

assets_dir=""
include_logits=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --assets-dir)
      [[ $# -ge 2 ]] || usage
      assets_dir="$2"
      shift 2
      ;;
    --include-logits)
      include_logits=true
      shift
      ;;
    *) usage ;;
  esac
done
[[ -d "$assets_dir" ]] || usage

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
reference_python="${SNNBASE_REFERENCE_PYTHON:-python3}"
model_id="Qwen/Qwen3-0.6B-Base"
revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
fingerprint="6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237"
reference_dir="$(mktemp -d "${TMPDIR:-/tmp}/snnbase-qwen-reference.XXXXXX")"
cleanup() {
  rm -rf -- "$reference_dir"
}
trap cleanup EXIT

offline_environment=(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)
verify_command=(
  "$reference_python" "$project_root/tools/qwen_assets.py" verify
  --assets-dir "$assets_dir"
  --model-id "$model_id"
  --revision "$revision"
  --expected-tokenizer-fingerprint "$fingerprint"
)
if [[ "$include_logits" == true ]]; then
  verify_command+=(--require-weights)
fi
env "${offline_environment[@]}" "${verify_command[@]}"

generated="$reference_dir/qwen3_phase0_tokenizer_golden.json"
env "${offline_environment[@]}" "$reference_python" \
  "$project_root/tools/qwen_reference.py" tokenizer-golden \
  --assets-dir "$assets_dir" \
  --model-id "$model_id" \
  --revision "$revision" \
  --expected-tokenizer-fingerprint "$fingerprint" \
  --cases "$project_root/tests/fixtures/chatbot/qwen3_phase0_cases.json" \
  --output "$generated"

expected="$project_root/tests/fixtures/chatbot/qwen3_phase0_tokenizer_golden.json"
if ! cmp -s "$expected" "$generated"; then
  diff -u "$expected" "$generated" || true
  echo "regenerated Qwen tokenizer golden does not match the checked-in fixture" >&2
  exit 1
fi
echo "Pinned Qwen tokenizer assets and golden fixture match."

if [[ "$include_logits" == true ]]; then
  generated_logits="$reference_dir/qwen3_phase0_reference_logits.json"
  env "${offline_environment[@]}" "$reference_python" \
    "$project_root/tools/qwen_reference.py" reference-logits \
    --assets-dir "$assets_dir" \
    --model-id "$model_id" \
    --revision "$revision" \
    --expected-tokenizer-fingerprint "$fingerprint" \
    --cases "$project_root/tests/fixtures/chatbot/qwen3_phase0_cases.json" \
    --output "$generated_logits" \
    --top-k 16 \
    --absolute-tolerance 1e-5 \
    --relative-tolerance 1e-5

  expected_logits="$project_root/tests/fixtures/chatbot/qwen3_phase0_reference_logits.json"
  if ! cmp -s "$expected_logits" "$generated_logits"; then
    diff -u "$expected_logits" "$generated_logits" || true
    echo "regenerated Qwen reference logits do not match the checked-in fixture" >&2
    exit 1
  fi
  echo "Pinned Qwen CPU reference-logit fixture matches."
fi
