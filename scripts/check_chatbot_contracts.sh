#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: check_chatbot_contracts.sh [all|python|cpp]

Runs the chatbot contracts without downloading dependencies or model assets.
The Python checks use only the standard library. The C++ checks compile the
conversation, dataset, and manifest sources directly, without snnbase or
LibTorch.
EOF
  exit 2
}

mode="${1:-all}"
[[ $# -le 1 ]] || usage
case "$mode" in
  all | python | cpp) ;;
  *) usage ;;
esac

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
reference_python="${SNNBASE_REFERENCE_PYTHON:-python3}"

if [[ "$mode" == "all" || "$mode" == "python" ]]; then
  command -v "$reference_python" >/dev/null 2>&1 || {
    echo "Python interpreter not found: $reference_python" >&2
    exit 1
  }
  env -u SNNBASE_QWEN_ASSETS HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$reference_python" -m unittest discover \
    -s "$project_root/tests/python" -v
  "$reference_python" "$project_root/tools/chatbot_config.py" validate \
    "$project_root/configs/chatbot/ann-baseline.json" \
    "$project_root/configs/chatbot/snn-baseline.json" \
    "$project_root/configs/chatbot/qwen3-0.6b-ann-import.json" \
    "$project_root/configs/chatbot/qwen3-0.6b-hybrid-snn.json"
fi

if [[ "$mode" == "all" || "$mode" == "cpp" ]]; then
  cxx_compiler="${CXX:-c++}"
  command -v "$cxx_compiler" >/dev/null 2>&1 || {
    echo "C++ compiler not found: $cxx_compiler" >&2
    exit 1
  }
  contract_build_dir="$(
    mktemp -d "${TMPDIR:-/tmp}/snnbase-chatbot-contracts.XXXXXX"
  )"
  cleanup() {
    rm -rf -- "$contract_build_dir"
  }
  trap cleanup EXIT

  common_flags=(
    -std=c++20
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    "-I$project_root/include"
  )
  "$cxx_compiler" "${common_flags[@]}" \
    "$project_root/src/chatbot/conversation.cpp" \
    "$project_root/src/chatbot/manifest.cpp" \
    "$project_root/src/chatbot/dataset.cpp" \
    "$project_root/tests/chatbot_dataset_tests.cpp" \
    -o "$contract_build_dir/chatbot_dataset_tests"
  "$cxx_compiler" "${common_flags[@]}" \
    "$project_root/src/chatbot/manifest.cpp" \
    "$project_root/tests/chatbot_manifest_tests.cpp" \
    -o "$contract_build_dir/chatbot_manifest_tests"

  "$contract_build_dir/chatbot_dataset_tests"
  SNNBASE_CHATBOT_MANIFEST_TEST_OUTPUT="$contract_build_dir/manifest.json" \
    "$contract_build_dir/chatbot_manifest_tests"

  "$reference_python" -c \
    'import json, sys; [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]]' \
    "$project_root/schemas/chatbot-config-v1.schema.json" \
    "$project_root/schemas/chatbot-run-v1.schema.json" \
    "$contract_build_dir/manifest.json"
fi
