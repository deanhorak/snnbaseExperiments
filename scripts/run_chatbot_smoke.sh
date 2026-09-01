#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_chatbot_smoke.sh --build-dir PATH

Builds and runs the dependency-free chatbot contracts plus the LibTorch runner
unit test and tiny CPU training self-test from an already configured chatbot
build directory. It does not run a dataset training or quality evaluation.
EOF
  exit 2
}

build_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      [[ $# -ge 2 ]] || usage
      build_dir="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done
[[ -n "$build_dir" ]] || usage
[[ -f "$build_dir/CMakeCache.txt" ]] || {
  echo "not a configured CMake build directory: $build_dir" >&2
  exit 1
}
grep -q '^SNNBASE_EXPERIMENTS_ENABLE_CHATBOT:BOOL=ON$' \
  "$build_dir/CMakeCache.txt" || {
  echo "build directory was not configured with chatbot support" >&2
  exit 1
}

cmake_command="${CMAKE_COMMAND:-$(command -v cmake)}"
ctest_command="${CTEST_COMMAND:-}"
if [[ -z "$ctest_command" ]]; then
  cmake_program_dir="${cmake_command%/*}"
  if [[ -x "$cmake_program_dir/ctest" ]]; then
    ctest_command="$cmake_program_dir/ctest"
  else
    ctest_command="$(command -v ctest)"
  fi
fi
torch_dir="$(sed -n 's/^Torch_DIR:PATH=//p' "$build_dir/CMakeCache.txt")"
[[ -n "$torch_dir" ]] || {
  echo "build directory does not record Torch_DIR" >&2
  exit 1
}
torch_library_dir="$(realpath "$torch_dir/../../..")/lib"
[[ -d "$torch_library_dir" ]] || {
  echo "could not resolve the configured LibTorch library directory" >&2
  exit 1
}

"$cmake_command" --build "$build_dir" --target \
  chatbot_dataset_tests \
  chatbot_manifest_tests \
  chatbot_qwen_weights_tests \
  chatbot_runner_tests \
  chatbot_experiment
env LD_LIBRARY_PATH="$torch_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ctest_command" --test-dir "$build_dir" \
  -R '^(chatbot_dataset_tests|chatbot_manifest_tests|chatbot_qwen_weights_tests|chatbot_runner_tests|chatbot_cli_smoke)$' \
  --output-on-failure
