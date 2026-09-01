#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_with_chatbot_libtorch.sh --build-dir PATH -- COMMAND [ARG ...]

Runs COMMAND directly with the LibTorch library directory recorded by the
configured chatbot build prepended to LD_LIBRARY_PATH. Arguments after the
required -- separator are passed verbatim and are never evaluated by a shell.
EOF
  exit 2
}

fail() {
  echo "run_with_chatbot_libtorch.sh: $1" >&2
  exit 1
}

build_dir=""
saw_build_dir=false
saw_separator=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      if [[ "$saw_build_dir" == true ]]; then
        echo "run_with_chatbot_libtorch.sh: conflicting --build-dir options" >&2
        usage
      fi
      [[ $# -ge 2 && -n "$2" && "$2" != "--" ]] || usage
      build_dir="$2"
      saw_build_dir=true
      shift 2
      ;;
    --)
      saw_separator=true
      shift
      break
      ;;
    *)
      echo "run_with_chatbot_libtorch.sh: unexpected argument before --: $1" >&2
      usage
      ;;
  esac
done

[[ "$saw_build_dir" == true && "$saw_separator" == true ]] || usage
[[ $# -gt 0 && -n "$1" ]] || usage

cache_path="$build_dir/CMakeCache.txt"
[[ -f "$cache_path" ]] || fail "not a configured CMake build directory: $build_dir"

IFS= read -r cache_header <"$cache_path" || fail "empty CMake cache: $cache_path"
[[ "$cache_header" == "# This is the CMakeCache file." ]] ||
  fail "malformed CMake cache header: $cache_path"

torch_entry_count=0
torch_dir=""
torch_entry_type=""
chatbot_entry_count=0
chatbot_enabled=false
while IFS= read -r cache_line || [[ -n "$cache_line" ]]; do
  case "$cache_line" in
    Torch_DIR:*)
      torch_entry_count=$((torch_entry_count + 1))
      if [[ "$cache_line" == Torch_DIR:PATH=* ]]; then
        torch_entry_type="PATH"
        torch_dir="${cache_line#Torch_DIR:PATH=}"
      else
        torch_entry_type="invalid"
      fi
      ;;
    SNNBASE_EXPERIMENTS_ENABLE_CHATBOT:*)
      chatbot_entry_count=$((chatbot_entry_count + 1))
      if [[ "$cache_line" == "SNNBASE_EXPERIMENTS_ENABLE_CHATBOT:BOOL=ON" ]]; then
        chatbot_enabled=true
      fi
      ;;
  esac
done <"$cache_path"

if [[ $torch_entry_count -eq 0 ]]; then
  fail "build directory does not record Torch_DIR"
fi
if [[ $torch_entry_count -ne 1 ]]; then
  fail "CMake cache contains conflicting Torch_DIR entries"
fi
[[ "$torch_entry_type" == "PATH" && -n "$torch_dir" ]] ||
  fail "CMake cache has a malformed Torch_DIR entry"
[[ "$torch_dir" != *-NOTFOUND ]] || fail "configured Torch_DIR was not found"
[[ "$torch_dir" == /* ]] || fail "configured Torch_DIR must be an absolute path"

if [[ $chatbot_entry_count -ne 1 || "$chatbot_enabled" != true ]]; then
  fail "build directory was not configured with chatbot support"
fi

[[ -d "$torch_dir" && -f "$torch_dir/TorchConfig.cmake" ]] ||
  fail "configured Torch_DIR is missing TorchConfig.cmake: $torch_dir"
torch_dir="$(cd -- "$torch_dir" && pwd -P)"
torch_library_candidate="$torch_dir/../../../lib"
[[ -d "$torch_library_candidate" ]] ||
  fail "configured Torch_DIR does not resolve to a LibTorch library directory"
torch_library_dir="$(cd -- "$torch_library_candidate" && pwd -P)"
[[ -f "$torch_library_dir/libtorch.so" ]] ||
  fail "configured LibTorch library directory is missing libtorch.so"
[[ "$torch_library_dir" != *:* ]] ||
  fail "configured LibTorch library path cannot contain ':'"

export LD_LIBRARY_PATH="${torch_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec -- "$@"
