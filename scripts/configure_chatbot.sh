#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: configure_chatbot.sh --snnbase-source PATH --snnbase-revision COMMIT \
  --torch-prefix PATH [--build-dir PATH] [--build-type Debug|Release] \
  [--toolchain default|none|PATH]

Configures a chatbot build only from a clean snnbase checkout at the exact
40-character commit supplied by the caller. The commit must already contain
the snnbase::language backend. The default toolchain is the validated CUDA
12.1/GCC 11/SM86 matrix. Use --toolchain none with a CPU-only LibTorch prefix.
This script never fetches dependencies.
EOF
  exit 2
}

snnbase_source=""
snnbase_revision=""
torch_prefix=""
build_dir="build-chatbot"
build_type="Debug"
toolchain="default"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --snnbase-source)
      [[ $# -ge 2 ]] || usage
      snnbase_source="$2"
      shift 2
      ;;
    --snnbase-revision)
      [[ $# -ge 2 ]] || usage
      snnbase_revision="$2"
      shift 2
      ;;
    --torch-prefix)
      [[ $# -ge 2 ]] || usage
      torch_prefix="$2"
      shift 2
      ;;
    --build-dir)
      [[ $# -ge 2 ]] || usage
      build_dir="$2"
      shift 2
      ;;
    --build-type)
      [[ $# -ge 2 ]] || usage
      build_type="$2"
      shift 2
      ;;
    --toolchain)
      [[ $# -ge 2 ]] || usage
      toolchain="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ -n "$snnbase_source" && -n "$snnbase_revision" && -n "$torch_prefix" ]] || usage
[[ "$snnbase_revision" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$build_type" == "Debug" || "$build_type" == "Release" ]] || usage

command -v cmake >/dev/null 2>&1 || {
  echo "cmake is required" >&2
  exit 1
}
command -v ninja >/dev/null 2>&1 || {
  echo "ninja is required" >&2
  exit 1
}
command -v git >/dev/null 2>&1 || {
  echo "git is required" >&2
  exit 1
}

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
snnbase_source="$(cd -- "$snnbase_source" && pwd)"
[[ -e "$torch_prefix" ]] || {
  echo "Torch CMake prefix does not exist: $torch_prefix" >&2
  exit 1
}
if [[ "$build_dir" != /* ]]; then
  build_dir="$project_root/$build_dir"
fi
if [[ "$toolchain" == "default" ]]; then
  toolchain="$project_root/cmake/toolchains/linux-cuda121-gcc11.cmake"
elif [[ "$toolchain" != "none" ]]; then
  [[ -f "$toolchain" ]] || {
    echo "CMake toolchain does not exist: $toolchain" >&2
    exit 1
  }
  toolchain="$(realpath "$toolchain")"
fi

actual_revision="$(git -C "$snnbase_source" rev-parse HEAD)"
[[ "$actual_revision" == "$snnbase_revision" ]] || {
  echo "snnbase revision mismatch: expected $snnbase_revision, found $actual_revision" >&2
  exit 1
}
[[ -z "$(git -C "$snnbase_source" status --porcelain --untracked-files=all)" ]] || {
  echo "snnbase checkout must be clean: $snnbase_source" >&2
  exit 1
}
[[ -f "$snnbase_source/include/snnbase/language/decoder.hpp" ]] || {
  echo "the pinned snnbase commit does not contain the language backend" >&2
  exit 1
}

# Do not let a shell-level LibTorch/CUDA override select ABI-incompatible
# runtime libraries while CMake is probing the explicitly pinned prefix.
cmake_arguments=(
  -S "$project_root"
  -B "$build_dir"
  -G Ninja
  -DCMAKE_BUILD_TYPE="$build_type"
  -DCMAKE_PREFIX_PATH="$torch_prefix"
  -DSNNBASE_EXPERIMENTS_ENABLE_TEMPORAL=OFF
  -DSNNBASE_EXPERIMENTS_ENABLE_CHATBOT=ON
  -DSNNBASE_EXPERIMENTS_ENABLE_GUI=OFF
  -DSNNBASE_SOURCE_DIR="$snnbase_source"
  -DSNNBASE_EXPECTED_SNNBASE_REVISION="$snnbase_revision"
  -DSNNBASE_REQUIRE_CLEAN_SNNBASE=ON
)
if [[ "$toolchain" != "none" ]]; then
  cmake_arguments+=("-DCMAKE_TOOLCHAIN_FILE=$toolchain")
fi
env -u LD_LIBRARY_PATH cmake "${cmake_arguments[@]}"

echo "Configured $build_dir"
echo "Next: $project_root/scripts/run_chatbot_smoke.sh --build-dir $build_dir"
