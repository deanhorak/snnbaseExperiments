# Reproducible build/runtime image for the LibTorch temporal experiments.
# Build from the experiments repository root:
#   docker build -f containers/temporal.Dockerfile -t snnbase-temporal:cu121 .
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04@sha256:21196d81f56b48dbee70494d5f10322e1a77cc47ffe202a3bf68eab81533c20f

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates cmake g++-11 gcc-11 git ninja-build python3-pip \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.5.1's cu121 wheel provides the LibTorch CMake package used by the
# C++ temporal backend. Keep the compiler ABI and toolkit in the supported
# CUDA 12.1/GCC 11 combination.
RUN python3 -m pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

ENV CC=/usr/bin/gcc-11 \
    CXX=/usr/bin/g++-11 \
    CUDAHOSTCXX=/usr/bin/g++-11 \
    CUDAToolkit_ROOT=/usr/local/cuda-12.1 \
    TORCH_CMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake \
    LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib

WORKDIR /workspace
