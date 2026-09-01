# Use with -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-cuda121-gcc11.cmake.
# The image in containers/temporal.Dockerfile supplies these paths. Override
# any value before configuring when the host stores CUDA/GCC elsewhere.
set(CMAKE_C_COMPILER /usr/bin/gcc-11 CACHE FILEPATH "CUDA 12.1 host C compiler")
set(CMAKE_CXX_COMPILER /usr/bin/g++-11 CACHE FILEPATH "CUDA 12.1 host C++ compiler")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc CACHE FILEPATH "CUDA 12.1 compiler")
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11 CACHE FILEPATH "CUDA host compiler")
set(CUDAToolkit_ROOT /usr/local/cuda-12.1 CACHE PATH "CUDA 12.1 toolkit root")
# LibTorch 2.3 still consults CMake's legacy FindCUDA variable while newer
# CMake/Torch packages use CUDAToolkit_ROOT. Keep both pinned to the same kit.
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.1 CACHE PATH "Legacy CUDA 12.1 toolkit root")

# The supported chatbot GPU is Ampere SM86. Pin both CMake's native CUDA
# architecture and LibTorch's legacy architecture selector so Torch 2.3 does
# not try to infer unsupported architectures from a newer host toolkit.
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")
set(TORCH_CUDA_ARCH_LIST 8.6 CACHE STRING "LibTorch CUDA architectures")
