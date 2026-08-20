# Use with -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-cuda121-gcc11.cmake.
# The image in containers/temporal.Dockerfile supplies these paths. Override
# any value before configuring when the host stores CUDA/GCC elsewhere.
set(CMAKE_C_COMPILER /usr/bin/gcc-11 CACHE FILEPATH "CUDA 12.1 host C compiler")
set(CMAKE_CXX_COMPILER /usr/bin/g++-11 CACHE FILEPATH "CUDA 12.1 host C++ compiler")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc CACHE FILEPATH "CUDA 12.1 compiler")
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11 CACHE FILEPATH "CUDA host compiler")
set(CUDAToolkit_ROOT /usr/local/cuda-12.1 CACHE PATH "CUDA 12.1 toolkit root")
