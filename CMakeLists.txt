cmake_minimum_required(VERSION 3.26)
project(WarpReduce CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

add_executable(WarpReduce
        src/main.cu
        src/warp_reduce.cu
        src/warp_reduce_with_padding.cu
)

set_property(TARGET WarpReduce PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET WarpReduce PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
