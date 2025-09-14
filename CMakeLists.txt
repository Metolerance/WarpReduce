cmake_minimum_required(VERSION 3.26)
project(TEST CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


include_directories(include)

add_executable(TEST
        src/main.cu
        src/warp_reduce.cu
)

set_property(TARGET TEST PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET TEST PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
