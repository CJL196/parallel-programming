cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(CUDA_Freshman CXX C CUDA)

# 设置模块路径，让 CMake 能找到我们的 FindCUDNN.cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# Find required packages
find_package(CUDA REQUIRED)
find_package(CUDNN REQUIRED)

set(CMAKE_CUDA_FLAGS "-arch=compute_35 -g -G -O3")
add_subdirectory(task1)
add_subdirectory(task2)
add_subdirectory(task3)