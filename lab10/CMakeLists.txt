cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_Freshman CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -g -G -O3")
add_executable(matrixmul matrixmul.cu)