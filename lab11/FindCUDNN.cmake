# FindCUDNN.cmake
#
# 这个文件帮助 CMake 找到 CUDNN 库和头文件
#
# 定义以下变量：
#  - CUDNN_FOUND        - 如果找到 CUDNN 则为 True
#  - CUDNN_INCLUDE_DIRS - CUDNN 头文件的位置
#  - CUDNN_LIBRARIES    - CUDNN 库的位置

# 首先尝试从环境变量查找
find_path(CUDNN_INCLUDE_DIR cudnn.h
  HINTS
    ${CUDNN_ROOT}
    $ENV{CUDNN_ROOT}
    ${CUDA_TOOLKIT_ROOT_DIR}
    $ENV{CUDA_PATH}
  PATH_SUFFIXES
    include
    cuda/include
)

# 寻找 CUDNN 库
find_library(CUDNN_LIBRARY
  NAMES cudnn
  HINTS
    ${CUDNN_ROOT}
    $ENV{CUDNN_ROOT}
    ${CUDA_TOOLKIT_ROOT_DIR}
    $ENV{CUDA_PATH}
  PATH_SUFFIXES
    lib
    lib64
    cuda/lib
    cuda/lib64
)

# 处理查找结果
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  CUDNN
  FOUND_VAR CUDNN_FOUND
  REQUIRED_VARS CUDNN_LIBRARY CUDNN_INCLUDE_DIR
)

# 设置输出变量
if(CUDNN_FOUND)
  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
  set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY) 