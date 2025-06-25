add_executable(cudnn_conv cudnn_conv.cu)

# Link the executable against the cuDNN library
target_link_libraries(cudnn_conv PRIVATE ${CUDNN_LIBRARIES})

# Add the cuDNN include directory to the target's include paths
target_include_directories(cudnn_conv PRIVATE ${CUDNN_INCLUDE_DIRS})