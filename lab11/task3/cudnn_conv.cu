#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUDNN(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)


void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <input_size> <stride>" << std::endl;
    std::cout << "  input_size: e.g., 32, 256, 512" << std::endl;
    std::cout << "  stride: 1, 2, or 3" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printUsage(argv[0]);
        return 1;
    }

    int input_size = std::atoi(argv[1]);
    int stride = std::atoi(argv[2]);

    // --- cuDNN Handle Initialization ---
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // --- Convolution Parameters ---
    const int N = 1, C = 3, H = input_size, W = input_size;
    const int K = 3, FH = 3, FW = 3; // K: num output channels, F: filter
    const int SH = stride, SW = stride;
    
    // Using "SAME" padding logic
    const int PH = (H * (SH - 1) + FH - SH) / 2;
    const int PW = (W * (SW - 1) + FW - SW) / 2;

    int out_N, out_C, out_H, out_W;

    // --- Tensor Descriptors ---
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t kernel_desc;
    
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernel_desc));
    
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, FH, FW));

    // --- Convolution Descriptor ---
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // --- Get Output Dimensions ---
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, kernel_desc, &out_N, &out_C, &out_H, &out_W));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_N, out_C, out_H, out_W));

    std::cout << "cuDNN Convolution" << std::endl;
    std::cout << "Input Size: " << H << "x" << W << ", Channels: " << C << std::endl;
    std::cout << "Kernel Size: " << FH << "x" << FW << ", Stride: " << stride << ", Output Channels: " << K << std::endl;
    std::cout << "Output Size: " << out_H << "x" << out_W << std::endl;

    // --- Algorithm Selection ---
    cudnnConvolutionFwdAlgo_t algo;
    
    int requestedAlgoCount = 1;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
        cudnn, 
        input_desc, 
        kernel_desc, 
        conv_desc, 
        output_desc, 
        requestedAlgoCount, 
        &returnedAlgoCount, 
        &perfResults));
    
    algo = perfResults.algo;
    std::cout << "Selected algorithm: " << algo << std::endl;

    // --- Workspace Management ---
    size_t workspace_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, kernel_desc, conv_desc, output_desc, algo, &workspace_bytes));
    void* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_bytes));
    }
    std::cout << "Workspace size: " << workspace_bytes / (1024.0 * 1024.0) << " MB" << std::endl;

    // --- Data Allocation and Initialization ---
    size_t input_bytes = N * C * H * W * sizeof(float);
    size_t kernel_bytes = K * C * FH * FW * sizeof(float);
    size_t output_bytes = out_N * out_C * out_H * out_W * sizeof(float);
    
    float *h_input, *h_kernel, *h_output;
    h_input = new float[N * C * H * W];
    h_kernel = new float[K * C * FH * FW];
    h_output = new float[out_N * out_C * out_H * out_W];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < N * C * H * W; ++i) h_input[i] = dis(gen);
    for (size_t i = 0; i < K * C * FH * FW; ++i) h_kernel[i] = dis(gen);

    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernel_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice));

    // --- GPU Warmup ---
    float alpha = 1.0f, beta = 0.0f;
    for (int i=0; i<5; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, kernel_desc, d_kernel, conv_desc, algo, d_workspace, workspace_bytes, &beta, output_desc, d_output));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- Execution and Timing ---
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, kernel_desc, d_kernel, conv_desc, algo, d_workspace, workspace_bytes, &beta, output_desc, d_output));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    
    // --- Copy result back and verify (optional) ---
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));
    std::cout << "Output_data[0] = " << h_output[0] << std::endl;
    std::cout << "Output_data[last] = " << h_output[out_N * out_C * out_H * out_W - 1] << std::endl;


    // --- Cleanup ---
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;
    
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output));
    if (d_workspace) CHECK_CUDA(cudaFree(d_workspace));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernel_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
} 