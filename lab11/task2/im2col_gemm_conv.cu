#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// 从 lab10/matrixmul.cu 复制并适配
// 使用共享内存优化的矩阵乘法核函数
template<int TILE_SIZE>
__global__ void matrixMulShared(const float* A, const float* B, float* C, int m, int n, int k) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载数据到共享内存
        if (row < m && t * TILE_SIZE + tx < n)
            As[ty][tx] = A[row * n + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (t * TILE_SIZE + ty < n && col < k)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * k + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        if (row < m && col < k) {
            for (int i = 0; i < TILE_SIZE; i++) {
                sum += As[ty][i] * Bs[i][tx];
            }
        }
        
        __syncthreads();
    }
    
    if (row < m && col < k) {
        C[row * k + col] = sum;
    }
}

// im2col kernel
__global__ void im2col_kernel(const float* input_data, int channels, int height, int width,
                              int kernel_h, int kernel_w, int stride_h, int stride_w,
                              int pad_h, int pad_w, int out_h, int out_w, float* col_data) {
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_y < out_h && out_x < out_w) {
        int col_index = out_y * out_w + out_x;
        for (int c = 0; c < channels; ++c) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int in_y = out_y * stride_h - pad_h + kh;
                    int in_x = out_x * stride_w - pad_w + kw;

                    int row_index = c * kernel_h * kernel_w + kh * kernel_w + kw;
                    int col_buffer_index = row_index * (out_h * out_w) + col_index;

                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        col_data[col_buffer_index] = input_data[c * height * width + in_y * width + in_x];
                    } else {
                        col_data[col_buffer_index] = 0.0f;
                    }
                }
            }
        }
    }
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <input_size> <kernel_size> <stride>" << std::endl;
    std::cout << "  input_size: e.g., 32, 256, 512" << std::endl;
    std::cout << "  kernel_size: e.g., 3" << std::endl;
    std::cout << "  stride: 1, 2, or 3" << std::endl;
}

void warmup_gpu() {
    const int warmup_size = 256;
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, warmup_size * warmup_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, warmup_size * warmup_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, warmup_size * warmup_size * sizeof(float)));

    dim3 grid((warmup_size + 15) / 16, (warmup_size + 15) / 16);
    dim3 block(16, 16);

    for (int i = 0; i < 5; ++i) {
        matrixMulShared<16><<<grid, block>>>(d_a, d_b, d_c, warmup_size, warmup_size, warmup_size);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }

    int input_size = atoi(argv[1]);
    int kernel_size = atoi(argv[2]);
    int stride = atoi(argv[3]);

    warmup_gpu();

    // Convolution parameters from requirement
    int in_channels = 3;
    int in_h = input_size;
    int in_w = input_size;
    int kernel_h = kernel_size;
    int kernel_w = kernel_size;
    // Let's assume number of output filters is also 3
    int out_channels = 3; 

    // For "SAME" padding
    int pad_h = (in_h * (stride - 1) + kernel_h - stride) / 2;
    int pad_w = (in_w * (stride - 1) + kernel_w - stride) / 2;

    int out_h = (in_h + 2 * pad_h - kernel_h) / stride + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride + 1;

    // Allocate host memory
    std::vector<float> h_input(in_channels * in_h * in_w);
    std::vector<float> h_kernel(out_channels * in_channels * kernel_h * kernel_w);
    std::vector<float> h_output(out_channels * out_h * out_w);

    // Initialize with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& val : h_input) val = dis(gen);
    for (auto& val : h_kernel) val = dis(gen);

    // Allocate device memory
    float *d_input, *d_kernel, *d_col, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, h_kernel.size() * sizeof(float)));
    
    size_t col_buffer_size = (size_t)in_channels * kernel_h * kernel_w * out_h * out_w;
    CHECK_CUDA(cudaMalloc(&d_col, col_buffer_size * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&d_output, h_output.size() * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel.data(), h_kernel.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // 1. im2col
    dim3 im2col_block(16, 16);
    dim3 im2col_grid((out_w + im2col_block.x - 1) / im2col_block.x, 
                     (out_h + im2col_block.y - 1) / im2col_block.y);
    im2col_kernel<<<im2col_grid, im2col_block>>>(d_input, in_channels, in_h, in_w, kernel_h, kernel_w, stride, stride, pad_h, pad_w, out_h, out_w, d_col);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // 2. GEMM
    int m = out_channels;
    int n = in_channels * kernel_h * kernel_w;
    int k = out_h * out_w;

    const int TILE_SIZE = 16;
    dim3 gemm_block(TILE_SIZE, TILE_SIZE);
    dim3 gemm_grid((k + gemm_block.x - 1) / gemm_block.x,
                   (m + gemm_block.y - 1) / gemm_block.y);

    // Kernel matrix: (out_channels, in_channels * kernel_h * kernel_w)
    // Col matrix: (in_channels * kernel_h * kernel_w, out_h * out_w)
    // Output matrix: (out_channels, out_h * out_w)
    matrixMulShared<TILE_SIZE><<<gemm_grid, gemm_block>>>(d_kernel, d_col, d_output, m, n, k);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print results
    std::cout << "im2col + GEMM Convolution" << std::endl;
    std::cout << "Input Size: " << in_h << "x" << in_w << ", Channels: " << in_channels << std::endl;
    std::cout << "Kernel Size: " << kernel_h << "x" << kernel_w << ", Stride: " << stride << ", Output Channels: " << out_channels << std::endl;
    std::cout << "Output Size: " << out_h << "x" << out_w << std::endl;
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    
    // Optional: Print a small part of the output for verification
    std::cout << "Output_data[0] = " << h_output[0] << std::endl;
    std::cout << "Output_data[last] = " << h_output.back() << std::endl;


    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_col);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
} 