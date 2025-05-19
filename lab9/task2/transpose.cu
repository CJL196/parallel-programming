#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string.h>

// 检查CUDA错误
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 基础矩阵转置核函数
__global__ void transposeNaive(float *input, float *output, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < n && y < n) {
        output[y * n + x] = input[x * n + y];
    }
}

// 使用共享内存的优化矩阵转置核函数
__global__ void transposeShared(float *input, float *output, int n) {
    __shared__ float tile[32][32];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = input[y * n + x];
    }
    
    __syncthreads();
    
    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (x < n && y < n) {
        output[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// 初始化矩阵
void initMatrix(float *matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// 验证转置结果
bool verifyTranspose(float *original, float *transposed, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(original[i * n + j] - transposed[j * n + i]) > 1e-5) {
                return false;
            }
        }
    }
    return true;
}

// GPU预热函数
void warmupGPU(int n, int blockSize, int kernelType) {
    // 分配测试用的内存
    float *h_input = (float*)malloc(n * n * sizeof(float));
    float *h_output = (float*)malloc(n * n * sizeof(float));
    float *d_input, *d_output;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * n * sizeof(float)));
    
    // 初始化输入矩阵
    initMatrix(h_input, n);
    
    // 设置线程块和网格大小
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);
    
    // 将数据复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, n * n * sizeof(float), cudaMemcpyHostToDevice));
    
    // 运行预热操作（运行5次）
    for (int i = 0; i < 5; i++) {
        if (kernelType == 0) {
            transposeNaive<<<gridDim, blockDim>>>(d_input, d_output, n);
        } else {
            transposeShared<<<gridDim, blockDim>>>(d_input, d_output, n);
        }
        cudaDeviceSynchronize();
    }
    
    // 清理资源
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
}

void printUsage() {
    printf("Usage: ./transpose <matrix_size> <block_size> <kernel_type>\n");
    printf("  matrix_size: size of the matrix (e.g., 512, 1024, 2048)\n");
    printf("  block_size: size of thread block (e.g., 16, 32)\n");
    printf("  kernel_type: 0 for naive, 1 for shared memory\n");
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printUsage();
        return 1;
    }

    int n = atoi(argv[1]);
    int blockSize = atoi(argv[2]);
    int kernelType = atoi(argv[3]);

    if (n <= 0 || blockSize <= 0 || (kernelType != 0 && kernelType != 1)) {
        printf("Invalid parameters!\n");
        printUsage();
        return 1;
    }
    
    // GPU预热
    warmupGPU(n, blockSize, kernelType);
    
    // 分配主机内存
    float *h_input = (float*)malloc(n * n * sizeof(float));
    float *h_output = (float*)malloc(n * n * sizeof(float));
    
    // 初始化输入矩阵
    srand(time(NULL));
    initMatrix(h_input, n);
    
    // 分配设备内存
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * n * sizeof(float)));
    
    // 将数据复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, n * n * sizeof(float), cudaMemcpyHostToDevice));
    
    // 设置线程块大小
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // 记录开始时间
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // 执行转置
    if (kernelType == 0) {
        transposeNaive<<<gridDim, blockDim>>>(d_input, d_output, n);
    } else {
        transposeShared<<<gridDim, blockDim>>>(d_input, d_output, n);
    }
    
    // 记录结束时间
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    // 计算执行时间
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // 将结果复制回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 验证结果
    if (verifyTranspose(h_input, h_output, n)) {
        printf("SUCCESS,%d,%d,%d,%.3f\n", n, blockSize, kernelType, milliseconds);
    } else {
        printf("FAILED,%d,%d,%d,%.3f\n", n, blockSize, kernelType, milliseconds);
    }
    
    // 清理资源
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
