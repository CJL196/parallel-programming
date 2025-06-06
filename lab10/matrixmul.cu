#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cstring>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 基础全局内存矩阵乘法核函数
__global__ void matrixMulBasic(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// 使用共享内存优化的矩阵乘法核函数
template<int TILE_SIZE>
__global__ void matrixMulShared(float* A, float* B, float* C, int m, int n, int k) {
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
            
        if (col < k && t * TILE_SIZE + ty < n)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * k + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < k) {
        C[row * k + col] = sum;
    }
}

// 1D分块版本 - 按行划分任务
__global__ void matrixMul1DRow(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m) {
        for (int col = 0; col < k; col++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += A[row * n + i] * B[i * k + col];
            }
            C[row * k + col] = sum;
        }
    }
}

// 1D分块版本 - 按列划分任务
__global__ void matrixMul1DCol(float* A, float* B, float* C, int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < k) {
        for (int row = 0; row < m; row++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += A[row * n + i] * B[i * k + col];
            }
            C[row * k + col] = sum;
        }
    }
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <m> <n> <k> <block_size> <memory_type> <partition_type>" << std::endl;
    std::cout << "  m, n, k: matrix dimensions (A: m×n, B: n×k)" << std::endl;
    std::cout << "  block_size: 8, 16, or 32" << std::endl;
    std::cout << "  memory_type: global or shared" << std::endl;
    std::cout << "  partition_type: 2d, 1d_row, or 1d_col" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        printUsage(argv[0]);
        return 1;
    }

    // GPU预热
    {
        const int warmup_size = 128;
        float *w_A, *w_B, *w_C;
        float *d_w_A, *d_w_B, *d_w_C;

        // 分配预热用的内存
        w_A = new float[warmup_size * warmup_size];
        w_B = new float[warmup_size * warmup_size];
        w_C = new float[warmup_size * warmup_size];
        
        CHECK_CUDA(cudaMalloc(&d_w_A, warmup_size * warmup_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_w_B, warmup_size * warmup_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_w_C, warmup_size * warmup_size * sizeof(float)));

        // 初始化预热数据
        for (int i = 0; i < warmup_size * warmup_size; i++) {
            w_A[i] = 1.0f;
            w_B[i] = 1.0f;
        }

        CHECK_CUDA(cudaMemcpy(d_w_A, w_A, warmup_size * warmup_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w_B, w_B, warmup_size * warmup_size * sizeof(float), cudaMemcpyHostToDevice));

        // 执行几次预热运算
        dim3 warmup_block(16, 16);
        dim3 warmup_grid((warmup_size + 15) / 16, (warmup_size + 15) / 16);
        
        for (int i = 0; i < 5; i++) {
            matrixMulBasic<<<warmup_grid, warmup_block>>>(d_w_A, d_w_B, d_w_C, 
                warmup_size, warmup_size, warmup_size);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // 清理预热用的内存
        delete[] w_A;
        delete[] w_B;
        delete[] w_C;
        cudaFree(d_w_A);
        cudaFree(d_w_B);
        cudaFree(d_w_C);
    }
    
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int blockSize = atoi(argv[4]);
    std::string memoryType = argv[5];
    std::string partitionType = argv[6];
    
    // 验证输入参数
    // if (m < 128 || m > 2048 || n < 128 || n > 2048 || k < 128 || k > 2048) {
    //     std::cerr << "Matrix dimensions must be between 128 and 2048" << std::endl;
    //     return 1;
    // }
    
    if (blockSize != 8 && blockSize != 16 && blockSize != 32) {
        std::cerr << "Block size must be 8, 16, or 32" << std::endl;
        return 1;
    }
    
    // 分配内存并初始化
    float *h_A = new float[m * n];
    float *h_B = new float[n * k];
    float *h_C = new float[m * k];
    float *d_A, *d_B, *d_C;
    
    CHECK_CUDA(cudaMalloc(&d_A, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, n * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, m * k * sizeof(float)));
    
    // 随机初始化矩阵
    std::random_device rd;
    std::mt19937 gen(42); // 固定种子确保结果可复现
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < m * n; i++) h_A[i] = dis(gen);
    for (int i = 0; i < n * k; i++) h_B[i] = dis(gen);
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, n * k * sizeof(float), cudaMemcpyHostToDevice));
    
    // 执行计算并测量时间
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    
    if (memoryType == "global") {
        if (partitionType == "2d") {
            dim3 dimBlock(blockSize, blockSize);
            dim3 dimGrid((k + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);
            matrixMulBasic<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k);
        } else if (partitionType == "1d_row") {
            dim3 dimBlock(blockSize * blockSize);
            dim3 dimGrid((m + dimBlock.x - 1) / dimBlock.x);
            matrixMul1DRow<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k);
        } else if (partitionType == "1d_col") {
            dim3 dimBlock(blockSize * blockSize);
            dim3 dimGrid((k + dimBlock.x - 1) / dimBlock.x);
            matrixMul1DCol<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k);
        }
    } else if (memoryType == "shared") {
        if (blockSize == 8) {
            dim3 dimBlock(8, 8);
            dim3 dimGrid((k + 7) / 8, (m + 7) / 8);
            matrixMulShared<8><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k);
        } else if (blockSize == 16) {
            dim3 dimBlock(16, 16);
            dim3 dimGrid((k + 15) / 16, (m + 15) / 16);
            matrixMulShared<16><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k);
        } else if (blockSize == 32) {
            dim3 dimBlock(32, 32);
            dim3 dimGrid((k + 31) / 32, (m + 31) / 32);
            matrixMulShared<32><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k);
        }
    } 
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // 拷贝结果
    CHECK_CUDA(cudaMemcpy(h_C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 计算性能指标
    double gflops = (2.0 * m * n * k) / (milliseconds * 1e6);
    
    // 输出结果（格式化以便Python解析）
    std::cout << "RESULT," << m << "," << n << "," << k << "," 
              << blockSize << "," << memoryType << "," << partitionType << ","
              << milliseconds << "," << gflops << std::endl;
    
    // 清理内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}