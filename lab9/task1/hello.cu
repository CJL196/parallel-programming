#include <stdio.h>

// CUDA核函数，每个线程都会执行这个函数
__global__ void helloWorld() {
    // 获取线程块ID
    int blockId = blockIdx.x;
    // 获取线程在块内的二维坐标
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    
    // 输出线程信息
    printf("Hello World from Thread (%d, %d) in Block %d!\n", 
           threadX, threadY, blockId);
}

int main() {
    int n, m, k;
    
    // 读取输入参数
    printf("请输入三个整数n, m, k (范围[1, 32]): ");
    scanf("%d %d %d", &n, &m, &k);
    
    // 验证输入范围
    if (n < 1 || n > 32 || m < 1 || m > 32 || k < 1 || k > 32) {
        printf("输入参数必须在[1, 32]范围内！\n");
        return 1;
    }
    
    // 设置线程块和线程的维度
    dim3 blockDim(m, k);  // 每个线程块的维度为 m x k
    int numBlocks = n;    // 线程块数量为 n
    
    // 启动核函数
    helloWorld<<<numBlocks, blockDim>>>();
    
    // 等待所有线程完成
    cudaDeviceSynchronize();
    
    // 主线程输出
    printf("Hello World from the host!\n");
    
    return 0;
}
