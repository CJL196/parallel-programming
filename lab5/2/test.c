#include "parallel_for.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// 矩阵乘法参数结构体
typedef struct {
    float *A;
    float *B;
    float *C;
    int n;
    int k;
} MatrixArgs;

// 矩阵乘法函数
void* matrix_multiply(int i, void* arg) {
    MatrixArgs* args = (MatrixArgs*)arg;
    int n = args->n;
    int k = args->k;
    
    for(int j = 0; j < k; j++) {
        float sum = 0.0f;
        for(int l = 0; l < n; l++) {
            sum += args->A[i * n + l] * args->B[l * k + j];
        }
        args->C[i * k + j] = sum;
    }
    return NULL;
}

// 获取当前时间（微秒）
long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// 初始化矩阵
void init_matrix(float* matrix, int rows, int cols) {
    for(int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// 验证结果
int verify_result(float* A, float* B, float* C, int m, int n, int k) {
    float* C_serial = (float*)malloc(m * k * sizeof(float));
    
    // 串行计算
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < k; j++) {
            float sum = 0.0f;
            for(int l = 0; l < n; l++) {
                sum += A[i * n + l] * B[l * k + j];
            }
            C_serial[i * k + j] = sum;
        }
    }
    
    // 比较结果
    for(int i = 0; i < m * k; i++) {
        if(fabs(C[i] - C_serial[i]) > 1e-5) {
            free(C_serial);
            return 0;
        }
    }
    
    free(C_serial);
    return 1;
}

int main() {
    srand(time(NULL));
    
    // 矩阵维度
    int m = 1024;  // A的行数
    int n = 1024;  // A的列数，B的行数
    int k = 1024;  // B的列数
    
    // 分配内存
    float* A = (float*)malloc(m * n * sizeof(float));
    float* B = (float*)malloc(n * k * sizeof(float));
    float* C = (float*)malloc(m * k * sizeof(float));
    
    // 初始化矩阵
    init_matrix(A, m, n);
    init_matrix(B, n, k);
    
    // 测试不同调度策略
    ScheduleType schedules[] = {STATIC, DYNAMIC, GUIDED};
    const char* schedule_names[] = {"Static", "Dynamic", "Guided"};
    int num_threads[] = {1, 2, 4, 8, 16};
    int chunk_sizes[] = {1, 16, 64, 256};
    
    printf("Matrix size: %dx%d\n", m, k);
    printf("Testing different scheduling strategies...\n\n");
    
    for(int s = 0; s < 3; s++) {
        printf("Schedule: %s\n", schedule_names[s]);
        printf("Threads\tChunk Size\tTime (ms)\tSpeedup\n");
        printf("------------------------------------------------\n");
        
        // 基准时间（单线程）
        MatrixArgs args = {A, B, C, n, k};
        long start_time = get_time();
        parallel_for(0, m, 1, matrix_multiply, &args, 1, schedules[s], 1);
        long base_time = get_time() - start_time;
        
        for(int t = 0; t < 5; t++) {
            for(int c = 0; c < 4; c++) {
                // 清空结果矩阵
                memset(C, 0, m * k * sizeof(float));
                
                // 测试并行版本
                start_time = get_time();
                parallel_for(0, m, 1, matrix_multiply, &args, 
                           num_threads[t], schedules[s], chunk_sizes[c]);
                long parallel_time = get_time() - start_time;
                
                // 验证结果
                if(!verify_result(A, B, C, m, n, k)) {
                    printf("Error: Results don't match!\n");
                    return 1;
                }
                
                // 计算加速比
                float speedup = (float)base_time / parallel_time;
                
                printf("%d\t%d\t\t%.2f\t\t%.2f\n", 
                       num_threads[t], chunk_sizes[c], 
                       parallel_time / 1000.0, speedup);
            }
        }
        printf("\n");
    }
    
    // 清理内存
    free(A);
    free(B);
    free(C);
    
    return 0;
} 