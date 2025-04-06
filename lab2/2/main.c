#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MIN_SIZE 128
#define MAX_SIZE 2048

// 定义矩阵尺寸结构体
typedef struct {
    int m;
    int n;
    int k;
} MatrixSize;

// 检查一个数是否为完全平方数
int is_perfect_square(int n) {
    int root = (int)sqrt(n);
    return root * root == n;
}

int main(int argc, char *argv[])
{
    int rank, size;
    MatrixSize matrix_size;
    double *A = NULL, *B = NULL, *C = NULL;
    double *C_parallel = NULL; // 进程0的并行结果和串行结果
    double *full_A = NULL, *full_B = NULL; // 用于进程0的完整矩阵
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 检查进程数是否为完全平方数
    if (!is_perfect_square(size)) {
        if (rank == 0) {
            printf("错误：进程数必须是完全平方数。当前进程数: %d\n", size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 计算块划分的行列数
    int block_size = (int)sqrt(size);
    int my_row = rank / block_size;
    int my_col = rank % block_size;

    // 创建MPI数据类型
    MPI_Datatype MPI_MATRIX_SIZE;
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    
    // 计算位移
    MPI_Aint base_address;
    MPI_Get_address(&matrix_size, &base_address);
    MPI_Get_address(&matrix_size.m, &displacements[0]);
    MPI_Get_address(&matrix_size.n, &displacements[1]);
    MPI_Get_address(&matrix_size.k, &displacements[2]);
    
    // 计算相对位移
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    displacements[2] = MPI_Aint_diff(displacements[2], base_address);
    
    // 创建结构体数据类型
    MPI_Type_create_struct(3, blocklengths, displacements, types, &MPI_MATRIX_SIZE);
    MPI_Type_commit(&MPI_MATRIX_SIZE);

    // 获取矩阵规模
    if (rank == 0)
    {
        if (argc != 2)
        {
            printf("Usage: %s <size>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int size = atoi(argv[1]);
        
        // 确保size可以被block_size整除
        if (size % block_size != 0) {
            printf("错误：矩阵大小必须能被块大小整除。\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        matrix_size.m = size;
        matrix_size.n = size;
        matrix_size.k = size;
        
        if (matrix_size.m < MIN_SIZE || matrix_size.m > MAX_SIZE)
        {
            printf("Matrix size must be between 128 and 2048\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // 使用新的数据类型广播矩阵维度
    MPI_Bcast(&matrix_size, 1, MPI_MATRIX_SIZE, 0, MPI_COMM_WORLD);

    // 计算每个块的大小
    int block_m = matrix_size.m / block_size;
    int block_n = matrix_size.n / block_size;
    int block_k = matrix_size.k / block_size;
    
    // 计算当前进程负责的块的起始位置
    int start_m = my_row * block_m;
    int start_n = my_col * block_n;
    int start_k = my_col * block_k;

    // 分配内存
    A = (double *)malloc(block_m * block_n * sizeof(double));
    B = (double *)malloc(block_n * block_k * sizeof(double));
    C = (double *)malloc(block_m * block_k * sizeof(double));
    
    if (A == NULL || B == NULL || C == NULL) {
        printf("进程 %d: 内存分配失败\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // 只有进程0需要分配完整矩阵和结果矩阵
    if (rank == 0) {
        full_A = (double *)malloc(matrix_size.m * matrix_size.n * sizeof(double));
        full_B = (double *)malloc(matrix_size.n * matrix_size.k * sizeof(double));
        C_parallel = (double *)malloc(matrix_size.m * matrix_size.k * sizeof(double));
        
        if (full_A == NULL || full_B == NULL || C_parallel == NULL) {
            printf("进程 0: 内存分配失败\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // 初始化矩阵
    if (rank == 0)
    {
        srand(time(NULL));
        for (int i = 0; i < matrix_size.m * matrix_size.n; i++)
            full_A[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < matrix_size.n * matrix_size.k; i++)
            full_B[i] = (double)rand() / RAND_MAX;
            
        // 复制进程0的块
        for (int r = 0; r < block_m; r++) {
            for (int c = 0; c < block_n; c++) {
                A[r * block_n + c] = full_A[r * matrix_size.n + c];
            }
        }
        
        for (int r = 0; r < block_n; r++) {
            for (int c = 0; c < block_k; c++) {
                B[r * block_k + c] = full_B[r * matrix_size.k + c];
            }
        }
    }
    
    // 初始化C矩阵
    for (int i = 0; i < block_m * block_k; i++)
        C[i] = 0.0;

    // 记录开始时间
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // 创建新的通信器用于块划分
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, my_col, my_row, &col_comm);

    // 分配矩阵块
    if (rank == 0) {
        // 进程0负责分配矩阵块
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                int target_rank = i * block_size + j;
                if (target_rank != 0) {
                    // 发送A矩阵块
                    double *temp_block = (double *)malloc(block_m * block_n * sizeof(double));
                    if (temp_block == NULL) {
                        printf("进程 0: 临时块内存分配失败\n");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    
                    for (int r = 0; r < block_m; r++) {
                        for (int c = 0; c < block_n; c++) {
                            temp_block[r * block_n + c] = full_A[(i * block_m + r) * matrix_size.n + (j * block_n + c)];
                        }
                    }
                    
                    MPI_Send(temp_block, block_m * block_n, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD);
                    free(temp_block);
                    
                    // 发送B矩阵块
                    temp_block = (double *)malloc(block_n * block_k * sizeof(double));
                    if (temp_block == NULL) {
                        printf("进程 0: 临时块内存分配失败\n");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    
                    for (int r = 0; r < block_n; r++) {
                        for (int c = 0; c < block_k; c++) {
                            temp_block[r * block_k + c] = full_B[(j * block_n + r) * matrix_size.k + (i * block_k + c)];
                        }
                    }
                    
                    MPI_Send(temp_block, block_n * block_k, MPI_DOUBLE, target_rank, 1, MPI_COMM_WORLD);
                    free(temp_block);
                }
            }
        }
    } else {
        // 其他进程接收自己的块
        MPI_Recv(A, block_m * block_n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B, block_n * block_k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 计算矩阵乘法
    for (int i = 0; i < block_m; i++) {
        for (int j = 0; j < block_k; j++) {
            C[i * block_k + j] = 0.0;  // 初始化结果元素
            for (int p = 0; p < block_n; p++) {
                C[i * block_k + j] += A[i * block_n + p] * B[p * block_k + j];
            }
        }
    }

    // 收集所有进程的计算结果
    if (rank == 0) {
        // 进程0先复制自己的结果
        for (int i = 0; i < block_m; i++) {
            for (int j = 0; j < block_k; j++) {
                C_parallel[i * matrix_size.k + j] = C[i * block_k + j];
            }
        }
        
        // 接收其他进程的结果
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                int source_rank = i * block_size + j;
                if (source_rank != 0) {
                    double *temp_block = (double *)malloc(block_m * block_k * sizeof(double));
                    if (temp_block == NULL) {
                        printf("进程 0: 临时块内存分配失败\n");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    
                    MPI_Recv(temp_block, block_m * block_k, MPI_DOUBLE, source_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    for (int r = 0; r < block_m; r++) {
                        for (int c = 0; c < block_k; c++) {
                            C_parallel[(i * block_m + r) * matrix_size.k + (j * block_k + c)] = temp_block[r * block_k + c];
                        }
                    }
                    
                    free(temp_block);
                }
            }
        }
    } else {
        // 其他进程发送结果给进程0
        MPI_Send(C, block_m * block_k, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    // 记录结束时间
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // 输出结果
    if (rank == 0) {
        printf("Matrix multiplication completed in %f seconds\n", end_time - start_time);
    }

    

    // 释放内存
    free(A);
    free(B);
    free(C);
    if (rank == 0) {
        free(C_parallel);
        free(full_A);
        free(full_B);
    }

    // 释放MPI数据类型和通信器
    MPI_Type_free(&MPI_MATRIX_SIZE);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();
    return 0;
}