#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MIN_SIZE 128
#define MAX_SIZE 2048

// 定义矩阵尺寸结构体
typedef struct {
    int m;
    int n;
    int k;
} MatrixSize;

// 串行矩阵乘法函数
void serial_matrix_mult(double *A, double *B, double *C_serial, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C_serial[i * k + j] = 0.0;
            for (int p = 0; p < n; p++) {
                C_serial[i * k + j] += A[i * n + p] * B[p * k + j];
            }
        }
    }
}

// 验证结果是否正确
int verify_results(double *C_parallel, double *C_serial, int m, int k) {
    double epsilon = 1e-10; // 浮点数比较的容差
    for (int i = 0; i < m * k; i++) {
        if (fabs(C_parallel[i] - C_serial[i]) > epsilon) {
            printf("Verification failed at index %d: parallel=%f, serial=%f\n", 
                   i, C_parallel[i], C_serial[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[])
{
    int rank, size;
    MatrixSize matrix_size;
    double *A, *B, *C, *C_serial;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
        if (argc != 4)
        {
            printf("Usage: %s <m> <n> <k>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        matrix_size.m = atoi(argv[1]);
        matrix_size.n = atoi(argv[2]);
        matrix_size.k = atoi(argv[3]);
        if (
            matrix_size.m < MIN_SIZE || matrix_size.m > MAX_SIZE ||
            matrix_size.n < MIN_SIZE || matrix_size.n > MAX_SIZE ||
            matrix_size.k < MIN_SIZE || matrix_size.k > MAX_SIZE)
        {
            printf("Matrix size must be between 128 and 2048\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // 使用新的数据类型广播矩阵维度
    MPI_Bcast(&matrix_size, 1, MPI_MATRIX_SIZE, 0, MPI_COMM_WORLD);

    // 分配内存
    A = (double *)malloc(matrix_size.m * matrix_size.n * sizeof(double));
    B = (double *)malloc(matrix_size.n * matrix_size.k * sizeof(double));
    C = (double *)malloc(matrix_size.m * matrix_size.k * sizeof(double));
    C_serial = (double *)malloc(matrix_size.m * matrix_size.k * sizeof(double));

    // 初始化矩阵
    if (rank == 0)
    {
        srand(time(NULL));
        for (int i = 0; i < matrix_size.m * matrix_size.n; i++)
            A[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < matrix_size.n * matrix_size.k; i++)
            B[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < matrix_size.m * matrix_size.k; i++)
        C[i] = 0.0;

    // 记录开始时间
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // 广播矩阵A和B
    MPI_Bcast(A, matrix_size.m * matrix_size.n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, matrix_size.n * matrix_size.k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 计算每个进程处理的行数
    int rows_per_process = matrix_size.m / size;
    int remainder = matrix_size.m % size;
    int start_row = rank * rows_per_process + (rank < remainder ? rank : remainder);
    int extra_rows = (rank < remainder) ? 1 : 0;
    rows_per_process = rows_per_process + extra_rows;
    int end_row = start_row + rows_per_process;


    // 计算矩阵乘法
    for (int i = start_row; i < end_row && i < matrix_size.m; i++) {
        for (int j = 0; j < matrix_size.k; j++) {
            for (int p = 0; p < matrix_size.n; p++) {
                C[i * matrix_size.k + j] += A[i * matrix_size.n + p] * B[p * matrix_size.k + j];
            }
        }
    }

    // 收集所有进程的计算结果
    if (rank == 0) {
        // 进程0先复制自己的结果
        int *recvcounts = (int *)malloc(size * sizeof(int));
        int *displs = (int *)malloc(size * sizeof(int));
        
        for (int i = 0; i < size; i++) {
            recvcounts[i] = (matrix_size.m / size + (i < matrix_size.m % size ? 1 : 0)) * matrix_size.k;
            displs[i] = (i * (matrix_size.m / size) + (i < matrix_size.m % size ? i : matrix_size.m % size)) * matrix_size.k;
        }
        
        MPI_Gatherv(MPI_IN_PLACE, rows_per_process * matrix_size.k, MPI_DOUBLE,
                   C, recvcounts, displs, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        free(recvcounts);
        free(displs);
    } else {
        MPI_Gatherv(&C[start_row * matrix_size.k], rows_per_process * matrix_size.k, MPI_DOUBLE,
                   NULL, NULL, NULL, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    }

    // 记录结束时间
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // 输出结果
    if (rank == 0) {
        printf("Matrix multiplication completed in %f seconds\n", end_time - start_time);
    }

    // 在进程0上进行串行计算和验证
    if (rank == 0) {
        // 执行串行矩阵乘法
        serial_matrix_mult(A, B, C_serial, matrix_size.m, matrix_size.n, matrix_size.k);
        
        // 验证结果
        if (verify_results(C, C_serial, matrix_size.m, matrix_size.k)) {
            printf("Verification passed!\n");
        } else {
            printf("Verification failed!\n");
        }
    }

    // 释放内存
    free(A);
    free(B);
    free(C);
    free(C_serial);

    // 释放MPI数据类型
    MPI_Type_free(&MPI_MATRIX_SIZE);

    MPI_Finalize();
    return 0;
}