#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MIN_SIZE 128
#define MAX_SIZE 2048


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
    int rank, size, m, n, k;
    double *A, *B, *C, *C_serial;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 获取矩阵规模
    if (rank == 0)
    {
        if (argc != 4)
        {
            printf("Usage: %s <m> <n> <k>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
        if (
            m < MIN_SIZE || m > MAX_SIZE ||
            n < MIN_SIZE || n > MAX_SIZE ||
            k < MIN_SIZE || k > MAX_SIZE)
        {
            printf("Matrix size must be between 128 and 2048\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // 使用点对点通信广播矩阵维度
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Send(&m, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&n, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&k, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&n, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&k, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 分配内存
    A = (double *)malloc(m * n * sizeof(double));
    B = (double *)malloc(n * k * sizeof(double));
    C = (double *)malloc(m * k * sizeof(double));
    C_serial = (double *)malloc(m * k * sizeof(double));

    // 初始化矩阵
    if (rank == 0)
    {
        srand(time(NULL));
        for (int i = 0; i < m * n; i++)
            A[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < n * k; i++)
            B[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < m * k; i++)
        C[i] = 0.0;

    // 使用点对点通信广播矩阵A和B
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Send(A, m * n, MPI_DOUBLE, i, 3, MPI_COMM_WORLD);
            MPI_Send(B, n * k, MPI_DOUBLE, i, 4, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(A, m * n, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B, n * k, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 计算每个进程处理的行数
    int rows_per_process = m / size;
    int remainder = m % size;
    int start_row = rank * rows_per_process + (rank < remainder ? rank : remainder);
    int extra_rows = (rank < remainder) ? 1 : 0;
    rows_per_process = rows_per_process + extra_rows;
    int end_row = start_row + rows_per_process;

    // 记录开始时间
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // 计算矩阵乘法
    for (int i = start_row; i < end_row && i < m; i++) {
        for (int j = 0; j < k; j++) {
            for (int p = 0; p < n; p++) {
                C[i * k + j] += A[i * n + p] * B[p * k + j];
            }
        }
    }

    // 使用点对点通信收集结果
    if (rank == 0) {
        // 进程0接收其他进程的结果
        for (int i = 1; i < size; i++) {
            int other_start_row = i * (m / size) + (i < m % size ? i : m % size);
            int other_rows = m / size + (i < m % size ? 1 : 0);
            MPI_Recv(&C[other_start_row * k], other_rows * k, MPI_DOUBLE, 
                    i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        // 其他进程发送结果给进程0
        MPI_Send(&C[start_row * k], rows_per_process * k, MPI_DOUBLE, 
                0, 5, MPI_COMM_WORLD);
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
        serial_matrix_mult(A, B, C_serial, m, n, k);
        
        // 验证结果
        if (verify_results(C, C_serial, m, k)) {
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

    MPI_Finalize();
    return 0;
}