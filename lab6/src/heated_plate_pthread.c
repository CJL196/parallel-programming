# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <pthread.h>
# include "parallel_for.h"

int main ( int argc, char *argv[] );
void *set_boundary_top(int i, void *args);
void *set_boundary_bottom(int i, void *args);
void *set_boundary_left(int i, void *args);
void *set_boundary_right(int i, void *args);
void *init_interior(int i, void *args);
void *copy_solution(int i, void *args);
void *compute_solution(int i, void *args);
void *compute_diff(int i, void *args);

# define M 500
# define N 500

typedef struct {
    double u[M][N];
    double w[M][N];
    double mean;
    double diff;
    double my_diff;
    pthread_mutex_t mutex;  // 差异更新的互斥锁
} PlateData;

/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for HEATED_PLATE_PTHREAD.

  Discussion:

    This code solves the steady state heat equation on a rectangular region.

    The physical region, and the boundary conditions, are suggested
    by this diagram;

                   W = 0
             +------------------+
             |                  |
    W = 100  |                  | W = 100
             |                  |
             +------------------+
                   W = 100

    The region is covered with a grid of M by N nodes, and an N by N
    array W is used to record the temperature.  The correspondence between
    array indices and locations in the region is suggested by giving the
    indices of the four corners:

                  I = 0
          [0][0]-------------[0][N-1]
             |                  |
      J = 0  |                  |  J = N-1
             |                  |
        [M-1][0]-----------[M-1][N-1]
                  I = M-1

    The steady state solution to the discrete heat equation satisfies the
    following condition at an interior grid point:

      W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    where "Central" is the index of the grid point, "North" is the index
    of its immediate neighbor to the "north", and so on.
   
    Given an approximate solution of the steady state heat equation, a
    "better" solution is given by replacing each interior point by the
    average of its 4 neighbors - in other words, by using the condition
    as an ASSIGNMENT statement:

      W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    If this process is repeated often enough, the difference between successive 
    estimates of the solution will go to zero.
*/
{
  PlateData data;
  double epsilon = 0.001;
  int i;
  int iterations;
  int iterations_print;
  int j;
  int num_threads = 4; // 默认4个线程，可通过命令行参数修改
  ScheduleType schedule_type = STATIC; // 默认静态调度
  int chunk_size = 10; // 默认块大小
  
  // 处理命令行参数
  if (argc > 1) {
    num_threads = atoi(argv[1]);
  }
  if (argc > 2) {
    if (argv[2][0] == 's' || argv[2][0] == 'S')
      schedule_type = STATIC;
    else if (argv[2][0] == 'd' || argv[2][0] == 'D')
      schedule_type = DYNAMIC;
    else if (argv[2][0] == 'g' || argv[2][0] == 'G')
      schedule_type = GUIDED;
  }
  if (argc > 3) {
    chunk_size = atoi(argv[3]);
  }

  printf("\n");
  printf("HEATED_PLATE_PTHREAD\n");
  printf("  C/Pthread version\n");
  printf("  A program to solve for the steady state temperature distribution\n");
  printf("  over a rectangular plate.\n");
  printf("\n");
  printf("  Spatial grid of %d by %d points.\n", M, N);
  printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
  printf("  Number of threads = %d\n", num_threads);
  printf("  Schedule type = %s\n", 
         schedule_type == STATIC ? "STATIC" : 
         (schedule_type == DYNAMIC ? "DYNAMIC" : "GUIDED"));
  printf("  Chunk size = %d\n", chunk_size);

  // 初始化差异值和互斥锁
  data.diff = 0.0;
  data.my_diff = 0.0;
  data.mean = 0.0;
  pthread_mutex_init(&data.mutex, NULL);

  /*
    Set the boundary values, which don't change.
  */
  // 设置上边界 w[0][j] = 0.0
  parallel_for(0, N, 1, set_boundary_top, &data, num_threads, schedule_type, chunk_size);
  
  // 设置下边界 w[M-1][j] = 100.0
  parallel_for(0, N, 1, set_boundary_bottom, &data, num_threads, schedule_type, chunk_size);
  
  // 设置左边界 w[i][0] = 100.0
  parallel_for(1, M-1, 1, set_boundary_left, &data, num_threads, schedule_type, chunk_size);
  
  // 设置右边界 w[i][N-1] = 100.0
  parallel_for(1, M-1, 1, set_boundary_right, &data, num_threads, schedule_type, chunk_size);

  /*
    计算平均边界值，作为内部节点的初始值
  */
  // 计算边界值和
  for (j = 0; j < N; j++) {
    data.mean += data.w[0][j] + data.w[M-1][j];
  }
  for (i = 1; i < M-1; i++) {
    data.mean += data.w[i][0] + data.w[i][N-1];
  }
  data.mean = data.mean / (double)(2 * M + 2 * N - 4);
  printf("\n");
  printf("  MEAN = %f\n", data.mean);

  // 初始化内部节点
  parallel_for(1, M-1, 1, init_interior, &data, num_threads, schedule_type, chunk_size);

  /*
    迭代直到新解 W 与旧解 U 之间的差异小于 EPSILON
  */
  clock_t start_time = clock();
  iterations = 0;
  iterations_print = 1;
  printf("\n");
  printf(" Iteration  Change\n");
  printf("\n");

  data.diff = epsilon;

  while (epsilon <= data.diff) {
    // 保存旧解到 U
    parallel_for(0, M, 1, copy_solution, &data, num_threads, schedule_type, chunk_size);

    // 计算新解 W
    parallel_for(1, M-1, 1, compute_solution, &data, num_threads, schedule_type, chunk_size);

    // 计算差异 
    data.diff = 0.0;
    parallel_for(1, M-1, 1, compute_diff, &data, num_threads, schedule_type, chunk_size);

    iterations++;
    if (iterations == iterations_print) {
      printf("  %8d  %f\n", iterations, data.diff);
      iterations_print = 2 * iterations_print;
    }
  }
  
  clock_t end_time = clock();
  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  printf("\n");
  printf("  %8d  %f\n", iterations, data.diff);
  printf("\n");
  printf("  Error tolerance achieved.\n");
  printf("  Elapsed time = %f seconds\n", elapsed_time);

  /*
    结束程序
  */
  printf("\n");
  printf("HEATED_PLATE_PTHREAD:\n");
  printf("  Normal end of execution.\n");

  // 销毁互斥锁
  pthread_mutex_destroy(&data.mutex);

  return 0;
}

// 设置上边界函数
void *set_boundary_top(int j, void *args) {
  PlateData *data = (PlateData *)args;
  data->w[0][j] = 0.0;
  return NULL;
}

// 设置下边界函数
void *set_boundary_bottom(int j, void *args) {
  PlateData *data = (PlateData *)args;
  data->w[M-1][j] = 100.0;
  return NULL;
}

// 设置左边界函数
void *set_boundary_left(int i, void *args) {
  PlateData *data = (PlateData *)args;
  data->w[i][0] = 100.0;
  return NULL;
}

// 设置右边界函数
void *set_boundary_right(int i, void *args) {
  PlateData *data = (PlateData *)args;
  data->w[i][N-1] = 100.0;
  return NULL;
}

// 初始化内部节点函数
void *init_interior(int i, void *args) {
  PlateData *data = (PlateData *)args;
  for (int j = 1; j < N-1; j++) {
    data->w[i][j] = data->mean;
  }
  return NULL;
}

// 复制解决方案函数
void *copy_solution(int i, void *args) {
  PlateData *data = (PlateData *)args;
  for (int j = 0; j < N; j++) {
    data->u[i][j] = data->w[i][j];
  }
  return NULL;
}

// 计算新解函数
void *compute_solution(int i, void *args) {
  PlateData *data = (PlateData *)args;
  for (int j = 1; j < N-1; j++) {
    data->w[i][j] = (data->u[i-1][j] + data->u[i+1][j] + 
                    data->u[i][j-1] + data->u[i][j+1]) / 4.0;
  }
  return NULL;
}

// 计算差异函数
void *compute_diff(int i, void *args) {
  PlateData *data = (PlateData *)args;
  double local_max = 0.0;
  
  // 计算行内最大差异
  for (int j = 1; j < N-1; j++) {
    double diff = fabs(data->w[i][j] - data->u[i][j]);
    if (local_max < diff) {
      local_max = diff;
    }
  }
  
  // 使用互斥锁更新全局最大差异
  pthread_mutex_lock(&data->mutex);
  if (data->diff < local_max) {
    data->diff = local_max;
  }
  pthread_mutex_unlock(&data->mutex);
  
  return NULL;
} 