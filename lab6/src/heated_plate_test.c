# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <unistd.h>

int main(int argc, char *argv[]) {
    int thread_counts[] = {1, 2, 4, 8, 16};
    char *schedule_types[] = {"s", "d", "g"};
    char *schedule_names[] = {"静态调度", "动态调度", "引导式调度"};
    int chunk_sizes[] = {10, 50, 100};
    int i, j, k;
    char command[256];
    
    printf("===== 热平板问题Pthread版性能测试 =====\n");
    
    // 编译程序
    system("gcc -o ./heated_plate_pthread ./heated_plate_pthread.c ./parallel_for.c -lpthread -lm");
    
    // 测试OpenMP版本
    printf("\n===== OpenMP版本性能测试 =====\n");
    for (i = 0; i < sizeof(thread_counts) / sizeof(thread_counts[0]); i++) {
        sprintf(command, "export OMP_NUM_THREADS=%d && ../reference/heated_plate_openmp", thread_counts[i]);
        printf("\n使用 %d 个线程的OpenMP版本：\n", thread_counts[i]);
        system(command);
    }
    
    // 测试不同线程数
    printf("\n===== 测试不同线程数的性能 (使用静态调度) =====\n");
    for (i = 0; i < sizeof(thread_counts) / sizeof(thread_counts[0]); i++) {
        sprintf(command, "./heated_plate_pthread %d s 10", thread_counts[i]);
        printf("\n使用 %d 个线程：\n", thread_counts[i]);
        system(command);
    }
    
    // 测试不同调度方式
    printf("\n===== 测试不同调度方式的性能 (使用4线程) =====\n");
    for (i = 0; i < sizeof(schedule_types) / sizeof(schedule_types[0]); i++) {
        sprintf(command, "./heated_plate_pthread 4 %s 10", schedule_types[i]);
        printf("\n使用%s：\n", schedule_names[i]);
        system(command);
    }
    
    // 测试不同块大小
    printf("\n===== 测试不同块大小的性能 (使用4线程，动态调度) =====\n");
    for (i = 0; i < sizeof(chunk_sizes) / sizeof(chunk_sizes[0]); i++) {
        sprintf(command, "./heated_plate_pthread 4 d %d", chunk_sizes[i]);
        printf("\n使用块大小 %d：\n", chunk_sizes[i]);
        system(command);
    }
    
    // 综合测试
    printf("\n===== 综合性能测试 =====\n");
    for (i = 0; i < sizeof(thread_counts) / sizeof(thread_counts[0]); i++) {
        for (j = 0; j < sizeof(schedule_types) / sizeof(schedule_types[0]); j++) {
            for (k = 0; k < sizeof(chunk_sizes) / sizeof(chunk_sizes[0]); k++) {
                sprintf(command, "./heated_plate_pthread %d %s %d", 
                        thread_counts[i], schedule_types[j], chunk_sizes[k]);
                printf("\n线程数: %d, 调度: %s, 块大小: %d\n", 
                       thread_counts[i], schedule_names[j], chunk_sizes[k]);
                system(command);
            }
        }
    }
    
    return 0;
} 