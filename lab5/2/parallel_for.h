#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

#include <pthread.h>

// 调度策略枚举
typedef enum {
    STATIC,     // 静态调度
    DYNAMIC,    // 动态调度
    GUIDED      // 引导式调度
} ScheduleType;

// 线程参数结构体
typedef struct {
    int start;          // 循环开始索引
    int end;           // 循环结束索引
    int inc;           // 索引增量
    void *(*functor)(int, void*);  // 函数指针
    void *arg;         // 函数参数
    int thread_id;     // 线程ID
    int num_threads;   // 总线程数
    ScheduleType schedule_type;  // 调度类型
    int chunk_size;    // 块大小（用于动态和引导式调度）
    pthread_mutex_t *mutex;  // 互斥锁（用于动态调度）
    int *current_index;  // 当前索引（用于动态调度）
} ThreadArgs;

// 函数声明
void parallel_for(int start, int end, int inc,
                 void *(*functor)(int, void*), void *arg, 
                 int num_threads, ScheduleType schedule_type, int chunk_size);

#endif // PARALLEL_FOR_H 