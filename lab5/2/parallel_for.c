#include "parallel_for.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// 线程函数
void* thread_function(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int i;
    
    switch(args->schedule_type) {
        case STATIC: {
            // 静态调度：每个线程处理固定大小的块
            int chunk = (args->end - args->start) / args->num_threads;
            int start = args->start + args->thread_id * chunk;
            int end = (args->thread_id == args->num_threads - 1) ? 
                     args->end : start + chunk;
            
            for(i = start; i < end; i += args->inc) {
                args->functor(i, args->arg);
            }
            break;
        }
        
        case DYNAMIC: {
            // 动态调度：线程动态获取工作块
            while(1) {
                pthread_mutex_lock(args->mutex);
                int current = *args->current_index;
                if(current >= args->end) {
                    pthread_mutex_unlock(args->mutex);
                    break;
                }
                *args->current_index = current + args->chunk_size;
                pthread_mutex_unlock(args->mutex);
                
                int end = (current + args->chunk_size > args->end) ? 
                         args->end : current + args->chunk_size;
                
                for(i = current; i < end; i += args->inc) {
                    args->functor(i, args->arg);
                }
            }
            break;
        }
        
        case GUIDED: {
            // 引导式调度：块大小逐渐减小
            while(1) {
                pthread_mutex_lock(args->mutex);
                int current = *args->current_index;
                if(current >= args->end) {
                    pthread_mutex_unlock(args->mutex);
                    break;
                }
                
                // 计算当前块大小
                int remaining = args->end - current;
                int chunk = (remaining + args->num_threads - 1) / args->num_threads;
                chunk = (chunk > args->chunk_size) ? args->chunk_size : chunk;
                
                *args->current_index = current + chunk;
                pthread_mutex_unlock(args->mutex);
                
                int end = current + chunk;
                for(i = current; i < end; i += args->inc) {
                    args->functor(i, args->arg);
                }
            }
            break;
        }
    }
    
    return NULL;
}

// parallel_for 函数实现
void parallel_for(int start, int end, int inc,
                 void *(*functor)(int, void*), void *arg, 
                 int num_threads, ScheduleType schedule_type, int chunk_size) {
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadArgs* thread_args = (ThreadArgs*)malloc(num_threads * sizeof(ThreadArgs));
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    int current_index = start;
    
    // 初始化线程参数
    for(int i = 0; i < num_threads; i++) {
        thread_args[i].start = start;
        thread_args[i].end = end;
        thread_args[i].inc = inc;
        thread_args[i].functor = functor;
        thread_args[i].arg = arg;
        thread_args[i].thread_id = i;
        thread_args[i].num_threads = num_threads;
        thread_args[i].schedule_type = schedule_type;
        thread_args[i].chunk_size = chunk_size;
        thread_args[i].mutex = &mutex;
        thread_args[i].current_index = &current_index;
    }
    
    // 创建线程
    for(int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, thread_function, &thread_args[i]);
    }
    
    // 等待所有线程完成
    for(int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // 清理资源
    free(threads);
    free(thread_args);
    pthread_mutex_destroy(&mutex);
} 