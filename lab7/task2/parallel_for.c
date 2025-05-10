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
                int chunk = remaining / args->num_threads;
                if(chunk < args->chunk_size) {
                    chunk = args->chunk_size;
                }
                
                *args->current_index = current + chunk;
                pthread_mutex_unlock(args->mutex);
                
                int end = (current + chunk > args->end) ? args->end : current + chunk;
                
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
        thread_args[i].needs_sync = 0;
        thread_args[i].shared_double = NULL;
        thread_args[i].shared_mutex = NULL;
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

// 带有共享变量的线程函数
void* shared_thread_function(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int i;
    
    // 对每个线程创建一个局部变量存储最大差异
    double local_max = 0.0;
    
    switch(args->schedule_type) {
        case STATIC: {
            // 静态调度：每个线程处理固定大小的块
            int chunk = (args->end - args->start) / args->num_threads;
            int start = args->start + args->thread_id * chunk;
            int end = (args->thread_id == args->num_threads - 1) ? 
                     args->end : start + chunk;
            
            for(i = start; i < end; i += args->inc) {
                // 调用用户函数
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
                    // 调用用户函数
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
                int chunk = remaining / args->num_threads;
                if(chunk < args->chunk_size) {
                    chunk = args->chunk_size;
                }
                
                *args->current_index = current + chunk;
                pthread_mutex_unlock(args->mutex);
                
                int end = (current + chunk > args->end) ? args->end : current + chunk;
                
                for(i = current; i < end; i += args->inc) {
                    // 调用用户函数
                    args->functor(i, args->arg);
                }
            }
            break;
        }
    }
    
    return NULL;
}

// 带有共享变量更新的parallel_for函数实现
void parallel_for_shared(int start, int end, int inc,
                       void *(*functor)(int, void*), void *arg, 
                       int num_threads, ScheduleType schedule_type, int chunk_size,
                       double *shared_value) {
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadArgs* thread_args = (ThreadArgs*)malloc(num_threads * sizeof(ThreadArgs));
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t shared_mutex = PTHREAD_MUTEX_INITIALIZER;
    int current_index = start;
    
    // 重置共享变量
    *shared_value = 0.0;
    
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
        thread_args[i].needs_sync = 1;
        thread_args[i].shared_double = shared_value;
        thread_args[i].shared_mutex = &shared_mutex;
    }
    
    // 创建线程
    for(int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, shared_thread_function, &thread_args[i]);
    }
    
    // 等待所有线程完成
    for(int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // 清理资源
    free(threads);
    free(thread_args);
    pthread_mutex_destroy(&mutex);
    pthread_mutex_destroy(&shared_mutex);
} 