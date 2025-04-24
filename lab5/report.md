## 1. 实验目的

本实验包括两个部分：

### 1.1 OpenMP 通用矩阵乘法
- 使用 OpenMP 实现并行通用矩阵乘法
- 分析不同线程数量（1-16）对性能的影响
- 研究不同矩阵规模（128-2048）下的计算效率
- 比较不同调度方式（默认、静态、动态）的性能差异

### 1.2 基于 Pthreads 的并行 for 循环
- 实现基于 Pthreads 的 parallel_for 函数
- 支持多种调度策略（静态、动态、引导式）
- 分析不同线程数（1-16）对性能的影响
- 研究不同块大小（1, 16, 64, 256）对性能的影响
- 比较不同调度策略的性能差异

## 2. 实验过程和核心代码

### 2.1 实验环境
- 操作系统：Linux
- 编译器：gcc/g++
- 编译选项：-Wall -O3 -fPIC -pthread

### 2.2 OpenMP 矩阵乘法实现
```cpp
void matrixMultiply(const vector<vector<double>>& A, 
                    const vector<vector<double>>& B,
                    vector<vector<double>>& C,
                    int m, int n, int k,
                    const string& schedule = "default",
                    int chunk_size = 1) {
    if (schedule == "static") {
        #pragma omp parallel for collapse(2) schedule(static, chunk_size)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                double sum = 0.0;
                for (int l = 0; l < n; l++) {
                    sum += A[i][l] * B[l][j];
                }
                C[i][j] = sum;
            }
        }
    }
    // ... 其他调度方式的实现 ...
}
```

### 2.3 基于 Pthreads 的 parallel_for 实现
```c
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
```

### 2.4 调度策略实现

实现了静态调度、动态调度、引导式调度三种策略

```c
// 静态调度
case STATIC: {
    int chunk = (args->end - args->start) / args->num_threads;
    int start = args->start + args->thread_id * chunk;
    int end = (args->thread_id == args->num_threads - 1) ? 
             args->end : start + chunk;
    
    for(i = start; i < end; i += args->inc) {
        args->functor(i, args->arg);
    }
    break;
}

// 动态调度
case DYNAMIC: {
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

// 引导式调度
case GUIDED: {
    while(1) {
        pthread_mutex_lock(args->mutex);
        int current = *args->current_index;
        if(current >= args->end) {
            pthread_mutex_unlock(args->mutex);
            break;
        }
        
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
```

## 3. 实验结果

### 3.1 OpenMP 矩阵乘法性能分析

```bash
cd 1
make run
```

运行结果如下：


```
矩阵大小        线程数            调度方式        执行时间(ms)
128             1               default         1
128             1               static          1
128             1               dynamic         2
128             2               default         0
128             2               static          0
128             2               dynamic         1
128             4               default         0
128             4               static          0
128             4               dynamic         0
128             8               default         0
128             8               static          0
128             8               dynamic         0
128             16              default         0
128             16              static          0
128             16              dynamic         0
256             1               default         12
256             1               static          11
256             1               dynamic         12
256             2               default         5
256             2               static          6
256             2               dynamic         8
256             4               default         3
256             4               static          3
256             4               dynamic         5
256             8               default         2
256             8               static          1
256             8               dynamic         3
256             16              default         1
256             16              static          1
256             16              dynamic         3
512             1               default         107
512             1               static          106
512             1               dynamic         124
512             2               default         59
512             2               static          58
512             2               dynamic         72
512             4               default         29
512             4               static          36
512             4               dynamic         38
512             8               default         16
512             8               static          19
512             8               dynamic         22
512             16              default         15
512             16              static          15
512             16              dynamic         14
1024            1               default         1140
1024            1               static          1140
1024            1               dynamic         1183
1024            2               default         570
1024            2               static          597
1024            2               dynamic         634
1024            4               default         290
1024            4               static          359
1024            4               dynamic         367
1024            8               default         169
1024            8               static          277
1024            8               dynamic         269
1024            16              default         142
1024            16              static          237
1024            16              dynamic         188
2048            1               default         13648
2048            1               static          13600
2048            1               dynamic         13999
2048            2               default         7232
2048            2               static          9629
2048            2               dynamic         10015
2048            4               default         3483
2048            4               static          5293
2048            4               dynamic         5352
2048            8               default         2130
2048            8               static          3390
2048            8               dynamic         3356
2048            16              default         1469
2048            16              static          2819
2048            16              dynamic         2395
```

1. 线程数影响：
   - 128x128矩阵：线程数增加对性能影响较小，所有配置下执行时间都在0-2ms之间
   - 256x256矩阵：从单线程到16线程，执行时间从12ms降至1ms，加速比约12倍
   - 512x512矩阵：从单线程到16线程，执行时间从107ms降至14ms，加速比约7.6倍
   - 1024x1024矩阵：从单线程到16线程，执行时间从1140ms降至142ms，加速比约8倍
   - 2048x2048矩阵：从单线程到16线程，执行时间从13648ms降至1469ms，加速比约9.3倍

2. 矩阵规模影响：
   - 小规模矩阵（128x128）：并行化收益不明显，线程创建开销可能超过计算收益
   - 中等规模矩阵（256x256-512x512）：并行化效果显著，加速比随线程数增加而提高
   - 大规模矩阵（1024x1024-2048x2048）：并行化效果最佳，但线程数增加到一定程度后收益递减

3. 调度方式比较：
   - 默认调度：在大多数情况下表现最好，特别是在大规模矩阵和高线程数时
   - 静态调度：在小规模矩阵和低线程数时表现较好，但在高线程数时性能下降明显
   - 动态调度：整体性能略逊于默认调度，但在某些特定场景下（如2048x2048矩阵，16线程）表现较好

### 3.2 基于 Pthreads 的 parallel_for 性能分析

运行`2/run.sh`结果如下：

```
Matrix size: 1024x1024
Testing different scheduling strategies...

Schedule: Static
Threads Chunk Size      Time (ms)       Speedup
------------------------------------------------
1       1               3452.47         0.98
1       16              3398.61         1.00
1       64              3405.44         1.00
1       256             3362.42         1.01
2       1               1621.83         2.09
2       16              1631.99         2.08
2       64              1640.35         2.07
2       256             1722.87         1.97
4       1               824.29          4.12
4       16              876.47          3.87
4       64              841.69          4.03
4       256             853.92          3.97
8       1               445.48          7.61
8       16              448.76          7.56
8       64              416.89          8.14
8       256             406.75          8.34
16      1               237.49          14.28
16      16              224.30          15.12
16      64              247.82          13.69
16      256             245.67          13.81

Schedule: Dynamic
Threads Chunk Size      Time (ms)       Speedup
------------------------------------------------
1       1               3347.35         1.00
1       16              3338.59         1.00
1       64              3363.41         1.00
1       256             3347.18         1.00
2       1               1660.86         2.02
2       16              1615.80         2.08
2       64              1610.97         2.08
2       256             1607.56         2.09
4       1               820.78          4.09
4       16              854.23          3.93
4       64              857.96          3.91
4       256             854.49          3.92
8       1               371.52          9.03
8       16              386.32          8.68
8       64              442.93          7.57
8       256             838.53          4.00
16      1               194.27          17.26
16      16              221.97          15.11
16      64              210.53          15.93
16      256             816.54          4.11

Schedule: Guided
Threads Chunk Size      Time (ms)       Speedup
------------------------------------------------
1       1               3367.20         1.03
1       16              3313.75         1.04
1       64              3342.34         1.03
1       256             3339.01         1.04
2       1               1608.46         2.15
2       16              1616.30         2.14
2       64              1674.36         2.06
2       256             1592.26         2.17
4       1               833.68          4.15
4       16              830.49          4.16
4       64              835.49          4.14
4       256             833.10          4.15
8       1               392.18          8.82
8       16              401.51          8.61
8       64              359.77          9.61
8       256             379.25          9.12
16      1               193.50          17.87
16      16              196.74          17.57
16      64              206.96          16.70
16      256             209.27          16.52
```

根据测试结果（1024x1024矩阵乘法），我们可以得出以下结论：

1. 线程数影响：
   - 单线程到16线程，执行时间从约3400ms降至约200ms
   - 最佳加速比达到17.87倍（引导式调度，16线程，块大小1）
   - 线程数增加带来的性能提升呈现边际递减趋势

2. 调度策略比较：
   - 静态调度（Static）：
     - 优点：实现简单，开销小
     - 缺点：负载不均衡时性能下降
     - 最佳配置：16线程，块大小16，加速比15.12
   
   - 动态调度（Dynamic）：
     - 优点：负载均衡性好
     - 缺点：同步开销大
     - 最佳配置：16线程，块大小1，加速比17.26
   
   - 引导式调度（Guided）：
     - 优点：结合了静态和动态调度的优点
     - 缺点：实现复杂
     - 最佳配置：16线程，块大小1，加速比17.87

3. 块大小影响：
   - 小块大小（1）：适合动态和引导式调度，减少负载不均衡
   - 中等块大小（16-64）：适合静态调度，减少线程同步开销
   - 大块大小（256）：性能普遍下降，因为可能导致负载不均衡

## 4. 实验感想

通过这次实验，我对并行计算有了更深的理解。起初，我发现小规模计算并不适合并行化，因为创建和调度线程的开销可能会抵消掉并行带来的好处。但随着问题规模的增大，并行化的优势逐渐显现出来，尽管这种优势并不是无止境的，收益会逐渐减弱，存在一种边际递减的现象。

我也体会到调度策略在不同场景下的重要性。实验中，OpenMP的默认调度在大多数情况下表现得很稳定，说明它的设计经过了充分优化。但在某些情况下，比如高线程数时，静态调度的性能会下降，可能是因为负载分配不够均衡。不同的调度方式各有千秋，选择哪种策略需要看具体情况。

性能优化这件事远比想象中复杂。想要找到最优配置，必须综合考虑矩阵规模、线程数和调度方式。没有一种配置是通用的，优化需要根据实际场景不断调整。更重要的是，优化不能只靠理论推测，必须要有实际测试数据的支撑。

另外，我还对比了OpenMP和Pthreads。OpenMP用起来更简单，抽象层次更高，适合快速开发。而Pthreads虽然复杂一些，但提供了更细致的控制，能实现更灵活的调度策略。两者各有优劣，选择哪种得看具体需求。

总的来说，这次实验让我更深刻地认识到并行计算的实际应用中需要权衡各种因素，才能尽可能提升性能。同时，我也意识到性能优化是个复杂的过程，需要大量实验和数据分析来支持。

