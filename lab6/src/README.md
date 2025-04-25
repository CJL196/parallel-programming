# 热平板问题 - Pthread并行实现

本项目将OpenMP版本的热平板问题实现改造为基于Pthreads的并行应用，并提供性能测试和分析工具。

## 文件结构

- `heated_plate_pthread.c`: 使用Pthreads实现的热平板问题
- `parallel_for.h` 和 `parallel_for.c`: 提供类似OpenMP的并行for循环功能
- `heated_plate_test.c`: 测试不同线程数和调度策略下的性能
- `performance_analyzer.py`: 性能分析脚本，生成报告和图表
- `Makefile`: 用于编译和运行测试

## 编译和运行

### 编译

```bash
# 编译所有程序
make all

# 编译OpenMP参考程序
make openmp
```

### 运行测试

```bash
# 运行简单测试
make test

# 运行性能分析
make analyze
```

### 直接运行

```bash
# 运行Pthread版本，参数：线程数 调度类型 块大小
./heated_plate_pthread 4 s 10

# 调度类型: s(静态), d(动态), g(引导式)
```

## 性能测试与分析

性能分析脚本会测试不同的配置并生成报告：

1. 不同线程数 (1, 2, 4, 8, 16)
2. 不同调度策略 (静态, 动态, 引导式)
3. 不同块大小 (10, 50, 100)

分析结果将保存在`reports`目录下:
- `thread_comparison.png`: 线程数比较
- `schedule_comparison.png`: 调度策略比较
- `chunk_comparison.png`: 块大小比较
- `performance_report.txt`: 详细性能报告

## 并行策略

该实现提供了三种并行调度策略：

1. **静态调度 (STATIC)**: 平均分配循环迭代给各线程
2. **动态调度 (DYNAMIC)**: 线程动态获取块大小固定的任务
3. **引导式调度 (GUIDED)**: 块大小随剩余工作量而变化

## 项目依赖

- GCC 编译器
- Pthread 库
- Python 3 (用于性能分析)
- Matplotlib 和 NumPy (用于生成图表)

## 注意事项

- 测试运行需要一定时间，请耐心等待
- 对于大型问题，建议使用更多线程
- 不同系统上的最佳配置可能不同 