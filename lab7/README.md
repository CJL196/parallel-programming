# 并行程序设计与分析实验

本实验包含两个主要任务：
1. 使用MPI对快速傅里叶变换(FFT)进行并行化
2. 对热平板问题的parallel_for并行应用进行性能和内存消耗分析

## 项目结构

```
lab7/
├── task1/                  # MPI并行FFT实现
│   ├── main.cpp            # MPI并行FFT源代码
│   ├── Makefile            # 编译脚本
│   └── out.txt             # 运行结果输出
│
├── task2/                  # 热平板问题并行性能分析
│   ├── heated_plate_pthread.c    # 热平板问题的pthread实现
│   ├── parallel_for.c      # parallel_for框架实现
│   ├── parallel_for.h      # parallel_for框架头文件
│   ├── analyze_performance.py    # 性能分析脚本
│   ├── analyze_memory.py   # 内存分析脚本
│   ├── run_analysis.sh     # 分析脚本运行器
│   ├── Makefile            # 编译脚本
│   └── reports/            # 分析结果报告目录
│       ├── performance_report.txt    # 性能分析报告
│       ├── memory_analysis_report.txt # 内存分析报告
│       ├── problem_size_comparison.png # 问题规模-执行时间关系图
│       ├── speedup_comparison.png     # 加速比图
│       ├── efficiency_comparison.png  # 并行效率图
│       ├── peak_memory_usage.png      # 峰值内存使用图
│       └── memory_usage_over_time_N*.png # 内存使用随时间变化图
│
├── reference/             # 参考资料
├── report.md              # 实验报告
└── requirement.md         # 实验要求
```

## 运行方式

### 任务1：MPI并行FFT

1. 编译程序：
```bash
cd task1
make
```

2. 运行程序：
```bash
# 使用4个进程运行
mpirun -np 4 ./fft_mpi
```


### 任务2：热平板问题并行性能分析

1. 编译程序：
```bash
cd task2
make
```

2. 运行性能分析：
```bash
python analyze_performance.py
```

3. 运行内存分析（需要安装Valgrind）：
```bash
# 运行内存分析
python analyze_memory.py
```

4. 一键运行所有分析：
```bash
./run_analysis.sh
```

5. 查看分析结果：
```bash
# 查看性能分析报告
cat reports/performance_report.txt

# 查看内存分析报告
cat reports/memory_analysis_report.txt

# 查看图表（需要图形界面）
xdg-open reports/problem_size_comparison.png
xdg-open reports/speedup_comparison.png
xdg-open reports/efficiency_comparison.png
xdg-open reports/peak_memory_usage.png
xdg-open reports/memory_usage_over_time_N256.png
```

## 实验参数说明

### 热平板程序参数

```bash
./heated_plate_pthread <线程数> <调度类型> <块大小> <问题规模>
```

- 线程数：1, 2, 4, 8等
- 调度类型：s (静态), d (动态), g (引导式)
- 块大小：任务分块大小，如10
- 问题规模：网格大小N，如32, 64, 128, 256等

示例：
```bash
# 使用4线程，引导式调度，块大小10，问题规模256
./heated_plate_pthread 4 g 10 256
```

## 实验报告

详细的实验过程、结果分析和结论请参考 [实验报告](report.md)。
