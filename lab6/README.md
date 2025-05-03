# 热平板并行模拟实验说明

## 目录结构
- `src/`：Pthreads 并行实现及 parallel_for 相关代码
- `reference/`：OpenMP 参考实现
- `reports/`：性能测试结果与对比图片
- `performance_analyzer.py`：自动化性能测试与报告生成脚本
- `report.md`：实验报告
- `requirement.md`：实验要求

## 编译方法

### 1. 编译 Pthreads 版本
```bash
make
# 或手动编译：
gcc -o src/heated_plate_pthread src/heated_plate_pthread.c src/parallel_for.c -lpthread -lm
```

### 2. 编译 OpenMP 参考版本
```bash
make openmp
# 或手动编译：
gcc -o reference/heated_plate_openmp reference/heated_plate_openmp.c -fopenmp -lm
```

## 运行方法

### 1. 运行 Pthreads 版本
```bash
./src/heated_plate_pthread [线程数] [调度方式] [块大小]
```
- 线程数：正整数，如 4、8
- 调度方式：s（静态 static）、d（动态 dynamic）、g（引导式 guided），如 s、d、g
- 块大小：正整数，决定每次分配给线程的任务块大小

**示例：**
```bash
./src/heated_plate_pthread 4 d 10   # 4线程，动态调度，块大小10
./src/heated_plate_pthread 8 g 50   # 8线程，引导式调度，块大小50
```

### 2. 运行 OpenMP 版本
```bash
export OMP_NUM_THREADS=4
./reference/heated_plate_openmp
```

## 性能测试与自动化分析

可使用 `performance_analyzer.py` 脚本自动测试不同线程数、调度方式和块大小下的性能，并生成对比图和报告。

```bash
python3 performance_analyzer.py
```
- 运行后将在 `reports/` 目录下生成：
  - `performance_report.txt`：详细性能数据与分析
  - `thread_comparison.png`：不同线程数性能对比图
  - `schedule_comparison.png`：不同调度策略性能对比图
  - `chunk_comparison.png`：不同块大小性能对比图

## 结果查看
- 详细实验结果和分析见 `reports/performance_report.txt` 和 `report.md`
- 主要性能对比图见 `reports/` 目录

## 其他
- 清理编译产物：
```bash
make clean
```
- 如需修改参数或调度策略，可直接编辑命令行参数或修改源代码后重新编译。
