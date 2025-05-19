# CUDA并行编程实验

本实验包含两个CUDA编程任务：
1. CUDA Hello World：理解CUDA线程模型和并行执行特性
2. CUDA矩阵转置：掌握CUDA程序性能优化技术

## 环境要求

- CUDA Toolkit
- CMake (>= 3.10)
- Python 3.x (用于性能分析)
- matplotlib
- pandas
- numpy

## 编译

```bash
# 创建并进入构建目录
mkdir -p build
cd build

# 配置和编译
cmake ..
make

# 返回项目根目录
cd ..
```

## 运行

### Task 1: CUDA Hello World

```bash
cd build/task1
./hello
```

程序会提示输入三个整数n, m, k（范围[1, 32]），分别表示：
- n：线程块数量
- m：每个线程块的x维度
- k：每个线程块的y维度

### Task 2: CUDA矩阵转置

#### 直接运行

```bash
cd build/task2
./transpose <matrix_size> <block_size> <kernel_type>
```

参数说明：
- matrix_size：矩阵大小（范围[512, 2048]）
- block_size：线程块大小（建议使用16或32）
- kernel_type：核函数类型（0：基础版本，1：共享内存版本）

#### 性能分析

**重要：** 在运行性能分析脚本之前，必须先完成编译步骤！

```bash
# 确保已创建results目录
mkdir -p results

# 运行性能分析脚本
python analyze_task2.py
```

性能分析脚本会：
1. 自动测试不同矩阵大小（512, 1024, 2048）
2. 测试不同线程块大小（16×16, 32×32）
3. 测试两种核函数实现
4. 生成性能分析图表（保存在results目录）
5. 输出详细的性能统计信息

## 文件说明

- `task1/hello.cu`：CUDA Hello World程序
- `task2/transpose.cu`：矩阵转置程序
- `analyze_task2.py`：性能分析脚本
- `report.md`：实验报告
- `results/`：性能分析结果和图表