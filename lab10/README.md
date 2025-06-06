# CUDA矩阵乘法性能分析

本项目实现了使用CUDA进行矩阵乘法计算，并分析不同实现方式对性能的影响。

## 编译方法

1. 确保已安装CUDA工具链和CMake
2. 执行以下命令进行编译：

```bash
mkdir -p build
cd build
cmake ..
make
```

## 运行方法

### 1. 运行单次测试

```bash
./build/matrixmul <m> <n> <k> <block_size> <memory_type> <partition_type>
```

参数说明：
- `m`, `n`, `k`: 矩阵维度，A(m×n)与B(n×k)相乘得到C(m×k)
- `block_size`: 线程块大小，可选值：8, 16, 32
- `memory_type`: 内存访问方式，可选值：global(全局内存), shared(共享内存)
- `partition_type`: 任务划分方式，可选值：2d(二维划分), 1d_row(按行划分), 1d_col(按列划分)

示例：
```bash
./build/matrixmul 1024 1024 1024 16 shared 2d
```

### 2. 运行性能分析

执行以下命令将自动运行多组测试并生成分析报告：

```bash
python3 performance_analysis.py
```

运行完成后会生成：
- `output.txt`: 详细的性能分析报告
- `cuda_performance_analysis.png`: 性能分析可视化图表

## 文件说明

- `matrixmul.cu`: CUDA矩阵乘法核心实现
- `performance_analysis.py`: 性能测试与分析脚本
- `CMakeLists.txt`: CMake构建配置文件