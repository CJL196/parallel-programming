## Lab11: GPU 卷积加速

本实验包含三个任务，分别使用直接法、im2col+GEMM 以及 cuDNN 库在 GPU 上实现卷积。

### 依赖

- CMake (>= 3.9)
- CUDA Toolkit (>= 11.0)
- NVIDIA cuDNN Library
- C++ 编译器 (如 g++)

### 编译指南

所有任务均可通过项目根目录下的 `CMakeLists.txt` 进行统一编译。

1.  **创建并进入 build 目录**
    ```bash
    mkdir build
    cd build
    ```

2.  **运行 CMake 配置项目**
    ```bash
    cmake ..
    ```
    此命令会查找 CUDA, cuDNN 等依赖，并生成 Makefiles。如果 cuDNN 未安装或未在标准路径下，此步可能会失败。

3.  **编译所有任务**
    ```bash
    make
    ```
    编译成功后，可执行文件将分别位于 `task1/`, `task2/`, `task3/` 目录下。

### 运行代码

#### 任务一：直接卷积

```bash
# 用法: ./task1/conv <input_size> <stride>
./task1/conv 512 1
```

#### 任务二：im2col + GEMM 卷积

```bash
# 用法: ./task2/im2col_gemm_conv <input_size> <kernel_size> <stride>
# kernel_size 在本实验中固定为 3
./task2/im2col_gemm_conv 512 3 1
```

#### 任务三：cuDNN 卷积

```bash
# 用法: ./task3/cudnn_conv <input_size> <stride>
./task3/cudnn_conv 512 1
```

### 自动化测试与绘图

为了方便地进行批量测试、数据收集和可视化，项目内提供了一个 Python 脚本 `run_and_plot.py`。

#### 依赖

- Python 3.x
- Pandas
- Matplotlib

如果尚未安装这些库，请运行：
```bash
pip install pandas matplotlib
```

#### 使用方法

在 `lab11/` 目录下运行脚本：
```bash
python3 run_and_plot.py
```

该脚本会自动完成以下操作：
1.  重新编译所有 C++/CUDA 代码。
2.  循环执行预设的测试用例。
3.  将所有性能数据输出到 `performance_results.csv` 文件中。
4.  根据 stride 生成性能对比图表（如 `performance_stride_1.png`）。