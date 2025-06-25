import os
import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 配置 ---
# 构建目录和可执行文件
BUILD_DIR = 'build'
EXECUTABLES = {
    'Task 1: Direct Conv': os.path.join(BUILD_DIR, 'task1', 'conv'),
    'Task 2: im2col+GEMM': os.path.join(BUILD_DIR, 'task2', 'im2col_gemm_conv'),
    'Task 3: cuDNN': os.path.join(BUILD_DIR, 'task3', 'cudnn_conv')
}

# 定义要运行的测试用例
# 格式：(input_size, kernel_size, stride)
# kernel_size 仅用于 Task 2
TEST_CASES = [
    (256, 3, 1),
    (512, 3, 1),
    (1024, 3, 1),
    (2048, 3, 1),
    (512, 3, 2),
    (1024, 3, 2),
    (2048, 3, 2),
    (512, 3, 3),
    (1024, 3, 3),
    (2048, 3, 3),
]

# 用于从程序输出中提取时间的正则表达式
TIME_REGEX = re.compile(r"(?:Execution|GPU Computation) time: ([\d.]+) ms")

def compile_code():
    """编译所有任务"""
    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)
    
    print("--- Compiling codes ---")
    try:
        # 进入 build 目录并运行 cmake 和 make
        cmake_process = subprocess.run(['cmake', '..'], cwd=BUILD_DIR, check=True, capture_output=True, text=True)
        print(cmake_process.stdout)
        make_process = subprocess.run(['make', '-j'], cwd=BUILD_DIR, check=True, capture_output=True, text=True)
        print(make_process.stdout)
        print("--- Compilation successful ---\n")
        return True
    except subprocess.CalledProcessError as e:
        print("--- Compilation failed! ---")
        print(e.stderr)
        return False

def run_single_test(task_name, executable_path, size, kernel, stride):
    """运行单个测试并返回执行时间"""
    command = [executable_path]
    if 'Task 2' in task_name:
        command.extend([str(size), str(kernel), str(stride)])
    else:
        command.extend([str(size), str(stride)])

    try:
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        output = result.stdout
        
        match = TIME_REGEX.search(output)
        if match:
            time = float(match.group(1))
            print(f"Success! Time: {time:.4f} ms")
            return time
        else:
            print("Error: Could not parse execution time from output.")
            print(output)
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        print(e.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(command)}")
        return None

def plot_results(df):
    """使用matplotlib根据DataFrame绘制结果图表"""
    strides = df['stride'].unique()
    
    for stride in strides:
        stride_df = df[df['stride'] == stride].set_index(['size', 'task_name'])['time'].unstack()
        
        # 绘图
        ax = stride_df.plot(kind='bar', figsize=(14, 8), logy=True)
        
        plt.title(f'Convolution Performance Comparison (Stride = {stride})', fontsize=16)
        plt.xlabel('Input Size (N x N)', fontsize=12)
        plt.ylabel('Execution Time (ms, log scale)', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Implementation')
        
        # 在条形图上添加数值标签
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8, padding=3)

        plt.tight_layout()
        
        # 保存图表
        output_filename = f'performance_stride_{stride}.png'
        plt.savefig(output_filename)
        print(f"\nChart saved to {output_filename}")
        plt.close()


def main():
    if not compile_code():
        return

    results = []
    for size, kernel, stride in TEST_CASES:
        for task_name, executable in EXECUTABLES.items():
            # 检查可执行文件是否存在
            if not os.path.exists(executable):
                print(f"Executable not found: {executable}, skipping.")
                continue

            # 运行测试
            time_ms = run_single_test(task_name, executable, size, kernel, stride)
            if time_ms is not None:
                results.append({
                    'task_name': task_name,
                    'size': size,
                    'stride': stride,
                    'time': time_ms
                })
        print("-" * 20)

    if not results:
        print("No results were collected. Exiting.")
        return

    # 将结果转换为Pandas DataFrame并保存
    df = pd.DataFrame(results)
    print("\n--- Performance Results ---")
    print(df)
    
    csv_filename = 'performance_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")
    
    # 绘图
    plot_results(df)

if __name__ == '__main__':
    main() 