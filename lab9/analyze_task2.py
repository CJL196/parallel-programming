import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product

def run_experiment(matrix_size, block_size, kernel_type):
    """运行单次实验并返回结果"""
    cmd = f"./build/task2/transpose {matrix_size} {block_size} {kernel_type}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout.strip()
    
    if output.startswith("SUCCESS"):
        status, n, block_size, kernel_type, time = output.split(',')
        return {
            'matrix_size': int(n),
            'block_size': int(block_size),
            'kernel_type': int(kernel_type),
            'time': float(time)
        }
    return None

def run_all_experiments():
    """运行所有实验组合"""
    matrix_sizes = [512, 768, 1024, 1536, 2048]
    block_sizes = [16, 32]
    kernel_types = [0, 1]  # 0: naive, 1: shared memory
    
    results = []
    
    for matrix_size, block_size, kernel_type in product(matrix_sizes, block_sizes, kernel_types):
        print(f"Running experiment: matrix_size={matrix_size}, block_size={block_size}, kernel_type={kernel_type}")
        result = run_experiment(matrix_size, block_size, kernel_type)
        if result:
            results.append(result)
    
    return pd.DataFrame(results)

def plot_results(df):
    """绘制性能分析图表"""
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 不同矩阵大小和线程块大小的性能对比
    for kernel_type in [0, 1]:
        kernel_name = "Naive" if kernel_type == 0 else "Shared Memory"
        for block_size in [16, 32]:
            data = df[(df['kernel_type'] == kernel_type) & (df['block_size'] == block_size)]
            ax1.plot(data['matrix_size'], data['time'], 
                    marker='o', 
                    label=f'{kernel_name} (Block Size={block_size})')
    
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Performance vs Matrix Size')
    ax1.grid(True)
    ax1.legend()
    
    # 2. 不同线程块大小的性能对比
    for kernel_type in [0, 1]:
        kernel_name = "Naive" if kernel_type == 0 else "Shared Memory"
        data = df[df['kernel_type'] == kernel_type].groupby('block_size')['time'].mean()
        ax2.bar(f'{kernel_name}\nBlock Size={data.index[0]}', data.iloc[0], 
                label=f'Block Size={data.index[0]}')
        ax2.bar(f'{kernel_name}\nBlock Size={data.index[1]}', data.iloc[1], 
                label=f'Block Size={data.index[1]}')
    
    ax2.set_ylabel('Average Execution Time (ms)')
    ax2.set_title('Performance vs Block Size')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/performance_analysis.png')
    plt.close()

def main():
    # 运行所有实验
    print("开始运行性能测试...")
    results_df = run_all_experiments()
    
    # 保存原始数据
    results_df.to_csv('results/performance_results.csv', index=False)
    
    # 绘制性能分析图表
    print("生成性能分析图表...")
    plot_results(results_df)
    
    # 打印统计信息
    print("\n性能测试结果统计：")
    print("\n按核函数类型和线程块大小的平均执行时间：")
    stats = results_df.groupby(['kernel_type', 'block_size'])['time'].agg(['mean', 'std'])
    print(stats)
    
    print("\n按矩阵大小的平均执行时间：")
    matrix_stats = results_df.groupby(['matrix_size', 'kernel_type'])['time'].mean()
    print(matrix_stats)

if __name__ == "__main__":
    main()
