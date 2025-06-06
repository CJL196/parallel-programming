#!/usr/bin/env python3
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import os
import sys


class CUDAPerformanceAnalyzer:
    def __init__(self, executable_path="./matrixmul"):
        self.executable_path = executable_path
        self.results = []
        self.output_file = "output.txt"
        
    def run_experiment(self, m, n, k, block_size, memory_type, partition_type):
        """运行单个实验"""
        try:
            cmd = [self.executable_path, str(m), str(n), str(k), 
                   str(block_size), memory_type, partition_type]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # 解析输出
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('RESULT,'):
                        parts = line.split(',')
                        return {
                            'm': int(parts[1]),
                            'n': int(parts[2]),
                            'k': int(parts[3]),
                            'block_size': int(parts[4]),
                            'memory_type': parts[5],
                            'partition_type': parts[6],
                            'time_ms': float(parts[7]),
                            'gflops': float(parts[8])
                        }
            else:
                print(f"Error running experiment: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"Timeout for experiment: {cmd}")
        except Exception as e:
            print(f"Exception in experiment: {e}")
            
        return None
    
    def run_all_experiments(self):
        """运行所有实验组合"""
        print("开始CUDA矩阵乘法性能测试...")
        
        # 实验参数
        matrix_sizes = [
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (1536, 1536, 1536),
            (2048, 2048, 2048)
        ]
        
        block_sizes = [8, 16, 32]
        memory_types = ['global', 'shared']
        partition_types = ['2d', '1d_row', '1d_col']
        
        total_experiments = len(matrix_sizes) * len(block_sizes) * len(memory_types) * len(partition_types)
        current_exp = 0
        
        for m, n, k in matrix_sizes:
            for block_size in block_sizes:
                for memory_type in memory_types:
                    for partition_type in partition_types:
                        # 跳过不兼容的组合
                        if memory_type == 'shared' and partition_type != '2d':
                            continue
                            
                        current_exp += 1
                        print(f"实验 {current_exp}: "
                              f"矩阵{m}x{n}x{k}, 块大小{block_size}, "
                              f"内存{memory_type}, 划分{partition_type}")
                        
                        result = self.run_experiment(m, n, k, block_size, memory_type, partition_type)
                        if result:
                            print(result)
                            self.results.append(result)
        
        print(f"完成 {len(self.results)} 个有效实验")
    
    def analyze_results(self):
        """分析实验结果"""
        if not self.results:
            print("没有实验结果可分析")
            return
            
        df = pd.DataFrame(self.results)
        
        # 创建分析报告
        report = []
        report.append("=" * 60)
        report.append("CUDA 矩阵乘法性能分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")
        
        # 1. 线程块大小对性能的影响
        report.append("1. 线程块大小对性能的影响")
        report.append("-" * 40)
        
        block_analysis = df[df['memory_type'] == 'shared'].groupby(['m', 'block_size']).agg({
            'time_ms': 'mean',
            'gflops': 'mean'
        }).reset_index()
        
        for size in sorted(df['m'].unique()):
            size_data = block_analysis[block_analysis['m'] == size]
            if not size_data.empty:
                report.append(f"\n矩阵大小 {size}x{size}:")
                for _, row in size_data.iterrows():
                    report.append(f"  块大小 {row['block_size']}x{row['block_size']}: "
                                f"{row['time_ms']:.2f} ms, {row['gflops']:.2f} GFLOPS")
        
        # 2. 访存方式对性能的影响
        report.append("\n\n2. 访存方式对性能的影响")
        report.append("-" * 40)
        
        memory_analysis = df[(df['block_size'] == 16) & (df['partition_type'] == '2d')].groupby(['m', 'memory_type']).agg({
            'time_ms': 'mean',
            'gflops': 'mean'
        }).reset_index()
        
        for size in sorted(df['m'].unique()):
            size_data = memory_analysis[memory_analysis['m'] == size]
            if not size_data.empty:
                report.append(f"\n矩阵大小 {size}x{size} (块大小16x16):")
                for _, row in size_data.iterrows():
                    report.append(f"  {row['memory_type']}: "
                                f"{row['time_ms']:.2f} ms, {row['gflops']:.2f} GFLOPS")
        
        # 3. 任务划分方式对性能的影响
        report.append("\n\n3. 任务划分方式对性能的影响")
        report.append("-" * 40)
        
        partition_analysis = df[(df['memory_type'] == 'global') & (df['block_size'] == 16)].groupby(['m', 'partition_type']).agg({
            'time_ms': 'mean',
            'gflops': 'mean'
        }).reset_index()
        
        for size in sorted(df['m'].unique()):
            size_data = partition_analysis[partition_analysis['m'] == size]
            if not size_data.empty:
                report.append(f"\n矩阵大小 {size}x{size} (全局内存, 块大小16x16):")
                for _, row in size_data.iterrows():
                    report.append(f"  {row['partition_type']}: "
                                f"{row['time_ms']:.2f} ms, {row['gflops']:.2f} GFLOPS")
        
        # 4. 最优配置推荐
        report.append("\n\n4. 最优配置推荐")
        report.append("-" * 40)
        
        best_configs = df.loc[df.groupby('m')['gflops'].idxmax()]
        for _, row in best_configs.iterrows():
            report.append(f"\n矩阵大小 {row['m']}x{row['m']}:")
            report.append(f"  最优配置: {row['memory_type']}, 块大小{row['block_size']}, {row['partition_type']}")
            report.append(f"  性能: {row['time_ms']:.2f} ms, {row['gflops']:.2f} GFLOPS")
        
        # 保存报告
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"分析报告已保存到 {self.output_file}")
        return df
    
    def create_visualizations(self, df):
        """创建可视化图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CUDA 矩阵乘法性能分析', fontsize=16, fontweight='bold')
        
        # 1. 线程块大小对性能的影响
        shared_data = df[df['memory_type'] == 'shared']
        if not shared_data.empty:
            pivot_data = shared_data.pivot_table(values='gflops', index='m', columns='block_size', aggfunc='mean')
            pivot_data.plot(kind='bar', ax=axes[0,0], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0,0].set_title('线程块大小对于性能的影响\n(共享内存版本)')
            axes[0,0].set_xlabel('矩阵大小')
            axes[0,0].set_ylabel('性能 (GFLOPS)')
            axes[0,0].legend(title='块大小')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. 访存方式对性能的影响
        memory_data = df[(df['block_size'] == 16) & (df['partition_type'] == '2d')]
        if not memory_data.empty:
            pivot_data = memory_data.pivot_table(values='gflops', index='m', columns='memory_type', aggfunc='mean')
            pivot_data.plot(kind='bar', ax=axes[0,1], color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
            axes[0,1].set_title('访存方式对性能的影响\n(块大小16x16, partition_type=2d)')
            axes[0,1].set_xlabel('矩阵大小')
            axes[0,1].set_ylabel('性能 (GFLOPS)')
            axes[0,1].legend(title='访存方式')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. 任务划分方式对性能的影响
        partition_data = df[(df['memory_type'] == 'global') & (df['block_size'] == 16)]
        if not partition_data.empty:
            pivot_data = partition_data.pivot_table(values='gflops', index='m', columns='partition_type', aggfunc='mean')
            pivot_data.plot(kind='bar', ax=axes[0,2], color=['#FF7675', '#74B9FF', '#00B894'])
            axes[0,2].set_title('任务划分方式对性能的影响\n(全局内存, 块大小16x16)')
            axes[0,2].set_xlabel('矩阵大小')
            axes[0,2].set_ylabel('性能 (GFLOPS)')
            axes[0,2].legend(title='划分方式')
            axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. 性能热力图 - 线程块大小 vs 矩阵大小
        if not shared_data.empty:
            heatmap_data = shared_data.pivot_table(values='gflops', index='block_size', columns='m', aggfunc='max')
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1,0])
            axes[1,0].set_title('共享内存版本性能上限热力图')
            axes[1,0].set_xlabel('矩阵大小')
            axes[1,0].set_ylabel('线程块大小')
        
        # 5. 全局内存版本性能热力图分析
        global_data = df[df['memory_type'] == 'global']
        heatmap_data = global_data.pivot_table(values='gflops', index='block_size', columns='m', aggfunc='max')
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1,1])
        axes[1,1].set_title('全局内存版本性能上限热力图')
        axes[1,1].set_xlabel('矩阵大小')
        axes[1,1].set_ylabel('线程块大小')
        
        # 6. 最佳性能对比
        best_perf = df.loc[df.groupby('m')['gflops'].idxmax()]
        axes[1,2].plot(best_perf['m'], best_perf['gflops'], 'o-', linewidth=2, markersize=8, color='#00B894')
        axes[1,2].set_title('各矩阵大小的最佳性能')
        axes[1,2].set_xlabel('矩阵大小')
        axes[1,2].set_ylabel('最佳性能 (GFLOPS)')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cuda_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("可视化图表已保存为 cuda_performance_analysis.png")

def main():
    # 检查可执行文件是否存在
    executable = "./build/matrixmul"
    if not os.path.exists(executable):
        print(f"错误: 找不到可执行文件 {executable}")
        print("请确保已经编译了 matrixmul.cu:")
        print("nvcc -O3 -arch=sm_75 -o matrixmul matrixmul.cu -lcublas")
        return
    
    analyzer = CUDAPerformanceAnalyzer(executable)
    
    try:
        # 运行所有实验
        analyzer.run_all_experiments()
        
        # 分析结果
        df = analyzer.analyze_results()
        
        if df is not None and not df.empty:
            # 创建可视化
            analyzer.create_visualizations(df)
            
            print("\n分析完成!")
            print(f"- 详细报告: {analyzer.output_file}")
            print("- 可视化图表: cuda_performance_analysis.png")
        else:
            print("没有有效的实验结果进行分析")
            
    except KeyboardInterrupt:
        print("\n实验被用户中断")
    except Exception as e:
        print(f"分析过程中发生错误: {e}")

if __name__ == "__main__":
    main()