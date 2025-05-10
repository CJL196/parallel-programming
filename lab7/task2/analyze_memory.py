#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import re
import time
from collections import defaultdict

def run_command(cmd):
    """运行命令并返回输出结果"""
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode('utf-8'), stderr.decode('utf-8')

def parse_massif_output(output_file):
    """解析massif输出文件，提取内存消耗数据"""
    # 直接读取massif文件而不是使用ms_print
    try:
        with open(output_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"  读取massif文件失败: {e}")
        return 0, [], []
    
    # 查找峰值内存使用
    peak_memory = 0
    peak_snapshot = None
    
    # 查找标记为peak的快照
    peak_match = re.search(r'snapshot=(\d+).*?heap_tree=peak', content, re.DOTALL)
    if peak_match:
        peak_snapshot = peak_match.group(1)
        # 在peak快照中查找内存使用信息
        mem_heap_match = re.search(r'snapshot=' + peak_snapshot + r'.*?mem_heap_B=(\d+).*?mem_heap_extra_B=(\d+).*?mem_stacks_B=(\d+)', content, re.DOTALL)
        if mem_heap_match:
            heap_b = int(mem_heap_match.group(1))
            heap_extra_b = int(mem_heap_match.group(2))
            stacks_b = int(mem_heap_match.group(3))
            # 总内存 = 堆内存 + 额外堆内存 + 栈内存，转换为KB
            peak_memory = (heap_b + heap_extra_b + stacks_b) / 1024.0
    
    if peak_memory == 0:
        # 如果没有找到peak标记，则查找所有快照中的最大内存使用
        snapshots = re.findall(r'snapshot=(\d+).*?mem_heap_B=(\d+).*?mem_heap_extra_B=(\d+).*?mem_stacks_B=(\d+)', content, re.DOTALL)
        for snapshot_id, heap_b, heap_extra_b, stacks_b in snapshots:
            total_memory = (int(heap_b) + int(heap_extra_b) + int(stacks_b)) / 1024.0
            if total_memory > peak_memory:
                peak_memory = total_memory
    
    # 提取时间和内存使用数据点
    time_points = []
    memory_points = []
    
    snapshots = re.findall(r'snapshot=(\d+).*?time=(\d+).*?mem_heap_B=(\d+).*?mem_heap_extra_B=(\d+).*?mem_stacks_B=(\d+)', content, re.DOTALL)
    for snapshot_id, time_val, heap_b, heap_extra_b, stacks_b in snapshots:
        time_points.append(float(time_val) / 1000000.0)  # 转换为更合理的单位
        total_memory = (int(heap_b) + int(heap_extra_b) + int(stacks_b)) / 1024.0
        memory_points.append(total_memory)
    
    print(f"  解析到的峰值内存使用: {peak_memory:.2f} KB")
    return peak_memory, time_points, memory_points

def run_memory_analysis():
    """运行内存分析并收集结果"""
    thread_counts = [1, 2, 4, 8]
    problem_sizes = [32, 64, 128, 256]  # 选择部分问题规模进行测试，避免测试时间过长
    
    # 创建目录存放massif输出文件
    if not os.path.exists('./massif_out'):
        os.makedirs('./massif_out')
    
    # 确保编译程序
    os.system("gcc -g -O0 -o ./heated_plate_pthread ./heated_plate_pthread.c ./parallel_for.c -lpthread -lm")
    
    results = defaultdict(dict)
    
    for size in problem_sizes:
        print(f"分析问题规模 N = {size} 的内存使用情况")
        for threads in thread_counts:
            print(f"  线程数: {threads}")
            
            # 使用Valgrind的massif工具运行程序
            massif_out = f"./massif_out/massif.out.{size}_{threads}"
            cmd = f"valgrind --tool=massif --stacks=yes --massif-out-file={massif_out} ./heated_plate_pthread {threads} g 10 {size}"
            print(f"  运行: {cmd}")
            
            run_output, stderr = run_command(cmd)
            
            # 检查是否成功运行
            if not os.path.exists(massif_out):
                print(f"  错误: 未能生成massif输出文件 {massif_out}")
                print(f"  错误信息: {stderr}")
                continue
            
            # 解析massif输出
            peak_memory, time_points, memory_points = parse_massif_output(massif_out)
            
            # 存储结果
            results[size][threads] = {
                'peak_memory': peak_memory,
                'time_points': time_points,
                'memory_points': memory_points,
                'massif_file': massif_out
            }
            
            print(f"  峰值内存使用: {peak_memory:.2f} KB")
            
            # 如果解析失败，尝试直接从输出中提取内存信息
            if peak_memory == 0:
                print("  尝试从valgrind输出中提取内存信息...")
                # 尝试从valgrind输出中提取内存信息
                mem_match = re.search(r'total heap usage: (\d+,\d+|\d+) allocs, (\d+,\d+|\d+) frees, (\d+,\d+|\d+) bytes allocated', stderr)
                if mem_match:
                    bytes_allocated = float(mem_match.group(3).replace(',', ''))
                    kb_allocated = bytes_allocated / 1024.0
                    results[size][threads]['peak_memory'] = kb_allocated
                    print(f"  从输出中提取的内存使用: {kb_allocated:.2f} KB")
    
    return results, thread_counts, problem_sizes

def generate_memory_reports(results, thread_counts, problem_sizes):
    """生成内存使用报告"""
    if not os.path.exists('./reports'):
        os.makedirs('./reports')
    
    # 检查是否有有效的内存数据
    has_valid_data = False
    for size in problem_sizes:
        for threads in thread_counts:
            if size in results and threads in results[size] and results[size][threads]['peak_memory'] > 0:
                has_valid_data = True
                break
        if has_valid_data:
            break
    
    if not has_valid_data:
        print("警告: 没有找到有效的内存数据，无法生成图表")
        # 创建一个简单的报告说明问题
        with open('./reports/memory_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("热平板问题内存分析报告\n")
            f.write("====================\n\n")
            f.write("未能获取有效的内存使用数据。可能的原因：\n")
            f.write("1. Valgrind massif工具未能正确运行\n")
            f.write("2. 解析massif输出文件失败\n")
            f.write("3. 程序内存使用过小，未被massif工具检测到\n\n")
            f.write("建议：\n")
            f.write("1. 检查Valgrind是否正确安装\n")
            f.write("2. 尝试增大问题规模\n")
            f.write("3. 检查massif输出文件内容\n")
        return
    
    # 生成峰值内存使用图表
    plt.figure(figsize=(12, 8))
    
    width = 0.2
    x = np.arange(len(problem_sizes))
    
    for i, threads in enumerate(thread_counts):
        peak_memories = []
        for size in problem_sizes:
            if size in results and threads in results[size]:
                peak_memories.append(results[size][threads]['peak_memory'])
            else:
                peak_memories.append(0)
        
        if any(peak_memories):  # 只有当有非零值时才绘制
            plt.bar(x + (i - len(thread_counts)/2 + 0.5) * width, peak_memories, width, label=f'{threads}线程')
    
    plt.xlabel('问题规模(N)')
    plt.ylabel('峰值内存使用(KB)')
    plt.title('不同问题规模和线程数的峰值内存使用')
    plt.xticks(x, problem_sizes)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('./reports/peak_memory_usage.png')
    
    # 为每个问题规模生成内存使用随时间变化的图表
    for size in problem_sizes:
        plt.figure(figsize=(12, 8))
        has_data = False
        
        for threads in thread_counts:
            if size in results and threads in results[size]:
                time_points = results[size][threads]['time_points']
                memory_points = results[size][threads]['memory_points']
                
                if time_points and memory_points and any(memory_points):
                    plt.plot(time_points, memory_points, marker='o', markersize=3, label=f'{threads}线程')
                    has_data = True
        
        if has_data:
            plt.xlabel('程序运行时间(ms)')
            plt.ylabel('内存使用(KB)')
            plt.title(f'问题规模N={size}的内存使用随时间变化')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'./reports/memory_usage_over_time_N{size}.png')
        else:
            plt.close()  # 如果没有数据，关闭图表
    
    # 生成文本报告
    with open('./reports/memory_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("热平板问题内存分析报告\n")
        f.write("====================\n\n")
        
        f.write("1. 峰值内存使用(KB)\n")
        f.write("-------------------------\n")
        f.write("问题规模(N)\\线程数\t" + "\t".join(map(str, thread_counts)) + "\n")
        
        for size in problem_sizes:
            line = f"{size}\t"
            for threads in thread_counts:
                if size in results and threads in results[size]:
                    line += f"{results[size][threads]['peak_memory']:.2f}\t"
                else:
                    line += "N/A\t"
            f.write(line + "\n")
        
        f.write("\n2. 内存消耗与问题规模关系分析\n")
        f.write("-------------------------\n")
        f.write("问题规模每增加一倍，理论上内存使用应该增加4倍（二维网格）。\n")
        f.write("实际增长率：\n")
        
        for threads in thread_counts:
            f.write(f"\n线程数: {threads}\n")
            prev_size = None
            prev_memory = None
            
            for size in problem_sizes:
                if size in results and threads in results[size]:
                    current_memory = results[size][threads]['peak_memory']
                    
                    if prev_size is not None and prev_memory > 0 and current_memory > 0:
                        ratio = current_memory / prev_memory
                        size_increase = size / prev_size
                        f.write(f"  N从{prev_size}增加到{size} (x{size_increase:.1f})，内存增加了{ratio:.2f}倍\n")
                    
                    prev_size = size
                    prev_memory = current_memory
        
        f.write("\n3. 内存消耗与线程数关系分析\n")
        f.write("-------------------------\n")
        f.write("随着线程数增加，每个线程需要自己的栈空间，因此内存使用会增加。\n")
        f.write("线程数增加对内存使用的影响：\n")
        
        for size in problem_sizes:
            f.write(f"\n问题规模: N={size}\n")
            base_threads = 1
            base_memory = None
            
            if size in results and base_threads in results[size]:
                base_memory = results[size][base_threads]['peak_memory']
                
                if base_memory > 0:  # 避免除零错误
                    for threads in thread_counts[1:]:  # 跳过第一个线程数
                        if threads in results[size] and results[size][threads]['peak_memory'] > 0:
                            current_memory = results[size][threads]['peak_memory']
                            increase = current_memory - base_memory
                            percentage = (increase / base_memory) * 100
                            f.write(f"  线程数从{base_threads}增加到{threads}，内存增加了{increase:.2f}KB ({percentage:.2f}%)\n")
                else:
                    f.write("  基准内存使用为0，无法计算增长率\n")

def main():
    print("开始内存分析...")
    results, thread_counts, problem_sizes = run_memory_analysis()
    print("生成内存分析报告...")
    generate_memory_reports(results, thread_counts, problem_sizes)
    print(f"分析完成，报告已保存到 reports/ 目录")

if __name__ == "__main__":
    main() 