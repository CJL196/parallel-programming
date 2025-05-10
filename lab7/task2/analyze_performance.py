#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def run_command(cmd):
    """运行命令并返回输出结果"""
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode('utf-8')

def extract_time(output):
    """从输出中提取运行时间"""
    
    # 尝试匹配Pthread格式
    match = re.search(r'Elapsed time = ([0-9.]+) seconds', output)
    if match:
        return float(match.group(1))
    
    return None

def extract_iterations(output):
    """从输出中提取迭代次数"""
    lines = output.strip().split('\n')
    for line in reversed(lines):
        match = re.search(r'^\s*(\d+)\s+', line)
        if match:
            return int(match.group(1))
    return None

def run_tests():
    """运行测试并收集结果"""
    thread_counts = [1, 2, 4, 8]  # 并行规模
    schedule_types = {'g': '引导式调度'}
    chunk_sizes = [10]
    problem_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]  # 问题规模N
    
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # 编译程序
    os.system("gcc -o ./heated_plate_pthread ./heated_plate_pthread.c ./parallel_for.c -lpthread -lm")
    
    # 测试不同问题规模和线程数
    for size in problem_sizes:
        print(f"测试问题规模 N = {size}")
        for threads in thread_counts:
            for sched, sched_name in schedule_types.items():
                for chunk in chunk_sizes:
                    cmd = f"./heated_plate_pthread {threads} {sched} {chunk} {size}"
                    print(f"  运行: {cmd}")
                    output = run_command(cmd)
                    time = extract_time(output)
                    iterations = extract_iterations(output)
                    if time is not None and iterations is not None:
                        results[size][threads][sched][chunk] = {
                            'time': time, 
                            'iterations': iterations
                        }
    
    return results, thread_counts, schedule_types, chunk_sizes, problem_sizes

def generate_reports(results, thread_counts, schedule_types, chunk_sizes, problem_sizes):
    """生成性能报告"""
    if not os.path.exists('./reports'):
        os.makedirs('./reports')
    
    # 生成问题规模-执行时间图
    plt.figure(figsize=(12, 8))
    
    markers = ['o', 's', 'D', '^']
    
    for i, threads in enumerate(thread_counts):
        times = []
        valid_sizes = []
        for size in problem_sizes:
            if size in results and threads in results[size] and 'g' in results[size][threads] and 10 in results[size][threads]['g']:
                times.append(results[size][threads]['g'][10]['time'])
                valid_sizes.append(size)
        
        if valid_sizes:
            plt.plot(valid_sizes, times, marker=markers[i % len(markers)], label=f'{threads}线程')
    
    plt.xlabel('问题规模(N)')
    plt.ylabel('执行时间(秒)')
    plt.title('不同问题规模下的执行时间')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('./reports/problem_size_comparison.png')
    
    # 生成加速比图
    plt.figure(figsize=(12, 8))
    
    for size in problem_sizes:
        if size in results and 1 in results[size] and 'g' in results[size][1] and 10 in results[size][1]['g']:
            serial_time = results[size][1]['g'][10]['time']
            speedups = []
            valid_threads = []
            
            for threads in thread_counts:
                if threads in results[size] and 'g' in results[size][threads] and 10 in results[size][threads]['g']:
                    parallel_time = results[size][threads]['g'][10]['time']
                    speedup = serial_time / parallel_time
                    speedups.append(speedup)
                    valid_threads.append(threads)
            
            if valid_threads and len(valid_threads) > 1:  # 至少需要两个点才能画线
                plt.plot(valid_threads, speedups, marker='o', label=f'N={size}')
    
    # 添加理想加速比参考线
    max_threads = max(thread_counts)
    plt.plot([1, max_threads], [1, max_threads], 'k--', label='理想加速比')
    
    plt.xlabel('线程数')
    plt.ylabel('加速比')
    plt.title('不同问题规模下的加速比')
    plt.grid(True)
    plt.legend()
    plt.savefig('./reports/speedup_comparison.png')
    
    # 生成效率图
    plt.figure(figsize=(12, 8))
    
    for size in problem_sizes:
        if size in results and 1 in results[size] and 'g' in results[size][1] and 10 in results[size][1]['g']:
            serial_time = results[size][1]['g'][10]['time']
            efficiencies = []
            valid_threads = []
            
            for threads in thread_counts:
                if threads in results[size] and 'g' in results[size][threads] and 10 in results[size][threads]['g']:
                    parallel_time = results[size][threads]['g'][10]['time']
                    efficiency = (serial_time / parallel_time) / threads
                    efficiencies.append(efficiency)
                    valid_threads.append(threads)
            
            if valid_threads and len(valid_threads) > 1:
                plt.plot(valid_threads, efficiencies, marker='o', label=f'N={size}')
    
    plt.xlabel('线程数')
    plt.ylabel('并行效率')
    plt.title('不同问题规模下的并行效率')
    plt.grid(True)
    plt.legend()
    plt.savefig('./reports/efficiency_comparison.png')
    
    # 生成文本报告
    with open('./reports/performance_report.txt', 'w', encoding='utf-8') as f:
        f.write("热平板问题性能测试报告\n")
        f.write("====================\n\n")
        
        f.write("1. 执行时间(秒)\n")
        f.write("-------------------------\n")
        f.write("问题规模(N)\\线程数\t" + "\t".join(map(str, thread_counts)) + "\n")
        
        for size in problem_sizes:
            if size in results:
                line = f"{size}\t"
                for threads in thread_counts:
                    if threads in results[size] and 'g' in results[size][threads] and 10 in results[size][threads]['g']:
                        time = results[size][threads]['g'][10]['time']
                        line += f"{time:.4f}\t"
                    else:
                        line += "N/A\t"
                f.write(line + "\n")
        
        f.write("\n2. 加速比\n")
        f.write("-------------------------\n")
        f.write("问题规模(N)\\线程数\t" + "\t".join(map(str, thread_counts[1:])) + "\n")
        
        for size in problem_sizes:
            if size in results and 1 in results[size] and 'g' in results[size][1] and 10 in results[size][1]['g']:
                serial_time = results[size][1]['g'][10]['time']
                line = f"{size}\t"
                
                for threads in thread_counts[1:]:  # 跳过1线程
                    if threads in results[size] and 'g' in results[size][threads] and 10 in results[size][threads]['g']:
                        parallel_time = results[size][threads]['g'][10]['time']
                        speedup = serial_time / parallel_time
                        line += f"{speedup:.4f}\t"
                    else:
                        line += "N/A\t"
                f.write(line + "\n")
        
        f.write("\n3. 并行效率\n")
        f.write("-------------------------\n")
        f.write("问题规模(N)\\线程数\t" + "\t".join(map(str, thread_counts[1:])) + "\n")
        
        for size in problem_sizes:
            if size in results and 1 in results[size] and 'g' in results[size][1] and 10 in results[size][1]['g']:
                serial_time = results[size][1]['g'][10]['time']
                line = f"{size}\t"
                
                for threads in thread_counts[1:]:  # 跳过1线程
                    if threads in results[size] and 'g' in results[size][threads] and 10 in results[size][threads]['g']:
                        parallel_time = results[size][threads]['g'][10]['time']
                        efficiency = (serial_time / parallel_time) / threads
                        line += f"{efficiency:.4f}\t"
                    else:
                        line += "N/A\t"
                f.write(line + "\n")

def main():
    print("开始性能测试...")
    results, thread_counts, schedule_types, chunk_sizes, problem_sizes = run_tests()
    print("生成性能报告...")
    generate_reports(results, thread_counts, schedule_types, chunk_sizes, problem_sizes)
    print(f"测试完成，报告已保存到 reports/ 目录")

if __name__ == "__main__":
    main() 