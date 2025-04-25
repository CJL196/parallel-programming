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
    # 尝试匹配OpenMP格式
    match = re.search(r'Wallclock time = ([0-9.]+)', output)
    if match:
        return float(match.group(1))
    
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
    thread_counts = [1, 2, 4, 8, 16]
    schedule_types = {'s': '静态调度', 'd': '动态调度', 'g': '引导式调度'}
    chunk_sizes = [10, 50, 100]
    
    results = {
        'openmp': defaultdict(dict),
        'pthread': defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    }
    
    # 编译程序
    os.system("gcc -o ./src/heated_plate_pthread ./src/heated_plate_pthread.c ./src/parallel_for.c -lpthread -lm")
    os.system("gcc -o ./reference/heated_plate_openmp ./reference/heated_plate_openmp.c -fopenmp -lm")
    
    # 测试OpenMP版本
    for threads in thread_counts:
        cmd = f"export OMP_NUM_THREADS={threads} && ./reference/heated_plate_openmp"
        output = run_command(cmd)
        time = extract_time(output)
        iterations = extract_iterations(output)
        if time is not None and iterations is not None:
            results['openmp'][threads] = {'time': time, 'iterations': iterations}
    
    # 测试Pthread版本
    for threads in thread_counts:
        for sched, sched_name in schedule_types.items():
            for chunk in chunk_sizes:
                cmd = f"./src/heated_plate_pthread {threads} {sched} {chunk}"
                output = run_command(cmd)
                time = extract_time(output)
                iterations = extract_iterations(output)
                if time is not None and iterations is not None:
                    results['pthread'][threads][sched][chunk] = {
                        'time': time, 
                        'iterations': iterations
                    }
    
    return results, thread_counts, schedule_types, chunk_sizes

def generate_reports(results, thread_counts, schedule_types, chunk_sizes):
    """生成性能报告"""
    if not os.path.exists('./reports'):
        os.makedirs('./reports')
    
    # 生成线程数对比图
    plt.figure(figsize=(10, 6))
    openmp_times = [results['openmp'][t]['time'] if t in results['openmp'] else 0 for t in thread_counts]
    pthread_times = [results['pthread'][t]['s'][10]['time'] if t in results['pthread'] and 's' in results['pthread'][t] and 10 in results['pthread'][t]['s'] else 0 for t in thread_counts]
    
    x = np.arange(len(thread_counts))
    width = 0.35
    
    plt.bar(x - width/2, openmp_times, width, label='OpenMP')
    plt.bar(x + width/2, pthread_times, width, label='Pthread (静态调度)')
    
    plt.xlabel('线程数')
    plt.ylabel('执行时间 (秒)')
    plt.title('OpenMP vs Pthread - 不同线程数性能对比')
    plt.xticks(x, thread_counts)
    plt.legend()
    plt.savefig('./reports/thread_comparison.png')
    
    # 生成调度策略对比图
    plt.figure(figsize=(10, 6))
    
    sched_times = []
    for sched in schedule_types:
        sched_times.append([results['pthread'][4][sched][10]['time'] if 4 in results['pthread'] and sched in results['pthread'][4] and 10 in results['pthread'][4][sched] else 0])
    
    x = np.arange(len(schedule_types))
    plt.bar(x, [t[0] for t in sched_times], width)
    
    plt.xlabel('调度策略')
    plt.ylabel('执行时间 (秒)')
    plt.title('Pthread - 不同调度策略性能对比 (4线程)')
    plt.xticks(x, [schedule_types[s] for s in schedule_types])
    plt.savefig('./reports/schedule_comparison.png')
    
    # 生成块大小对比图
    plt.figure(figsize=(10, 6))
    
    chunk_times = [results['pthread'][4]['d'][c]['time'] if 4 in results['pthread'] and 'd' in results['pthread'][4] and c in results['pthread'][4]['d'] else 0 for c in chunk_sizes]
    
    plt.bar(range(len(chunk_sizes)), chunk_times)
    
    plt.xlabel('块大小')
    plt.ylabel('执行时间 (秒)')
    plt.title('Pthread - 不同块大小性能对比 (4线程，动态调度)')
    plt.xticks(range(len(chunk_sizes)), chunk_sizes)
    plt.savefig('./reports/chunk_comparison.png')
    
    # 生成文本报告
    with open('./reports/performance_report.txt', 'w', encoding='utf-8') as f:
        f.write("热平板问题性能测试报告\n")
        f.write("====================\n\n")
        
        f.write("1. OpenMP vs Pthread性能对比\n")
        f.write("-------------------------\n")
        f.write("线程数\tOpenMP时间(秒)\tPthread时间(秒)\t加速比\n")
        for t in thread_counts:
            if t in results['openmp'] and t in results['pthread'] and 's' in results['pthread'][t] and 10 in results['pthread'][t]['s']:
                openmp_time = results['openmp'][t]['time']
                pthread_time = results['pthread'][t]['s'][10]['time']
                speedup = openmp_time / pthread_time if pthread_time > 0 else 0
                f.write(f"{t}\t{openmp_time:.4f}\t\t{pthread_time:.4f}\t\t{speedup:.4f}\n")
        
        f.write("\n2. 不同调度策略性能对比 (4线程)\n")
        f.write("----------------------------\n")
        f.write("调度策略\t执行时间(秒)\t迭代次数\n")
        for sched, sched_name in schedule_types.items():
            if 4 in results['pthread'] and sched in results['pthread'][4] and 10 in results['pthread'][4][sched]:
                time = results['pthread'][4][sched][10]['time']
                iterations = results['pthread'][4][sched][10]['iterations']
                f.write(f"{sched_name}\t{time:.4f}\t\t{iterations}\n")
        
        f.write("\n3. 不同块大小性能对比 (4线程，动态调度)\n")
        f.write("--------------------------------\n")
        f.write("块大小\t执行时间(秒)\t迭代次数\n")
        for c in chunk_sizes:
            if 4 in results['pthread'] and 'd' in results['pthread'][4] and c in results['pthread'][4]['d']:
                time = results['pthread'][4]['d'][c]['time']
                iterations = results['pthread'][4]['d'][c]['iterations']
                f.write(f"{c}\t{time:.4f}\t\t{iterations}\n")
        
        f.write("\n4. 综合性能结果\n")
        f.write("-------------\n")
        f.write("线程数\t调度策略\t块大小\t执行时间(秒)\t迭代次数\n")
        best_time = float('inf')
        best_config = None
        
        for t in thread_counts:
            for sched, sched_name in schedule_types.items():
                for c in chunk_sizes:
                    if (t in results['pthread'] and sched in results['pthread'][t] and 
                        c in results['pthread'][t][sched]):
                        time = results['pthread'][t][sched][c]['time']
                        iterations = results['pthread'][t][sched][c]['iterations']
                        f.write(f"{t}\t{sched_name}\t{c}\t{time:.4f}\t\t{iterations}\n")
                        
                        if time < best_time:
                            best_time = time
                            best_config = (t, sched_name, c)
        
        if best_config:
            f.write(f"\n最佳配置: 线程数={best_config[0]}, 调度策略={best_config[1]}, 块大小={best_config[2]}, 时间={best_time:.4f}秒\n")

def main():
    print("开始性能测试...")
    results, thread_counts, schedule_types, chunk_sizes = run_tests()
    print("生成性能报告...")
    generate_reports(results, thread_counts, schedule_types, chunk_sizes)
    print(f"测试完成，报告已保存到 lab6/src/reports/ 目录")

if __name__ == "__main__":
    main() 