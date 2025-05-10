#!/bin/bash

# 创建报告目录
mkdir -p reports

echo "===== 运行性能分析 ====="
python3 analyze_performance.py

echo "===== 运行内存分析 ====="
python3 analyze_memory.py

echo "===== 分析完成 ====="
echo "报告已保存到 reports/ 目录" 