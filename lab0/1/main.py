import random
import time

def generate_matrix(m, n, randomize=True):
    # 生成浮点数矩阵，范围为 [0.0, 99.0]
    matrix = [[random.uniform(0.0, 99.0) if randomize else 0.0 for _ in range(n)] for _ in range(m)]
    return matrix

def multiply(a, b, m, n, k):
    # 初始化结果矩阵为浮点数 0.0
    c = [[0.0] * k for _ in range(m)]
    for i in range(m):
        for j in range(k):
            for l in range(n):
                c[i][j] += a[i][l] * b[l][j]  # 浮点数乘法
    return c

def main():
    m, n, k = 512, 512, 512  # 矩阵维度
    a = generate_matrix(m, n)  # 生成浮点数矩阵 A
    b = generate_matrix(n, k)  # 生成浮点数矩阵 B
    
    start = time.time()
    c = multiply(a, b, m, n, k)  # 计算矩阵乘法
    end = time.time()
    
    print(f"代码段执行时间为: {end - start:.6f} 秒")

if __name__ == "__main__":
    main()