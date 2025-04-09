#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <cmath>

// 矩阵类
class Matrix {
public:
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

    // 构造函数
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(r, std::vector<double>(c, 0.0));
    }

    // 初始化随机矩阵
    void initRandom() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = dis(gen);
            }
        }
    }

    
    // 计算与另一个矩阵的最大误差
    double maxError(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            return -1.0; // 矩阵维度不匹配
        }
        
        double max_err = 0.0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double err = std::abs(data[i][j] - other.data[i][j]);
                max_err = std::max(max_err, err);
            }
        }
        
        return max_err;
    }
};

// 线程参数结构体
struct ThreadArgs {
    int thread_id;
    int num_threads;
    Matrix* A;
    Matrix* B;
    Matrix* C;
};

// 矩阵乘法线程函数
void matrixMultiplyThread(ThreadArgs* args) {
    int thread_id = args->thread_id;
    int num_threads = args->num_threads;
    Matrix* A = args->A;
    Matrix* B = args->B;
    Matrix* C = args->C;
    
    int m = A->rows;
    int n = A->cols;
    int k = B->cols;
    
    // 计算每个线程处理的行数
    int rows_per_thread = m / num_threads;
    int extra_rows = m % num_threads;
    
    // 计算当前线程的起始和结束行
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    
    // 执行矩阵乘法
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < k; j++) {
            double sum = 0.0;
            for (int l = 0; l < n; l++) {
                sum += A->data[i][l] * B->data[l][j];
            }
            C->data[i][j] = sum;
        }
    }
}

// 串行矩阵乘法函数
Matrix serialMatrixMultiply(const Matrix& A, const Matrix& B) {
    int m = A.rows;
    int n = A.cols;
    int k = B.cols;
    
    if (n != B.rows) {
        std::cerr << "错误: 矩阵维度不匹配，无法进行乘法运算" << std::endl;
        return Matrix(0, 0);
    }
    
    Matrix C(m, k);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            double sum = 0.0;
            for (int l = 0; l < n; l++) {
                sum += A.data[i][l] * B.data[l][j];
            }
            C.data[i][j] = sum;
        }
    }
    
    return C;
}

// 获取当前时间（毫秒）
double getTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
}

int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc != 5) {
        std::cout << "用法: " << argv[0] << " <m> <n> <k> <num_threads>" << std::endl;
        std::cout << "其中 m, n, k 是矩阵维度，取值范围为[128, 2048]" << std::endl;
        std::cout << "num_threads 是并行线程数" << std::endl;
        return 1;
    }
    
    // 解析命令行参数
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);
    int num_threads = std::atoi(argv[4]);
    
    // 验证参数
    if (m < 128 || m > 2048 || n < 128 || n > 2048 || k < 128 || k > 2048) {
        std::cout << "错误: 矩阵维度必须在[128, 2048]范围内" << std::endl;
        return 1;
    }
    
    if (num_threads <= 0) {
        std::cout << "错误: 线程数必须大于0" << std::endl;
        return 1;
    }
    
    // 限制线程数不超过行数
    if (num_threads > m) {
        num_threads = m;
        std::cout << "警告: 线程数已调整为" << num_threads << "（不超过矩阵行数）" << std::endl;
    }
    
    // 创建矩阵
    Matrix A(m, n);
    Matrix B(n, k);
    Matrix C(m, k);
    
    // 初始化随机矩阵
    A.initRandom();
    B.initRandom();
    
    
    // 创建线程
    std::vector<std::thread> threads;
    std::vector<ThreadArgs> thread_args(num_threads);
    
    // 记录开始时间
    double start_time = getTime();
    
    // 创建并启动线程
    for (int i = 0; i < num_threads; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].num_threads = num_threads;
        thread_args[i].A = &A;
        thread_args[i].B = &B;
        thread_args[i].C = &C;
        
        threads.emplace_back(matrixMultiplyThread, &thread_args[i]);
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    // 记录结束时间
    double end_time = getTime();
    double elapsed_time = end_time - start_time;
    
    
    // 打印计算时间
    std::cout << "并行计算时间: " << std::fixed << std::setprecision(2) << elapsed_time << " 毫秒" << std::endl;
    
    // 使用串行方法验证结果
    start_time = getTime();
    Matrix C_serial = serialMatrixMultiply(A, B);
    end_time = getTime();
    double serial_time = end_time - start_time;
    
    // 计算最大误差
    double max_error = C.maxError(C_serial);
    
    
    if (max_error < 1e-10) {
        std::cout << "验证结果: 并行计算正确！" << std::endl;
    } else {
        std::cout << "验证结果: 并行计算可能存在误差！" << std::endl;
    }
    
    
    return 0;
} 