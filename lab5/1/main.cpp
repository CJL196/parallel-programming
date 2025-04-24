#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <iomanip>

using namespace std;

// 生成随机矩阵
void generateRandomMatrix(vector<vector<double>>& matrix, int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = dis(gen);
        }
    }
}

// 矩阵乘法函数
void matrixMultiply(const vector<vector<double>>& A, 
                    const vector<vector<double>>& B,
                    vector<vector<double>>& C,
                    int m, int n, int k,
                    const string& schedule = "default",
                    int chunk_size = 1) {
    if (schedule == "static") {
        #pragma omp parallel for collapse(2) schedule(static, chunk_size)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                double sum = 0.0;
                for (int l = 0; l < n; l++) {
                    sum += A[i][l] * B[l][j];
                }
                C[i][j] = sum;
            }
        }
    }
    else if (schedule == "dynamic") {
        #pragma omp parallel for collapse(2) schedule(dynamic, chunk_size)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                double sum = 0.0;
                for (int l = 0; l < n; l++) {
                    sum += A[i][l] * B[l][j];
                }
                C[i][j] = sum;
            }
        }
    }
    else {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                double sum = 0.0;
                for (int l = 0; l < n; l++) {
                    sum += A[i][l] * B[l][j];
                }
                C[i][j] = sum;
            }
        }
    }
}

int main() {
    vector<int> matrix_sizes = {128, 256, 512, 1024, 2048};
    vector<int> thread_counts = {1, 2, 4, 8, 16};
    vector<string> schedules = {"default", "static", "dynamic"};
    
    cout << "矩阵大小\t线程数\t调度方式\t执行时间(ms)" << endl;
    
    for (int size : matrix_sizes) {
        int m = size, n = size, k = size;
        
        // 初始化矩阵
        vector<vector<double>> A(m, vector<double>(n));
        vector<vector<double>> B(n, vector<double>(k));
        vector<vector<double>> C(m, vector<double>(k));
        
        // 生成随机矩阵
        generateRandomMatrix(A, m, n);
        generateRandomMatrix(B, n, k);
        
        for (int num_threads : thread_counts) {
            omp_set_num_threads(num_threads);
            
            for (const string& schedule : schedules) {
                // 计时开始
                auto start = chrono::high_resolution_clock::now();
                
                // 执行矩阵乘法
                matrixMultiply(A, B, C, m, n, k, schedule);
                
                // 计时结束
                auto end = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
                
                cout << size << "\t\t" 
                     << num_threads << "\t\t" 
                     << schedule << "\t\t" 
                     << duration.count() << endl;
            }
        }
    }
    
    return 0;
}
