#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <pthread.h>
#include <iomanip>
#include <cstdlib>

// 全局变量
std::vector<int> A;  // 输入数组
long long sum = 0;   // 最终结果
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;  // 互斥锁

// 线程参数结构体
struct ThreadArgs {
    int thread_id;
    int num_threads;
    int start_idx;
    int end_idx;
};

// 线程函数
void* partial_sum(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    long long local_sum = 0;
    
    // 计算分配给该线程的数组部分的和
    for (int i = args->start_idx; i < args->end_idx; i++) {
        local_sum += A[i];
    }
    
    // 使用互斥锁保护全局和的更新
    pthread_mutex_lock(&mutex);
    sum += local_sum;
    pthread_mutex_unlock(&mutex);
    
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <数组大小n> <线程数>" << std::endl;
        return 1;
    }
    
    // 解析命令行参数
    int n = std::atoi(argv[1]);
    int num_threads = std::atoi(argv[2]);
    
    // 验证输入参数
    if (n < 1000000 || n > 128000000) {
        std::cerr << "错误: 数组大小必须在1M到128M之间" << std::endl;
        return 1;
    }
    
    if (num_threads <= 0) {
        std::cerr << "错误: 线程数必须为正数" << std::endl;
        return 1;
    }
    
    // 调整线程数，确保不超过数组大小
    num_threads = std::min(num_threads, n);
    
    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000);
    
    // 生成随机数组
    A.resize(n);
    for (int i = 0; i < n; i++) {
        A[i] = dis(gen);
    }
    
    // 创建线程数组和参数数组
    pthread_t* threads = new pthread_t[num_threads];
    ThreadArgs* thread_args = new ThreadArgs[num_threads];
    
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    
    // 创建并启动线程
    int chunk_size = n / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].num_threads = num_threads;
        thread_args[i].start_idx = i * chunk_size;
        thread_args[i].end_idx = (i == num_threads - 1) ? n : (i + 1) * chunk_size;
        
        if (pthread_create(&threads[i], NULL, partial_sum, &thread_args[i]) != 0) {
            std::cerr << "无法创建线程 " << i << std::endl;
            delete[] threads;
            delete[] thread_args;
            return 1;
        }
    }
    
    // 等待所有线程完成
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<60>> duration = end - start;
    
    std::cout << "并行计算结果: " << sum << std::endl;
    std::cout << "并行计算时间: " << duration.count()*60 << "秒" << std::endl;
    
    
    long long serial_sum = 0;
    for (int i = 0; i < n; i++) {
        serial_sum += A[i];
    }
    
    std::cout << "串行计算结果: " << serial_sum << std::endl;
    
    // 比较结果
    if (sum == serial_sum) {
        std::cout << "验证结果: 并行计算与串行计算结果一致 ✓" << std::endl;
    } else {
        std::cout << "验证结果: 并行计算与串行计算结果不一致 ✗" << std::endl;
        std::cout << "差异: " << std::abs(sum - serial_sum) << std::endl;
    }
    

    // 清理资源
    pthread_mutex_destroy(&mutex);
    delete[] threads;
    delete[] thread_args;
    
    return 0;
}
