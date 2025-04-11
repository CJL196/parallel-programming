#include <iostream>
#include <pthread.h>
#include <random>
#include <chrono>
#include <cmath>

// 全局变量用于存储计算结果和同步
struct ThreadData {
    int total_points;
    int points_in_circle;
    int thread_count;
    pthread_mutex_t mutex;
};

// 线程函数：生成随机点并统计落在圆内的点数
void* monteCarloPi(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int points_per_thread = data->total_points / data->thread_count;
    int local_points_in_circle = 0;
    
    // 使用线程本地随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    // 生成随机点并统计
    for (int i = 0; i < points_per_thread; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        if (x * x + y * y <= 1.0) {
            local_points_in_circle++;
        }
    }
    
    // 使用互斥锁保护共享数据
    pthread_mutex_lock(&data->mutex);
    data->points_in_circle += local_points_in_circle;
    pthread_mutex_unlock(&data->mutex);
    
    return nullptr;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " 总点数 线程数" << std::endl;
        return 1;
    }
    
    int total_points = std::stoi(argv[1]);
    int thread_count = std::stoi(argv[2]);
    
    // 验证输入范围
    if (total_points < 1024 || total_points > 65536) {
        std::cerr << "错误：总点数必须在[1024, 65536]范围内" << std::endl;
        return 1;
    }
    
    if (thread_count <= 0) {
        std::cerr << "错误：线程数必须大于0" << std::endl;
        return 1;
    }
    
    ThreadData data;
    data.total_points = total_points;
    data.points_in_circle = 0;
    data.thread_count = thread_count;
    pthread_mutex_init(&data.mutex, nullptr);
    
    // 创建线程数组
    pthread_t* threads = new pthread_t[thread_count];
    
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    
    // 创建并启动线程
    for (int i = 0; i < thread_count; ++i) {
        pthread_create(&threads[i], nullptr, monteCarloPi, &data);
    }
    
    // 等待所有线程完成
    for (int i = 0; i < thread_count; ++i) {
        pthread_join(threads[i], nullptr);
    }
    
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 计算π的近似值
    double pi_estimate = 4.0 * data.points_in_circle / total_points;
    
    // 输出结果
    std::cout << "总点数: " << total_points << std::endl;
    std::cout << "圆内点数: " << data.points_in_circle << std::endl;
    std::cout << "π的估计值: " << pi_estimate << std::endl;
    std::cout << "计算耗时: " << duration.count() << " 微秒" << std::endl;
    
    // 清理资源
    delete[] threads;
    pthread_mutex_destroy(&data.mutex);
    
    return 0;
}
