#include <iostream>
#include <pthread.h>
#include <cmath>
#include <chrono>

// 全局变量用于存储计算结果和同步
struct ThreadData {
    double a, b, c;
    double discriminant;
    double sqrt_discriminant;
    double x1, x2;
    bool discriminant_ready = false;
    bool sqrt_ready = false;
    pthread_mutex_t mutex;
    pthread_cond_t cond_discriminant;
    pthread_cond_t cond_sqrt;
};

// 计算判别式的线程函数
void* calculateDiscriminant(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    double discriminant = data->b * data->b - 4 * data->a * data->c;
    
    pthread_mutex_lock(&data->mutex);
    data->discriminant = discriminant;
    data->discriminant_ready = true;
    pthread_cond_signal(&data->cond_discriminant);
    pthread_mutex_unlock(&data->mutex);
    
    return nullptr;
}

// 计算平方根的线程函数
void* calculateSqrt(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    pthread_mutex_lock(&data->mutex);
    while (!data->discriminant_ready) {
        pthread_cond_wait(&data->cond_discriminant, &data->mutex);
    }
    double sqrt_discriminant = sqrt(data->discriminant);
    data->sqrt_discriminant = sqrt_discriminant;
    data->sqrt_ready = true;
    pthread_cond_signal(&data->cond_sqrt);
    pthread_mutex_unlock(&data->mutex);
    
    return nullptr;
}

// 计算根的线程函数
void* calculateRoots(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    pthread_mutex_lock(&data->mutex);
    while (!data->sqrt_ready) {
        pthread_cond_wait(&data->cond_sqrt, &data->mutex);
    }
    data->x1 = (-data->b + data->sqrt_discriminant) / (2 * data->a);
    data->x2 = (-data->b - data->sqrt_discriminant) / (2 * data->a);
    pthread_mutex_unlock(&data->mutex);
    
    return nullptr;
}

int main(int argc, char* argv[]) {
    ThreadData data;
    
    // 初始化互斥锁和条件变量
    pthread_mutex_init(&data.mutex, nullptr);
    pthread_cond_init(&data.cond_discriminant, nullptr);
    pthread_cond_init(&data.cond_sqrt, nullptr);
    
    // 检查命令行参数
    if (argc != 4) {
        std::cerr << "用法: " << argv[0] << " a b c" << std::endl;
        return 1;
    }
    
    // 从命令行参数获取系数
    data.a = std::stod(argv[1]);
    data.b = std::stod(argv[2]); 
    data.c = std::stod(argv[3]);
    
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    
    // 创建线程
    pthread_t threads[3];
    pthread_create(&threads[0], nullptr, calculateDiscriminant, &data);
    pthread_create(&threads[1], nullptr, calculateSqrt, &data);
    pthread_create(&threads[2], nullptr, calculateRoots, &data);
    
    // 等待所有线程完成
    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], nullptr);
    }
    
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 输出结果
    std::cout << "方程的解: x1 = " << data.x1 << ", x2 = " << data.x2 << std::endl;
    std::cout << "计算耗时: " << duration.count() << " 微秒" << std::endl;
    
    // 清理资源
    pthread_mutex_destroy(&data.mutex);
    pthread_cond_destroy(&data.cond_discriminant);
    pthread_cond_destroy(&data.cond_sqrt);
    
    return 0;
}
