#include <iostream>
#include <chrono>
#include "mkl.h"
using namespace std;

double ** generateMatrix(int m, int n, bool random=true)
{
    double **matrix = new double *[m];
    for (int i = 0; i < m; i++)
    {
        matrix[i] = new double[n];
    }
    if(random){
        for (int i = 0; i < m;++i){
            for (int j = 0; j < n;++j){
                matrix[i][j] = static_cast<double>(rand()) / RAND_MAX * 99.0;
            }
        }
    }else{
        for (int i = 0; i < m;++i){
            for (int j = 0; j < n;++j){
                matrix[i][j] = 0.0;
            }
        }
    }
    return matrix;
}
void destroyMatrix(double **matrix, int m)
{
    for (int i = 0; i < m; i++)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}



int main()
{
    int m = 512, n = 512, k = 512;
    double **a = generateMatrix(m, n);
    double **b = generateMatrix(n, k);
    double **c = generateMatrix(m, k, false);

    // 将二维数组转换为一维数组（MKL 需要连续内存布局）
    double *A = (double *)mkl_malloc(m * k * sizeof(double), 64);
    double *B = (double *)mkl_malloc(k * n * sizeof(double), 64);
    double *C = (double *)mkl_malloc(m * n * sizeof(double), 64);

    if (A == NULL || B == NULL || C == NULL)
    {
        cout << "Error: Memory allocation failed!" << endl;
        return 1;
    }

    // 将二维数组的数据复制到一维数组中
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            A[j + i * k] = a[i][j];
        }
    }
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            B[j + i * n] = b[i][j];
        }
    }
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            C[j + i * n] = c[i][j];
        }
    }

    // 设置 alpha 和 beta 参数
    double alpha = 1.0;
    double beta = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    // 调用 MKL 的 cblas_dgemm 函数进行矩阵乘法
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, alpha, A, k, B, n, beta, C, n);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<60>> duration = end - start;
    std::cout << "MKL: " << duration.count()*100 << "s\n";
    // 将结果从一维数组复制回二维数组
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            c[i][j] = C[j + i * n];
        }
    }


    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    destroyMatrix(a, m);
    destroyMatrix(b, n);
    destroyMatrix(c, m);

   
    return 0;
}