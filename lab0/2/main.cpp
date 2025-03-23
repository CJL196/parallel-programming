#include <iostream>
#include <chrono>
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

void multiply(double **a, double **b, double**c, int m, int n, int k)
{
    
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            for (int l = 0; l < n; l++)
            {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
}

int main()
{
    int m = 512, n = 512, k = 512;
    double **a = generateMatrix(m, n);
    double **b = generateMatrix(n, k);
    double **c = generateMatrix(m, k, false);
    auto start = std::chrono::high_resolution_clock::now();
    multiply(a, b, c, m, n, k);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::ratio<60>> duration = end - start;
    std::cout << "代码段执行时间为: " << duration.count()*100 << " 秒\n";
    destroyMatrix(a, m);
    destroyMatrix(b, n);
    destroyMatrix(c, m);

   
    return 0;
}