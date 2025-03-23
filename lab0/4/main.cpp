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

void multiply_ijl(double **a, double **b, double **c, int m, int n, int k)
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
void multiply_ilj(double **a, double **b, double **c, int m, int n, int k)
{
    for (int i = 0; i < m; i++)
    {
        for (int l = 0; l < n; l++)
        {
            for (int j = 0; j < k; j++)
            {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
}
void multiply_jil(double **a, double **b, double **c, int m, int n, int k)
{
    for (int j = 0; j < k; j++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int l = 0; l < n; l++)
            {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
}
void multiply_jli(double **a, double **b, double **c, int m, int n, int k)
{
    for (int j = 0; j < k; j++)
    {
        for (int l = 0; l < n; l++)
        {
            for (int i = 0; i < m; i++)
            {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
}
void multiply_lij(double **a, double **b, double **c, int m, int n, int k)
{
    for (int l = 0; l < n; l++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
}
void multiply_lji(double **a, double **b, double **c, int m, int n, int k)
{
    for (int l = 0; l < n; l++)
    {
        for (int j = 0; j < k; j++)
        {
            for (int i = 0; i < m; i++)
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
    multiply_ijl(a, b, c, m, n, k);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<60>> duration = end - start;
    std::cout << "ijl: " << duration.count()*100 << "s\n";
    
    destroyMatrix(c, m);
    c = generateMatrix(m, k, false);
    start = std::chrono::high_resolution_clock::now();
    multiply_ilj(a, b, c, m, n, k);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "ilj: " << duration.count()*100 << "s\n";

    destroyMatrix(c, m);
    c = generateMatrix(m, k, false);
    start = std::chrono::high_resolution_clock::now();
    multiply_jil(a, b, c, m, n, k);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "jil: " << duration.count()*100 << "s\n";

    destroyMatrix(c, m);
    c = generateMatrix(m, k, false);
    start = std::chrono::high_resolution_clock::now();
    multiply_jli(a, b, c, m, n, k);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "jli: " << duration.count()*100 << "s\n";

    destroyMatrix(c, m);
    c = generateMatrix(m, k, false);
    start = std::chrono::high_resolution_clock::now();
    multiply_lij(a, b, c, m, n, k);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "lij: " << duration.count()*100 << "s\n";

    destroyMatrix(c, m);
    c = generateMatrix(m, k, false);
    start = std::chrono::high_resolution_clock::now();
    multiply_lji(a, b, c, m, n, k);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "lji: " << duration.count()*100 << "s\n";

    destroyMatrix(a, m);
    destroyMatrix(b, n);
    destroyMatrix(c, m);

   
    return 0;
}