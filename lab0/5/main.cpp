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

void multiply_ijl(double **a, double **b, double **c, int m, int n, int k) {
    const int UNROLL_FACTOR = 4;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            double sum = 0.0;
            int l = 0;
            for (; l + UNROLL_FACTOR - 1 < n; l += UNROLL_FACTOR) {
                sum += a[i][l] * b[l][j] + a[i][l + 1] * b[l + 1][j] +
                       a[i][l + 2] * b[l + 2][j] + a[i][l + 3] * b[l + 3][j];
            }
            for (; l < n; l++) {
                sum += a[i][l] * b[l][j];
            }
            c[i][j] = sum;
        }
    }
}

void multiply_ilj(double **a, double **b, double **c, int m, int n, int k) {
    const int UNROLL_FACTOR = 4;
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < n; l++) {
            double temp = a[i][l];
            int j = 0;
            for (; j + UNROLL_FACTOR - 1 < k; j += UNROLL_FACTOR) {
                c[i][j] += temp * b[l][j];
                c[i][j + 1] += temp * b[l][j + 1];
                c[i][j + 2] += temp * b[l][j + 2];
                c[i][j + 3] += temp * b[l][j + 3];
            }
            for (; j < k; j++) {
                c[i][j] += temp * b[l][j];
            }
        }
    }
}

void multiply_jil(double **a, double **b, double **c, int m, int n, int k) {
    const int UNROLL_FACTOR = 4;
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            int l = 0;
            for (; l + UNROLL_FACTOR - 1 < n; l += UNROLL_FACTOR) {
                sum += a[i][l] * b[l][j] + a[i][l + 1] * b[l + 1][j] +
                       a[i][l + 2] * b[l + 2][j] + a[i][l + 3] * b[l + 3][j];
            }
            for (; l < n; l++) {
                sum += a[i][l] * b[l][j];
            }
            c[i][j] = sum;
        }
    }
}

void multiply_jli(double **a, double **b, double **c, int m, int n, int k) {
    const int UNROLL_FACTOR = 4;
    for (int j = 0; j < k; j++) {
        for (int l = 0; l < n; l++) {
            double temp = b[l][j];
            int i = 0;
            for (; i + UNROLL_FACTOR - 1 < m; i += UNROLL_FACTOR) {
                c[i][j] += a[i][l] * temp;
                c[i + 1][j] += a[i + 1][l] * temp;
                c[i + 2][j] += a[i + 2][l] * temp;
                c[i + 3][j] += a[i + 3][l] * temp;
            }
            for (; i < m; i++) {
                c[i][j] += a[i][l] * temp;
            }
        }
    }
}

void multiply_lij(double **a, double **b, double **c, int m, int n, int k) {
    const int UNROLL_FACTOR = 4;
    for (int l = 0; l < n; l++) {
        for (int i = 0; i < m; i++) {
            double temp = a[i][l];
            int j = 0;
            for (; j + UNROLL_FACTOR - 1 < k; j += UNROLL_FACTOR) {
                c[i][j] += temp * b[l][j];
                c[i][j + 1] += temp * b[l][j + 1];
                c[i][j + 2] += temp * b[l][j + 2];
                c[i][j + 3] += temp * b[l][j + 3];
            }
            for (; j < k; j++) {
                c[i][j] += temp * b[l][j];
            }
        }
    }
}

void multiply_lji(double **a, double **b, double **c, int m, int n, int k) {
    const int UNROLL_FACTOR = 4;
    for (int l = 0; l < n; l++) {
        for (int j = 0; j < k; j++) {
            double temp = b[l][j];
            int i = 0;
            for (; i + UNROLL_FACTOR - 1 < m; i += UNROLL_FACTOR) {
                c[i][j] += a[i][l] * temp;
                c[i + 1][j] += a[i + 1][l] * temp;
                c[i + 2][j] += a[i + 2][l] * temp;
                c[i + 3][j] += a[i + 3][l] * temp;
            }
            for (; i < m; i++) {
                c[i][j] += a[i][l] * temp;
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