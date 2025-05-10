# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <mpi.h>

using namespace std;

void ccopy(int n, double x[], double y[]);
void cfft2(int n, double x[], double y[], double w[], double sgn);
void cffti(int n, double w[]);
double ggl(double *seed);
void step(int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn);
void timestamp();

int main(int argc, char *argv[])
{
  int rank, size;
  double start_time, end_time;
  double ctime;
  double error;
  int first;
  double flops;
  double fnm1;
  int i;
  int icase;
  int it;
  int ln2;
  double mflops;
  int n;
  int nits = 10000;
  static double seed;
  double sgn;
  double *w = NULL;
  double *x = NULL;
  double *y = NULL;
  double *z = NULL;
  double *local_x = NULL;
  double *local_y = NULL;
  double *local_z = NULL;
  double z0, z1;
  int local_n;

  // 初始化MPI环境
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    timestamp();
    cout << "\n";
    cout << "FFT_MPI\n";
    cout << "  C++ version with MPI\n";
    cout << "\n";
    cout << "  Demonstrate an implementation of the Fast Fourier Transform\n";
    cout << "  of a complex data vector using " << size << " processes.\n";
    cout << "\n";
    cout << "  Accuracy check:\n";
    cout << "\n";
    cout << "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n";
    cout << "\n";
    cout << "             N      NITS    Error         Time          Time/Call     MFLOPS\n";
    cout << "\n";
  }

  seed = 331.0;
  n = 1;

  for (ln2 = 1; ln2 <= 20; ln2++)
  {
    n = 2 * n;

    // 确保n是进程数的倍数，简化并行划分
    if (n % size != 0) {
      if (rank == 0) {
        cout << "Skipping n = " << n << " as it's not divisible by process count " << size << "\n";
      }
      continue;
    }

    // 计算每个进程处理的元素数量
    local_n = n / size;

    // 分配内存
    w = new double[2 * n]; // 修正：确保w有足够的空间
    
    // 主进程需要完整数组用于初始化和验证
    if (rank == 0) {
      x = new double[2 * n];
      y = new double[2 * n];
      z = new double[2 * n];
      
      // 初始化为0，避免未初始化警告
      for (i = 0; i < 2 * n; i++) {
        x[i] = 0.0;
        y[i] = 0.0;
        z[i] = 0.0;
      }
    } else {
      // 非主进程也需要分配内存用于MPI操作
      x = new double[2 * n];
      y = new double[2 * n];
      z = new double[2 * n];
    }
    
    // 每个进程的本地数组
    local_x = new double[2 * local_n];
    local_y = new double[2 * local_n];
    local_z = new double[2 * local_n];
    
    // 初始化本地数组
    for (i = 0; i < 2 * local_n; i++) {
      local_x[i] = 0.0;
      local_y[i] = 0.0;
      local_z[i] = 0.0;
    }

    // 初始化正弦和余弦表
    cffti(n, w);

    first = 1;

    for (icase = 0; icase < 2; icase++)
    {
      if (rank == 0) {
        if (first) {
          // 初始化数据
          for (i = 0; i < 2 * n; i = i + 2) {
            z0 = ggl(&seed);
            z1 = ggl(&seed);
            x[i] = z0;
            z[i] = z0;
            x[i + 1] = z1;
            z[i + 1] = z1;
          }
        }
        else {
          // 初始化为零
          for (i = 0; i < 2 * n; i = i + 2) {
            z0 = 0.0;
            z1 = 0.0;
            x[i] = z0;
            z[i] = z0;
            x[i + 1] = z1;
            z[i + 1] = z1;
          }
        }
      }

      // 将数据分发给所有进程
      MPI_Scatter(x, 2 * local_n, MPI_DOUBLE,
                 local_x, 2 * local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // 广播正弦和余弦表
      MPI_Bcast(w, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // 第一次运行：精度检查
      if (first) {
        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);
        
        // 执行正向FFT
        sgn = +1.0;
        
        // 在本地执行FFT的一部分
        for (i = 0; i < 2 * local_n; i++) {
          local_y[i] = local_x[i];
        }
        
        // 收集所有进程的数据
        MPI_Allgather(local_y, 2 * local_n, MPI_DOUBLE,
                     y, 2 * local_n, MPI_DOUBLE, MPI_COMM_WORLD);
        
        // 主进程执行完整的FFT
        if (rank == 0) {
          cfft2(n, x, y, w, sgn);
        }
        
        // 分发结果
        MPI_Scatter(y, 2 * local_n, MPI_DOUBLE,
                   local_y, 2 * local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // 执行反向FFT
        sgn = -1.0;
        
        // 在本地执行FFT的一部分
        for (i = 0; i < 2 * local_n; i++) {
          local_x[i] = local_y[i];
        }
        
        // 收集所有进程的数据
        MPI_Allgather(local_x, 2 * local_n, MPI_DOUBLE,
                     x, 2 * local_n, MPI_DOUBLE, MPI_COMM_WORLD);
        
        // 主进程执行完整的FFT
        if (rank == 0) {
          cfft2(n, y, x, w, sgn);
          
          // 计算误差
          fnm1 = 1.0 / (double)n;
          error = 0.0;
          for (i = 0; i < 2 * n; i = i + 2) {
            error = error
              + pow(z[i] - fnm1 * x[i], 2)
              + pow(z[i + 1] - fnm1 * x[i + 1], 2);
          }
          error = sqrt(fnm1 * error);
          cout << "  " << setw(12) << n
               << "  " << setw(8) << nits
               << "  " << setw(12) << error;
        }
        
        first = 0;
      }
      else {
        // 性能测试
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        
        for (it = 0; it < nits; it++) {
          // 正向FFT
          sgn = +1.0;
          
          // 分发数据
          MPI_Scatter(x, 2 * local_n, MPI_DOUBLE,
                     local_x, 2 * local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          
          // 收集所有进程的数据
          MPI_Allgather(local_x, 2 * local_n, MPI_DOUBLE,
                       x, 2 * local_n, MPI_DOUBLE, MPI_COMM_WORLD);
          
          // 主进程执行完整的FFT
          if (rank == 0) {
            cfft2(n, x, y, w, sgn);
          }
          
          // 分发结果
          MPI_Scatter(y, 2 * local_n, MPI_DOUBLE,
                     local_y, 2 * local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          
          // 反向FFT
          sgn = -1.0;
          
          // 收集所有进程的数据
          MPI_Allgather(local_y, 2 * local_n, MPI_DOUBLE,
                       y, 2 * local_n, MPI_DOUBLE, MPI_COMM_WORLD);
          
          // 主进程执行完整的FFT
          if (rank == 0) {
            cfft2(n, y, x, w, sgn);
          }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        end_time = MPI_Wtime();
        ctime = end_time - start_time;
        
        if (rank == 0) {
          flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);
          mflops = flops / 1.0E+06 / ctime;
          
          cout << "  " << setw(12) << ctime
               << "  " << setw(12) << ctime / (double)(2 * nits)
               << "  " << setw(12) << mflops << "\n";
        }
      }
    }

    if ((ln2 % 4) == 0) {
      nits = nits / 10;
    }
    if (nits < 1) {
      nits = 1;
    }

    // 释放内存
    delete[] w;
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] local_x;
    delete[] local_y;
    delete[] local_z;
  }

  if (rank == 0) {
    cout << "\n";
    cout << "FFT_MPI:\n";
    cout << "  Normal end of execution.\n";
    cout << "\n";
    timestamp();
  }

  MPI_Finalize();
  return 0;
}

//****************************************************************************80

void ccopy(int n, double x[], double y[])
{
  int i;

  for (i = 0; i < n; i++) {
    y[i * 2 + 0] = x[i * 2 + 0];
    y[i * 2 + 1] = x[i * 2 + 1];
  }
  return;
}

//****************************************************************************80

void cfft2(int n, double x[], double y[], double w[], double sgn)
{
  int j;
  int m;
  int mj;
  int tgle;

  m = (int)(log((double)n) / log(1.99));
  mj = 1;

  // Toggling switch for work array.
  tgle = 1;
  step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

  if (n == 2) {
    return;
  }

  for (j = 0; j < m - 2; j++) {
    mj = mj * 2;
    if (tgle) {
      step(n, mj, &y[0 * 2 + 0], &y[(n / 2) * 2 + 0], &x[0 * 2 + 0], &x[mj * 2 + 0], w, sgn);
      tgle = 0;
    }
    else {
      step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);
      tgle = 1;
    }
  }

  // Last pass thru data: move y to x if needed 
  if (tgle) {
    ccopy(n, y, x);
  }

  mj = n / 2;
  step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

  return;
}

//****************************************************************************80

void cffti(int n, double w[])
{
  double arg;
  double aw;
  int i;
  int n2;
  const double pi = 3.141592653589793;

  n2 = n / 2;
  aw = 2.0 * pi / ((double)n);

  for (i = 0; i < n2; i++) {
    arg = aw * ((double)i);
    w[i * 2 + 0] = cos(arg);
    w[i * 2 + 1] = sin(arg);
  }
  return;
}

//****************************************************************************80

double ggl(double *seed)
{
  double d2 = 0.2147483647e10;
  double t;
  double value;

  t = *seed;
  t = fmod(16807.0 * t, d2);
  *seed = t;
  value = (t - 1.0) / (d2 - 1.0);

  return value;
}

//****************************************************************************80

void step(int n, int mj, double a[], double b[], double c[],
  double d[], double w[], double sgn)
{
  double ambr;
  double ambu;
  int j;
  int ja;
  int jb;
  int jc;
  int jd;
  int jw;
  int k;
  int lj;
  int mj2;
  double wjw[2];

  mj2 = 2 * mj;
  lj = n / mj2;

  for (j = 0; j < lj; j++) {
    jw = j * mj;
    ja = jw;
    jb = ja;
    jc = j * mj2;
    jd = jc;

    wjw[0] = w[jw * 2 + 0];
    wjw[1] = w[jw * 2 + 1];

    if (sgn < 0.0) {
      wjw[1] = -wjw[1];
    }

    for (k = 0; k < mj; k++) {
      c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
      c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

      ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0];
      ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

      d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
      d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
    }
  }
  return;
}

//****************************************************************************80

void timestamp()
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}
