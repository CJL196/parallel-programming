
### **1. parallel_for 并行应用**

使用此前构造的 `parallel_for` 并行结构，将 `heated_plate_openmp` 改造为基于 Pthreads 的并行应用。

#### **heated plate 问题描述：**
规则网格上的热传导模拟，其具体过程为每次循环中通过对邻域内热量平均模拟热传导过程，即：
$$
w_{i,j}^{t+1} = \frac{1}{4}(w_{i-1,j-1}^t + w_{i-1,j+1}^t + w_{i+1,j-1}^t + w_{i+1,j+1}^t),
$$
其 OpenMP 实现见课程资料中的 `heated_plate_openmp.c`。

#### **要求：**
使用此前构造的 `parallel_for` 并行结构，将 `heated_plate_openmp` 实现改造为基于 Pthreads 的并行应用。测试不同线程、调度方式下的程序并行性能，并与原始 `heated_plate_openmp.c` 实现对比。
