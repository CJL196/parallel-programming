1. MPI并行应用
使用MPI对快速傅里叶变换进行并行化。
问题描述：阅读参考文献中的串行傅里叶变换代码(fft_serial.cpp)，并使用MPI对其进行并行化。
要求：并行化：使用MPI多进程对fft_serial.cpp进行并行化。为适应MPI的消息传递机制，可能需要对fft_serial代码进行一定调整。

2．parallel_for并行应用分析
对于Lab6实现的parallel_for版本heated_plate_openmp应用，a) 改变并行规模（线程数）及问题规模（N），分析程序的并行性能，例如问题规模N，值为2，4，6，8，16，32，64，128，……；并行规模，值为1，2，4，8进程/线程。b) 使用Valgrind massif工具集采集并分析并行程序的内存消耗。注意Valgrind命令中增加--stacks=yes 参数采集程序运行栈内内存消耗。Valgrind massif输出日志（massif.out.pid）经过ms_print打印后示例如下图，其中x轴为程序运行时间，y轴为内存消耗量：
注：该工具使用可参考https://valgrind.org/docs/manual/ms-manual.html