CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -lpthread -lm

# 目标文件
TARGETS = heated_plate_pthread heated_plate_test

all: $(TARGETS)

# 编译heated_plate_pthread
heated_plate_pthread: heated_plate_pthread.c parallel_for.c parallel_for.h
	$(CC) $(CFLAGS) -o heated_plate_pthread heated_plate_pthread.c parallel_for.c $(LDFLAGS)

# 编译测试程序
heated_plate_test: heated_plate_test.c heated_plate_pthread
	$(CC) $(CFLAGS) -o heated_plate_test heated_plate_test.c $(LDFLAGS)

# 编译OpenMP参考程序
openmp:
	$(CC) $(CFLAGS) -o ../reference/heated_plate_openmp ../reference/heated_plate_openmp.c -fopenmp $(LDFLAGS)

# 运行测试
test: heated_plate_test
	./heated_plate_test

# 运行性能分析
analyze:
	python3 performance_analyzer.py

# 清理
clean:
	rm -f $(TARGETS)
	rm -rf reports
	rm -f *.o

.PHONY: all openmp test analyze clean 