CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -lpthread -lm

# 目标文件
TARGETS = src/heated_plate_pthread 

all: $(TARGETS)

# 编译heated_plate_pthread
src/heated_plate_pthread: src/heated_plate_pthread.c src/parallel_for.c src/parallel_for.h
	$(CC) $(CFLAGS) -o src/heated_plate_pthread src/heated_plate_pthread.c src/parallel_for.c $(LDFLAGS)

# 编译OpenMP参考程序
openmp:
	$(CC) $(CFLAGS) -o reference/heated_plate_openmp reference/heated_plate_openmp.c -fopenmp $(LDFLAGS)

# 清理
clean:
	rm -f src/heated_plate_pthread
	rm -f src/*.o

.PHONY: all openmp clean 