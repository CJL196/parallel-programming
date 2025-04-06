# 编译器
CC = mpicc

# 编译选项
CFLAGS = -Wall -O2

# 链接选项（添加数学库）
LDFLAGS = -lm

# 目标可执行文件
TARGET = main

# 源文件和目标文件
SOURCES = main.c
OBJECTS = $(SOURCES:.c=.o)

# 运行参数
RUN_ARGS = 128
NP = 4

# 默认目标
all: build

# 编译目标
build: $(TARGET)

# 链接目标文件生成可执行文件
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# 编译源文件为目标文件
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 运行目标
run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) $(RUN_ARGS)

# 清理目标
clean:
	rm -f $(OBJECTS) $(TARGET)

# 声明伪目标
.PHONY: all build run clean