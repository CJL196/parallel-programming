CC = mpicc
CFLAGS = -Wall -O2
TARGET = main
SOURCES = main.c
OBJECTS = $(SOURCES:.c=.o)
RUN_ARGS = 128 129 130
NP = 4

# 默认目标
all: build

# 编译目标
build: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 运行目标
run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) $(RUN_ARGS)

# 清理目标
clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all build run clean