CXX = g++
CXXFLAGS = -std=c++11 -pthread -O2

TARGET = parallel_sum
SRCS = main.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) 100000000 10

.PHONY: all clean run