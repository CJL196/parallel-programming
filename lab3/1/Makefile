CXX = g++
CXXFLAGS = -Wall -O2 -std=c++11 -pthread
TARGET = matrix_multiply
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) 1024 1024 1024 4

.PHONY: all clean run