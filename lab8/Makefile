CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp
TARGET = mssp

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) data/updated_flower.csv data/test_flower.csv 8

clean:
	rm -rf $(TARGET)

.PHONY: all clean run
