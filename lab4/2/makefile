CXX = g++
CXXFLAGS = -Wall -O2 -std=c++11 -pthread
TARGET = monte_carlo
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) 65536 8

.PHONY: all clean run