CXX = g++
CXXFLAGS = -Wall -O2 -std=c++11 -pthread
TARGET = quadratic_equation
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) 100 200 100

.PHONY: all clean run