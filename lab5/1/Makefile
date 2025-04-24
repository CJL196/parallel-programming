CXX = g++
CXXFLAGS = -std=c++11 -fopenmp -O3

TARGET = matrix_multiply
SRC = main.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) 

run: $(TARGET)
	./$(TARGET)