CXX = g++
TARGET = main
SRC = $(wildcard *.cpp)
OBJ = $(patsubst %.cpp, %.o, $(SRC))
CXXFLAGS = -c -Wall -O3 -I/opt/intel/oneapi/mkl/2025.0/include/
LDFLAGS = -lpthread -lm -ldl -L/opt/intel/oneapi/mkl/2025.0/lib/ -lmkl_intel_lp64 -lmkl_core -lmkl_sequential

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f *.o $(TARGET)

run: $(TARGET)
	./$(TARGET)