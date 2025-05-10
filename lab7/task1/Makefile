CXX = mpic++
CXXFLAGS = -Wall -O3 -std=c++11
LDFLAGS = -lm
TARGET = fft_mpi
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	mpirun -np 4 ./$(TARGET)

.PHONY: all clean run 