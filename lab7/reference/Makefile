CC = g++
CFLAGS = -Wall -Wextra -g

all: fft_serial

fft_serial: fft_serial.cpp
	$(CC) $(CFLAGS) -o fft_serial fft_serial.cpp

clean:
	rm -f fft_serial
	rm -f *.o

run: fft_serial
	./fft_serial

.PHONY: all clean run
