CC = gcc
CFLAGS = -Wall -O3 -fPIC
LDFLAGS = -pthread

all: libparallel_for.so test

libparallel_for.so: parallel_for.o
	$(CC) -shared -o $@ $^ $(LDFLAGS)

parallel_for.o: parallel_for.c parallel_for.h
	$(CC) $(CFLAGS) -c $<

test: test.o libparallel_for.so
	$(CC) -o $@ test.o -L. -lparallel_for $(LDFLAGS)

test.o: test.c parallel_for.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o *.so test

.PHONY: all clean 