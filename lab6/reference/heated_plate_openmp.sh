#! /bin/bash
#
gcc -c -Wall -fopenmp heated_plate_openmp.c
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
gcc -fopenmp heated_plate_openmp.o -lm -o heated_plate_openmp
if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi
rm heated_plate_openmp.o
./heated_plate_openmp
#
echo "Normal end of execution."
