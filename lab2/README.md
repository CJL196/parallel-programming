# Lab1: MPI Parallel Matrix Multiplication

This project implements a parallel matrix multiplication algorithm using MPI (Message Passing Interface) point-to-point communication. It measures performance under varying process counts and matrix sizes, comparing parallel execution with a serial baseline for verification.

## Project Overview

- **Purpose**: Implement and evaluate a parallel general matrix multiplication using MPI.
- **Input**: Three integers $ m $, $ n $, $ k $ (matrix dimensions), each in the range [128, 2048].
- **Problem**: Compute $ C = A \times B $, where $ A $ is an $ m \times n $ matrix and $ B $ is an $ n \times k $ matrix, both randomly generated.
- **Output**: Matrices $ A $, $ B $, $ C $, and the computation time.
- **Features**:
  - Parallel computation with MPI point-to-point communication.
  - Serial implementation for result verification.
  - Performance testing across different process counts (1-16) and matrix sizes (128-2048).

## Directory Structure

```
lab1/
├── main.c         # Main source code with MPI parallel matrix multiplication
├── makefile       # Build and run script
├── out.txt        # Performance test results
└── test.py        # Script to automate testing with various configurations
```

## Dependencies

- **MPI**: An MPI implementation (e.g., OpenMPI or MPICH).
- **C Compiler**: `mpicc` (MPI C compiler).
- **Python**: For running the `test.py` script (optional).


## How to Build and Run

### Build
Compile the project using the provided `makefile`:
```bash
make build
```
This generates the executable `main`.

### Run Manually
Run the program with specific matrix dimensions and process count:
```bash
mpirun -np <num_processes> ./main <m> <n> <k>
```
Example:
```bash
mpirun -np 4 ./main 512 512 512
```

### Run Automated Tests
Execute the `test.py` script to test all configurations (process counts: 1, 2, 4, 8, 16; matrix sizes: 128, 256, 512, 1024, 2048):
```bash
python test.py
```
Results will be printed to the console and saved in `out.txt`.

### Clean Up
Remove compiled files:
```bash
make clean
```

## Core Implementation

- **Parallelization**: The matrix $ A $ is divided by rows among processes. Each process computes a portion of $ C $ using local data, and results are gathered using `MPI_Gatherv`.
- **Verification**: A serial matrix multiplication function verifies the parallel results.
- **Timing**: Computation time is measured between `MPI_Barrier` calls to ensure synchronization.

See `main.c` for detailed code.

## Sample Results

Performance (in seconds) from `out.txt`:

| Processes / Size | 128    | 256    | 512    | 1024   | 2048   |
|------------------|--------|--------|--------|--------|--------|
| 1               | 0.0014 | 0.0112 | 0.2297 | 3.6402 | 56.0374|
| 2               | 0.0017 | 0.0059 | 0.1097 | 1.8022 | 40.6280|
| 4               | 0.0007 | 0.0032 | 0.0513 | 0.9666 | 23.9980|
| 8               | 0.0005 | 0.0033 | 0.0303 | 0.5627 | 9.8432 |
| 16              | 0.0010 | 0.0029 | 0.0331 | 0.4078 | 5.5830 |

- **Observation**: Time decreases with more processes, especially for larger matrices, though communication overhead can dominate for small sizes.

## Notes

- Ensure matrix dimensions are within [128, 2048], or the program will abort with an error.
- Results are verified against a serial implementation to ensure correctness.
- For detailed analysis and optimization suggestions, refer to the associated lab report.
