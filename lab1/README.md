# Parallel Matrix Multiplication with MPI

This project implements a parallel matrix multiplication algorithm using MPI (Message Passing Interface) in C. It compares the performance of parallel computation against a serial implementation across various matrix sizes and process counts.

## Project Overview

- **Objective**: Accelerate matrix multiplication using MPI-based parallelism and verify correctness against a serial implementation.
- **Features**:
  - Supports matrix sizes from 128x128 to 2048x2048.
  - Distributes workload across multiple processes using row-based partitioning.
  - Measures execution time and verifies results.
- **Language**: C with MPI
- **Tools**: MPICC, Python (for testing)

## Directory Structure

```
lab1/
├── main         # Compiled executable
├── main.c       # Source code for parallel and serial matrix multiplication
├── makefile     # Makefile for building and running the project
├── out.txt      # Output log from test runs
└── test.py      # Python script to automate testing with varying process counts and matrix sizes
```

## Dependencies

- **MPI**: Install an MPI implementation (e.g., OpenMPI or MPICH).
  - Ubuntu: `sudo apt install libopenmpi-dev`
  - CentOS: `sudo yum install openmpi-devel`
- **GCC**: C compiler for building the project.
- **Python 3**: For running the test script (`test.py`).

## Usage

1. **Build the Project**:
   ```bash
   make build
   ```
   This compiles `main.c` into the executable `main` using `mpicc`.

2. **Run Manually**:
   ```bash
   mpirun -np <num_processes> ./main <m> <n> <k>
   ```
   - `<num_processes>`: Number of MPI processes (e.g., 4).
   - `<m> <n> <k>`: Matrix dimensions (e.g., 128 128 128 for \( A_{m \times n} \times B_{n \times k} \)).
   - Constraints: Matrix sizes must be between 128 and 2048.

3. **Run Automated Tests**:
   ```bash
   python test.py
   ```
   This script tests the program with process counts (1, 2, 4, 8, 16) and square matrix sizes (128, 256, 512, 1024, 2048), outputting results to the console.

4. **Clean Up**:
   ```bash
   make clean
   ```
   Removes object files and the executable.

## Sample Output

From `out.txt`:
```
mpirun -np 4 ./main 1024 1024 1024
Matrix multiplication completed in 0.628244 seconds
Verification passed!
```

## Performance Results

Execution times (in seconds) for different process counts and matrix sizes:

| Processes / Size | 128     | 256     | 512     | 1024    | 2048    |
|------------------|---------|---------|---------|---------|---------|
| 1                | 0.00158 | 0.00766 | 0.12191 | 2.25700 | 35.97293|
| 2                | 0.00066 | 0.00373 | 0.06618 | 1.03311 | 30.09187|
| 4                | 0.00099 | 0.00245 | 0.03762 | 0.62824 | 20.31865|
| 8                | 0.00175 | 0.00459 | 0.01786 | 0.42257 | 12.49699|
| 16               | 0.00063 | 0.00593 | 0.02407 | 0.34999 | 8.04389 |

- **Observation**: Speedup is significant for larger matrices (e.g., 2048x2048), with execution time dropping from 35.97s (1 process) to 8.04s (16 processes). Smaller matrices show less benefit due to communication overhead.

## Notes

- The program uses point-to-point communication (`MPI_Send` and `MPI_Recv`) for data distribution and result collection, which may limit scalability for very high process counts.
- Verification is performed by comparing parallel results against a serial computation, with a tolerance of \( 1 \times 10^{-10} \) for floating-point errors.

## License

This project is for educational purposes and has no specific license.
