# Sparse Matrix Formats Benchmark: CRS, CCS, JDS,  TJDS

This project benchmarks sparse matrix-vector multiply

- **Dense**
- **CRS** (Compressed Row Storage)
- **CCS/CSC** (Compressed Column Storage)
- **JDS** (Jagged Diagonal Storage)
- **TJDS** (Transpose-JDS)

## Project layout

- `src/main.c` – entry point that runs Q1, Q2,Q3,  Q4
- `src/formats.c` – builders, frees  and print helpers for CRS/CCS/JDS/TJDS
- `src/spmv.c` – SpMV kernels + double versions
- `src/matrix_multiply_io.c` – .mtx loader and io
- `src/bench.c` – timing loops + checksum helpers
- `src/util.c` – small helpers for timing, printing, nnz count, and more
- `include/` – headers

Matrices:
- `ibm32.mtx`
- `memplus.mtx`

## Build  and run instuctions
This shoudl work on a linux machine. I am using arch. You can compile with the following command:
### manual build
```
gcc -Iinclude -O2 -Wall -Wextra -std=c11 src/main.c src/util.c src/matrix_multiply_io.c src/formats.c src/spmv.c src/bench.c -o main
```
### make file
```
make
```

running:
```
./main
```
I have a run.txt which makes it easy to compile and run. This uses my existing [run](https://github.com/chrissolanilla/run) utility.
This way its easy to build and run by simplying typing `run`

## output:

```
=== Q1 sanity matrix A 6x6 ===
10 0 0 0 -2 0
3 9 0 0 0 0
0 0 8 7 3 0
0 0 8 7 0 0
0 8 0 9 9 0
0 5 0 0 2 -1

CRS 0-based
nRows = 6
nCols = 6
nnz = 15
values = [10, -2, 3, 9, 8, 7, 3, 8, 7, 8, 9, 9, 5, 2, -1]
column_index = [0, 4, 0, 1, 2, 3, 4, 2, 3, 1, 3, 4, 1, 4, 5]
row_pointers = [0, 2, 4, 7, 9, 12, 15]

CcS 0-based
nRows = 6
nCols = 6
nnz = 15
values = [10, 3, 9, 8, 5, 8, 8, 7, 7, 9, -2, 3, 9, 2, -1]
row_index = [0, 1, 1, 4, 5, 2, 3, 2, 3, 4, 0, 2, 4, 5, 5]
column_pointers = [0, 2, 5, 7, 10, 14, 15]

JDS 0-based
nRows = 6
nCols = 6
nnz = 15
numJd = 3
jdiag = [8, 8, 5, 10, 3, 8, 7, 9, 2, -2, 9, 7, 3, 9, -1]
column_index = [2, 1, 1, 0, 0, 2, 3, 3, 4, 4, 1, 3, 4, 4, 5]
perm = [2, 4, 5, 0, 1, 3]
jdiag_ptr = [0, 6, 12, 15]

TJDS 0-based
nRows = 6
nCols = 6
nnz = 15
numTjd = 4
tjd = [-2, 9, 7, 10, 8, -1, 3, 8, 7, 3, 8, 9, 5, 9, 2]
row_index = [0, 1, 2, 0, 2, 5, 2, 4, 3, 1, 3, 4, 5, 4, 5]
perm = [4, 1, 3, 0, 2, 5]
tjd_ptr = [0, 6, 11, 14, 15]

x = [1, 1, 1, 1, 1, 1]
y dense = [8, 12, 18, 15, 26, 6]
y crs = [8, 12, 18, 15, 26, 6]
y ccs = [8, 12, 18, 15, 26, 6]
y jds = [8, 12, 18, 15, 26, 6]
y tjds = [8, 12, 18, 15, 26, 6]

=== Q2/Q3 IBM32 loaded ===
file = ibm32.mtx
nRows = 32
nCols = 32
nnz  = 126

dense iters = 1000 time_ns = 384837 check = 6000
dense iters = 10000 time_ns = 3890972 check = 60000
crs iters = 1000 time_ns = 91440 check = 6000
crs iters = 10000 time_ns = 877783 check = 60000
tjds iters = 1000 time_ns = 107550 check = 6000
tjds iters = 10000 time_ns = 1070212 check = 60000

quick y peek: y[0]=6 y[1]=6 y[2]=8

=== Q4 MEMPLUS loaded (double) ===
file = memplus.mtx
nRows = 17758
nCols = 17758
nnz  = 126150

crs iters = 1000 time_ns = 85977006 check = 0.0395903
crs iters = 10000 time_ns = 857776616 check = 37.8752
tjds iters = 1000 time_ns = 128432472 check = 0.0395903
tjds iters = 10000 time_ns = 1283238167 check = 37.8752

verify:
  checksum crs  = 1096310.63023
  checksum tjds = 1096310.63023
  y[0]=8.17812e-05 vs 8.17812e-05
  y[1]=0.000567391 vs 0.000567391
  y[2]=5.11383e-05 vs 5.11383e-05
  ok: checksums match
```
