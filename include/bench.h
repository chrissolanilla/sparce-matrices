#pragma once
#include "sparse_types.h"

long long checksum_vec(const int *y, int n);
double checksum_vec_double(const double *y, int n);

void bench_dense(const int *a, int n_rows, int n_cols, const int *x, int *y, int iters);
void bench_crs(const crs_t *a, const int *x, int *y, int iters);
void bench_tjds(const tjds_t *a, const int *x, int *y, int iters);

void bench_crs_double(const crs_d_t *a, const double *x, double *y, int iters);
void bench_tjds_double(const tjds_d_t *a, const double *x, double *y, int iters);

