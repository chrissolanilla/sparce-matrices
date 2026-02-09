#include <stdio.h>
#include "bench.h"
#include "util.h"
#include "spmv.h"

long long checksum_vec(const int *y, int n) {
    long long s = 0;
    for (int i = 0; i < n; i++) s += (long long)y[i] * (long long)(i + 1);
    return s;
}

double checksum_vec_double(const double *y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += y[i] * (double)(i + 1);
    return s;
}

void bench_crs_double(const crs_d_t *a, const double *x, double *y, int iters) {
    long long t0 = now_ns();
    double check = 0.0;
    for (int k = 0; k < iters; k++) {
        crs_spmv_double(a, x, y);
        check += y[k % a->n_rows];
    }
    long long t1 = now_ns();
    printf("crs iters = %d time_ns = %lld check = %.6g\n", iters, (t1 - t0), check);
}

void bench_tjds_double(const tjds_d_t *a, const double *x, double *y, int iters) {
    long long t0 = now_ns();
    double check = 0.0;
    for (int k = 0; k < iters; k++) {
        tjds_spmv_double(a, x, y);
        check += y[k % a->n_rows];
    }
    long long t1 = now_ns();
    printf("tjds iters = %d time_ns = %lld check = %.6g\n", iters, (t1 - t0), check);
}


void bench_dense(const int *a, int n_rows, int n_cols, const int *x, int *y, int iters) {
    long long t0 = now_ns();
    long long check = 0;
    for (int k = 0; k < iters; k++) {
        dense_spmv(a, n_rows, n_cols, x, y);
        check += y[0];
    }
    long long t1 = now_ns();
    printf("dense iters = %d time_ns = %lld check = %lld\n", iters, (t1 - t0), check);
}

void bench_crs(const crs_t *a, const int *x, int *y, int iters) {
    long long t0 = now_ns();
    long long check = 0;
    for (int k = 0; k < iters; k++) {
        crs_spmv(a, x, y);
        check += y[0];
    }
    long long t1 = now_ns();
    printf("crs iters = %d time_ns = %lld check = %lld\n", iters, (t1 - t0), check);
}

void bench_tjds(const tjds_t *a, const int *x, int *y, int iters) {
    long long t0 = now_ns();
    long long check = 0;
    for (int k = 0; k < iters; k++) {
        tjds_spmv(a, x, y);
        check += y[0];
    }
    long long t1 = now_ns();
    printf("tjds iters = %d time_ns = %lld check = %lld\n", iters, (t1 - t0), check);
}

