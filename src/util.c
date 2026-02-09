#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <time.h>

#include "util.h"

long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}

void print_int_array(const char *name, const int *a, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n; i++) {
        printf("%d", a[i]);
        if (i + 1 < n) printf(", ");
    }
    printf("]\n");
}

void print_vec(const char *name, const int *v, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n; i++) {
        printf("%d", v[i]);
        if (i + 1 < n) printf(", ");
    }
    printf("]\n");
}

void print_matrix_6x6(int a[6][6]) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
}

void flatten_6x6(const int a[6][6], int *out_flat) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            out_flat[i * 6 + j] = a[i][j];
        }
    }
}

int count_nnz_dense(const int *dense, int n_rows, int n_cols) {
    int nnz = 0;
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            if (dense[i * n_cols + j] != 0) nnz++;
        }
    }
    return nnz;
}

