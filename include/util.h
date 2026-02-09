#pragma once
#include <time.h>

long long now_ns(void);

void print_int_array(const char *name, const int *a, int n);
void print_vec(const char *name, const int *v, int n);

void print_matrix_6x6(int a[6][6]);
void flatten_6x6(const int a[6][6], int *out_flat);

int count_nnz_dense(const int *dense, int n_rows, int n_cols);

