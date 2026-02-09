#pragma once
#include "sparse_types.h"

int mm_read_dense_int(const char *path, int **out_a, int *out_rows, int *out_cols, int *out_nnz);

int mm_read_triplets_int(const char *path, triplet_t **out_t, int *out_rows, int *out_cols, int *out_nnz);
int mm_read_triplets_double(const char *path, triplet_d_t **out_t, int *out_rows, int *out_cols, int *out_nnz);

