#pragma once
//all indices are 0 based
typedef struct { int n_rows, n_cols, nnz; int *values, *col_idx, *row_ptr; } crs_t;
typedef struct { int n_rows, n_cols, nnz; int *values, *row_idx, *col_ptr; } ccs_t;

//perm maps packed order into original order of row col
typedef struct {
    int n_rows, n_cols, nnz, num_jd;
    int *jdiag, *col_idx, *perm, *jdiag_ptr;
} jds_t;

typedef struct {
    int n_rows, n_cols, nnz, num_tjd;
    int *tjd, *row_idx, *perm, *tjd_ptr;
} tjds_t;

typedef struct { int i, j, v; } triplet_t;
typedef struct { int i, j; double v; } triplet_d_t;

typedef struct { int n_rows, n_cols, nnz; double *values; int *col_idx, *row_ptr; } crs_d_t;
typedef struct { int n_rows, n_cols, nnz; double *values; int *row_idx, *col_ptr; } ccs_d_t;

typedef struct {
    int n_rows, n_cols, nnz, num_tjd;
    double *tjd;
    int *row_idx, *perm, *tjd_ptr;
} tjds_d_t;

