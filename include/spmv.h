#pragma once
#include "sparse_types.h"

void dense_spmv(const int *a, int n_rows, int n_cols, const int *x, int *y);
void crs_spmv(const crs_t *a, const int *x, int *y);
void ccs_spmv(const ccs_t *a, const int *x, int *y);
void jds_spmv(const jds_t *a, const int *x, int *y);
void tjds_spmv(const tjds_t *a, const int *x, int *y);

void crs_spmv_double(const crs_d_t *a, const double *x, double *y);
void tjds_spmv_double(const tjds_d_t *a, const double *x, double *y);

