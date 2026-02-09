#pragma once
#include "sparse_types.h"

void free_crs(crs_t *a);
void free_ccs(ccs_t *a);
void free_jds(jds_t *a);
void free_tjds(tjds_t *a);

void free_crs_d(crs_d_t *a);
void free_ccs_d(ccs_d_t *a);
void free_tjds_d(tjds_d_t *a);

crs_t  build_crs_from_dense(const int *dense, int n_rows, int n_cols);
ccs_t  build_ccs_from_dense(const int *dense, int n_rows, int n_cols);
jds_t  build_jds_from_dense(const int *dense, int n_rows, int n_cols);
tjds_t build_tjds_from_dense(const int *dense, int n_rows, int n_cols);

int build_crs_from_triplets(int n_rows, int n_cols, const triplet_t *t, int nnz, crs_t *out);
int build_ccs_from_triplets(int n_rows, int n_cols, const triplet_t *t, int nnz, ccs_t *out);

int build_crs_from_triplets_double(int n_rows, int n_cols, const triplet_d_t *t, int nnz, crs_d_t *out);
int build_ccs_from_triplets_double(int n_rows, int n_cols, const triplet_d_t *t, int nnz, ccs_d_t *out);

tjds_t   build_tjds_from_ccs(const ccs_t *c);
tjds_d_t build_tjds_from_ccs_double(const ccs_d_t *c);

void print_crs_hw(const crs_t *a);
void print_ccs_hw(const ccs_t *a);
void print_jds_hw(const jds_t *a);
void print_tjds_hw(const tjds_t *a);

