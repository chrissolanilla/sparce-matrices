#include <stdlib.h>

#include "spmv.h"

void dense_spmv(const int *a, int n_rows, int n_cols, const int *x, int *y) {
    for (int i = 0; i < n_rows; i++) {
        int sum = 0;
        for (int j = 0; j < n_cols; j++) {
            sum += a[i * n_cols + j] * x[j];
        }
        y[i] = sum;
    }
}

void crs_spmv(const crs_t *a, const int *x, int *y) {
    for (int i = 0; i < a->n_rows; i++) {
        int sum = 0;
        for (int k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++) {
            sum += a->values[k] * x[a->col_idx[k]];
        }
        y[i] = sum;
    }
}

void ccs_spmv(const ccs_t *a, const int *x, int *y) {
    for (int i = 0; i < a->n_rows; i++) y[i] = 0;

    for (int j = 0; j < a->n_cols; j++) {
        for (int k = a->col_ptr[j]; k < a->col_ptr[j + 1]; k++) {
            int i = a->row_idx[k];
            y[i] += a->values[k] * x[j];
        }
    }
}

void jds_spmv(const jds_t *a, const int *x, int *y) {
    int *temp = (int *)calloc((size_t)a->n_rows, sizeof(int));
    if (!temp) return;

    for (int d = 0; d < a->num_jd; d++) {
        int start = a->jdiag_ptr[d];
        int end = a->jdiag_ptr[d + 1];
        int len = end - start;

        for (int r = 0; r < len; r++) {
            int k = start + r;
            temp[r] += a->jdiag[k] * x[a->col_idx[k]];
        }
    }

    for (int r = 0; r < a->n_rows; r++) {
        y[a->perm[r]] = temp[r];
    }

    free(temp);
}

//iterates by diagonal depth d and then walks the coluimsn that have an entry at depth d
void tjds_spmv(const tjds_t *a, const int *x, int *y) {
    for (int i = 0; i < a->n_rows; i++) y[i] = 0;

    for (int d = 0; d < a->num_tjd; d++) {
        int start = a->tjd_ptr[d];
        int end = a->tjd_ptr[d + 1];
        int len = end - start;

        for (int cidx = 0; cidx < len; cidx++) {
            int k = start + cidx;
            int col = a->perm[cidx];
            int row = a->row_idx[k];
            y[row] += a->tjd[k] * x[col];
        }
    }
}

//q4, file has doubls not ints lol
void crs_spmv_double(const crs_d_t *a, const double *x, double *y) {
    for (int i = 0; i < a->n_rows; i++) {
        double sum = 0.0;
        for (int k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++) {
            sum += a->values[k] * x[a->col_idx[k]];
        }
        y[i] = sum;
    }
}

void tjds_spmv_double(const tjds_d_t *a, const double *x, double *y) {
    for (int i = 0; i < a->n_rows; i++) y[i] = 0.0;

    for (int d = 0; d < a->num_tjd; d++) {
        int start = a->tjd_ptr[d];
        int end = a->tjd_ptr[d + 1];
        int len = end - start;

        for (int cidx = 0; cidx < len; cidx++) {
            int k = start + cidx;
            int col = a->perm[cidx];
            int row = a->row_idx[k];
            y[row] += a->tjd[k] * x[col];
        }
    }
}

