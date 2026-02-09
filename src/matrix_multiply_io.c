#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix_multiply_io.h"

int mm_read_dense_int(const char *path, int **out_a, int *out_rows, int *out_cols, int *out_nnz) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    char line[512];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return 0; }

    if (line[0] != '%' || line[1] != '%') { fclose(f); return 0; }

    int is_pattern = 0;
    int symmetric = 0;
    if (strstr(line, "pattern")) is_pattern = 1;
    if (strstr(line, "skew-symmetric")) symmetric = 2;
    else if (strstr(line, "symmetric")) symmetric = 1;

    do {
        if (!fgets(line, sizeof(line), f)) { fclose(f); return 0; }
    } while (line[0] == '%');

    int n_rows = 0, n_cols = 0, nnz = 0;
    if (sscanf(line, "%d %d %d", &n_rows, &n_cols, &nnz) != 3) { fclose(f); return 0; }

    int *a = (int *)calloc((size_t)n_rows * (size_t)n_cols, sizeof(int));
    if (!a) { fclose(f); return 0; }

    for (int k = 0; k < nnz; k++) {
        int i = 0, j = 0;
        double v = 1.0;

        if (is_pattern) {
            if (fscanf(f, "%d %d", &i, &j) != 2) { free(a); fclose(f); return 0; }
            v = 1.0;
        } else {
            if (fscanf(f, "%d %d %lf", &i, &j, &v) != 3) { free(a); fclose(f); return 0; }
        }

        i--; j--;
        a[i * n_cols + j] = (int)v;

        if (symmetric == 1 && i != j) a[j * n_cols + i] = (int)v;
        else if (symmetric == 2 && i != j) a[j * n_cols + i] = -(int)v;
    }

    fclose(f);

    *out_a = a;
    *out_rows = n_rows;
    *out_cols = n_cols;
    *out_nnz = nnz;
    return 1;
}

int mm_read_triplets_int(const char *path, triplet_t **out_t, int *out_rows, int *out_cols, int *out_nnz) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    char line[512];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return 0; }
    if (line[0] != '%' || line[1] != '%') { fclose(f); return 0; }

    int is_pattern = 0;
    int symmetric = 0;
    if (strstr(line, "pattern")) is_pattern = 1;
    if (strstr(line, "skew-symmetric")) symmetric = 2;
    else if (strstr(line, "symmetric")) symmetric = 1;

    do {
        if (!fgets(line, sizeof(line), f)) { fclose(f); return 0; }
    } while (line[0] == '%');

    int n_rows = 0, n_cols = 0, nnz = 0;
    if (sscanf(line, "%d %d %d", &n_rows, &n_cols, &nnz) != 3) { fclose(f); return 0; }

    int cap = (symmetric ? (2 * nnz) : nnz);
    triplet_t *t = (triplet_t *)malloc((size_t)cap * sizeof(triplet_t));
    if (!t) { fclose(f); return 0; }

    int used = 0;

    for (int k = 0; k < nnz; k++) {
        int i = 0, j = 0;
        double v = 1.0;

        if (is_pattern) {
            if (fscanf(f, "%d %d", &i, &j) != 2) { free(t); fclose(f); return 0; }
            v = 1.0;
        } else {
            if (fscanf(f, "%d %d %lf", &i, &j, &v) != 3) { free(t); fclose(f); return 0; }
        }

        i--; j--;

        if (used >= cap) {
            cap *= 2;
            triplet_t *nt = (triplet_t *)realloc(t, (size_t)cap * sizeof(triplet_t));
            if (!nt) { free(t); fclose(f); return 0; }
            t = nt;
        }

        t[used++] = (triplet_t){ .i = i, .j = j, .v = (int)v };

        if (symmetric && i != j) {
            int vv = (int)v;
            if (symmetric == 2) vv = -vv;

            if (used >= cap) {
                cap *= 2;
                triplet_t *nt = (triplet_t *)realloc(t, (size_t)cap * sizeof(triplet_t));
                if (!nt) { free(t); fclose(f); return 0; }
                t = nt;
            }

            t[used++] = (triplet_t){ .i = j, .j = i, .v = vv };
        }
    }

    fclose(f);

    *out_t = t;
    *out_rows = n_rows;
    *out_cols = n_cols;
    *out_nnz = used;
    return 1;
}

int mm_read_triplets_double(const char *path, triplet_d_t **out_t, int *out_rows, int *out_cols, int *out_nnz) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    char line[512];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return 0; }
    if (line[0] != '%' || line[1] != '%') { fclose(f); return 0; }

    int is_pattern = 0;
    int symmetric = 0;
    if (strstr(line, "pattern")) is_pattern = 1;
    if (strstr(line, "skew-symmetric")) symmetric = 2;
    else if (strstr(line, "symmetric")) symmetric = 1;

    do {
        if (!fgets(line, sizeof(line), f)) { fclose(f); return 0; }
    } while (line[0] == '%');

    int n_rows = 0, n_cols = 0, nnz = 0;
    if (sscanf(line, "%d %d %d", &n_rows, &n_cols, &nnz) != 3) { fclose(f); return 0; }

    int cap = (symmetric ? (2 * nnz) : nnz);
    triplet_d_t *t = (triplet_d_t *)malloc((size_t)cap * sizeof(triplet_d_t));
    if (!t) { fclose(f); return 0; }

    int used = 0;

    for (int k = 0; k < nnz; k++) {
        int i = 0, j = 0;
        double v = 1.0;

        if (is_pattern) {
            if (fscanf(f, "%d %d", &i, &j) != 2) { free(t); fclose(f); return 0; }
            v = 1.0;
        } else {
            if (fscanf(f, "%d %d %lf", &i, &j, &v) != 3) { free(t); fclose(f); return 0; }
        }

        i--; j--;

        if (i < 0 || i >= n_rows || j < 0 || j >= n_cols) { free(t); fclose(f); return 0; }

        if (used >= cap) {
            cap *= 2;
            triplet_d_t *nt = (triplet_d_t *)realloc(t, (size_t)cap * sizeof(triplet_d_t));
            if (!nt) { free(t); fclose(f); return 0; }
            t = nt;
        }

        t[used++] = (triplet_d_t){ .i = i, .j = j, .v = v };

        if (symmetric && i != j) {
            double vv = v;
            if (symmetric == 2) vv = -vv;

            if (used >= cap) {
                cap *= 2;
                triplet_d_t *nt = (triplet_d_t *)realloc(t, (size_t)cap * sizeof(triplet_d_t));
                if (!nt) { free(t); fclose(f); return 0; }
                t = nt;
            }

            t[used++] = (triplet_d_t){ .i = j, .j = i, .v = vv };
        }
    }

    fclose(f);

    *out_t = t;
    *out_rows = n_rows;
    *out_cols = n_cols;
    *out_nnz = used;
    return 1;
}

