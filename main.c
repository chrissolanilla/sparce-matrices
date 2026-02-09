#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

/* chris note:
   - q1: hardcoded 6x6 sanity
   - q2/q3: ibm32 as dense (32x32 is fine)
   - q4: memplus must be sparse (no dense alloc) or you will swap/hang
*/

typedef struct {
    int n_rows;
    int n_cols;
    int nnz;
    int *values;
    int *col_idx;
    int *row_ptr;
} crs_t;

typedef struct {
    int n_rows;
    int n_cols;
    int nnz;
    int *values;
    int *row_idx;
    int *col_ptr;
} ccs_t;

/* jds for q1 only (small matrix) */
typedef struct {
    int n_rows;
    int n_cols;
    int nnz;
    int num_jd;
    int *jdiag;
    int *col_idx;
    int *perm;
    int *jdiag_ptr;
} jds_t;

/* tjds = jds built over A^T (so columns of A are "rows" here) */
typedef struct {
    int n_rows;
    int n_cols;
    int nnz;
    int num_tjd;
	/* values */
    int *tjd;
	/* original row i for each value */
    int *row_idx;
	/* permuted columns: tjds-col -> original col j */
    int *perm;
    int *tjd_ptr;
} tjds_t;

typedef struct {
    int i;
    int j;
    int v;
} triplet_t;

typedef struct {
    int i;
    int j;
    double v;
} triplet_d_t;

typedef struct {
    int n_rows;
    int n_cols;
    int nnz;
    double *values;
    int *col_idx;
    int *row_ptr;
} crs_d_t;

typedef struct {
    int n_rows;
    int n_cols;
    int nnz;
    double *values;
    int *row_idx;
    int *col_ptr;
} ccs_d_t;

typedef struct {
    int n_rows;
    int n_cols;
    int nnz;
    int num_tjd;
    double *tjd;
    int *row_idx;
    int *perm;
    int *tjd_ptr;
} tjds_d_t;


static void free_crs(crs_t *a) {
    if (!a) return;
    free(a->values);
    free(a->col_idx);
    free(a->row_ptr);
    a->values = NULL;
    a->col_idx = NULL;
    a->row_ptr = NULL;
    a->nnz = 0;
}

static void free_ccs(ccs_t *a) {
    if (!a) return;
    free(a->values);
    free(a->row_idx);
    free(a->col_ptr);
    a->values = NULL;
    a->row_idx = NULL;
    a->col_ptr = NULL;
    a->nnz = 0;
}

static void free_jds(jds_t *a) {
    if (!a) return;
    free(a->jdiag);
    free(a->col_idx);
    free(a->perm);
    free(a->jdiag_ptr);
    a->jdiag = NULL;
    a->col_idx = NULL;
    a->perm = NULL;
    a->jdiag_ptr = NULL;
    a->nnz = 0;
    a->num_jd = 0;
}

static void free_tjds(tjds_t *a) {
    if (!a) return;
    free(a->tjd);
    free(a->row_idx);
    free(a->perm);
    free(a->tjd_ptr);
    a->tjd = NULL;
    a->row_idx = NULL;
    a->perm = NULL;
    a->tjd_ptr = NULL;
    a->nnz = 0;
    a->num_tjd = 0;
}

static void free_crs_d(crs_d_t *a) {
    if (!a) return;
    free(a->values);
    free(a->col_idx);
    free(a->row_ptr);
    a->values = NULL;
    a->col_idx = NULL;
    a->row_ptr = NULL;
    a->nnz = 0;
}

static void free_ccs_d(ccs_d_t *a) {
    if (!a) return;
    free(a->values);
    free(a->row_idx);
    free(a->col_ptr);
    a->values = NULL;
    a->row_idx = NULL;
    a->col_ptr = NULL;
    a->nnz = 0;
}

static void free_tjds_d(tjds_d_t *a) {
    if (!a) return;
    free(a->tjd);
    free(a->row_idx);
    free(a->perm);
    free(a->tjd_ptr);
    a->tjd = NULL;
    a->row_idx = NULL;
    a->perm = NULL;
    a->tjd_ptr = NULL;
    a->nnz = 0;
    a->num_tjd = 0;
}


static void print_int_array(const char *name, const int *a, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n; i++) {
        printf("%d", a[i]);
        if (i + 1 < n) printf(", ");
    }
    printf("]\n");
}

static void print_vec(const char *name, const int *v, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n; i++) {
        printf("%d", v[i]);
        if (i + 1 < n) printf(", ");
    }
    printf("]\n");
}

static void print_matrix_6x6(int a[6][6]) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
}

static void flatten_6x6(const int a[6][6], int *out_flat) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            out_flat[i * 6 + j] = a[i][j];
        }
    }
}

static int count_nnz_dense(const int *dense, int n_rows, int n_cols) {
    int nnz = 0;
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            if (dense[i * n_cols + j] != 0) nnz++;
        }
    }
    return nnz;
}

/* dense MatrixMarket reader for ibm32 only */
static int mm_read_dense_int(const char *path, int **out_a, int *out_rows, int *out_cols, int *out_nnz) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    char line[512];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return 0;
    }

    if (line[0] != '%' || line[1] != '%') {
        fclose(f);
        return 0;
    }

    int is_pattern = 0;
    int symmetric = 0;
    if (strstr(line, "pattern")) is_pattern = 1;
    if (strstr(line, "skew-symmetric")) symmetric = 2;
    else if (strstr(line, "symmetric")) symmetric = 1;

    do {
        if (!fgets(line, sizeof(line), f)) {
            fclose(f);
            return 0;
        }
    } while (line[0] == '%');

    int n_rows = 0, n_cols = 0, nnz = 0;
    if (sscanf(line, "%d %d %d", &n_rows, &n_cols, &nnz) != 3) {
        fclose(f);
        return 0;
    }

    int *a = (int *)calloc((size_t)n_rows * (size_t)n_cols, sizeof(int));
    if (!a) {
        fclose(f);
        return 0;
    }

    for (int k = 0; k < nnz; k++) {
        int i = 0, j = 0;
        double v = 1.0;

        if (is_pattern) {
            if (fscanf(f, "%d %d", &i, &j) != 2) {
                free(a);
                fclose(f);
                return 0;
            }
            v = 1.0;
        } else {
            if (fscanf(f, "%d %d %lf", &i, &j, &v) != 3) {
                free(a);
                fclose(f);
                return 0;
            }
        }

        i--; j--;

        a[i * n_cols + j] = (int)v;

        if (symmetric == 1 && i != j) {
            a[j * n_cols + i] = (int)v;
        } else if (symmetric == 2 && i != j) {
            a[j * n_cols + i] = -(int)v;
        }
    }

    fclose(f);

    *out_a = a;
    *out_rows = n_rows;
    *out_cols = n_cols;
    *out_nnz = nnz;
    return 1;
}

/* sparse reader for big matrices for memplus, my other implementation wont work ngl */
static int mm_read_triplets_int(const char *path, triplet_t **out_t, int *out_rows, int *out_cols, int *out_nnz) {
    FILE *f = fopen(path, "r");
    if (!f) {
        return 0;
    }

    char line[512];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return 0;
    }

    if (line[0] != '%' || line[1] != '%') {
        fclose(f);
        return 0;
    }

    int is_pattern = 0;
    int symmetric = 0;
    if (strstr(line, "pattern")) is_pattern = 1;
    if (strstr(line, "skew-symmetric")) symmetric = 2;
    else if (strstr(line, "symmetric")) symmetric = 1;

    do {
        if (!fgets(line, sizeof(line), f)) {
            fclose(f);
            return 0;
        }
    } while (line[0] == '%');

    int n_rows = 0, n_cols = 0, nnz = 0;
    if (sscanf(line, "%d %d %d", &n_rows, &n_cols, &nnz) != 3) {
        fclose(f);
        return 0;
    }

    /*  if symmetric, we lowkeyh expand entries */
    int cap = (symmetric ? (2 * nnz) : nnz);
    triplet_t *t = (triplet_t *)malloc((size_t)cap * sizeof(triplet_t));
    if (!t) {
        fclose(f);
        return 0;
    }

    int used = 0;

    for (int k = 0; k < nnz; k++) {
        int i = 0, j = 0;
        double v = 1.0;

        if (is_pattern) {
            if (fscanf(f, "%d %d", &i, &j) != 2) {
                free(t);
                fclose(f);
                return 0;
            }
            v = 1.0;
        } else {
            if (fscanf(f, "%d %d %lf", &i, &j, &v) != 3) {
                free(t);
                fclose(f);
                return 0;
            }
        }

        i--; j--;

        if (used >= cap) {
            cap *= 2;
            triplet_t *nt = (triplet_t *)realloc(t, (size_t)cap * sizeof(triplet_t));
            if (!nt) {
                free(t);
                fclose(f);
                return 0;
            }
            t = nt;
        }

        t[used++] = (triplet_t){ .i = i, .j = j, .v = (int)v };

        if (symmetric && i != j) {
            int vv = (int)v;
			/* skew-symmetric mirrored sign flip */
            if (symmetric == 2) vv = -vv;
            if (used >= cap) {
                cap *= 2;
                triplet_t *nt = (triplet_t *)realloc(t, (size_t)cap * sizeof(triplet_t));
                if (!nt) {
                    free(t);
                    fclose(f);
                    return 0;
                }
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

static int mm_read_triplets_double(const char *path, triplet_d_t **out_t, int *out_rows, int *out_cols, int *out_nnz) {
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

    /*  if symmetric, we lowkeyh expand entries */
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

        if (i < 0 || i >= n_rows || j < 0 || j >= n_cols) {
            free(t); fclose(f); return 0;
        }

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


static crs_t build_crs_from_dense(const int *dense, int n_rows, int n_cols) {
    crs_t a;
    a.n_rows = n_rows;
    a.n_cols = n_cols;
    a.nnz = count_nnz_dense(dense, n_rows, n_cols);

    a.values = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.col_idx = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.row_ptr = (int *)malloc((size_t)(n_rows + 1) * sizeof(int));

    if (!a.values || !a.col_idx || !a.row_ptr) {
        fprintf(stderr, "crs malloc failed\n");
        free_crs(&a);
        return a;
    }

    int k = 0;
    a.row_ptr[0] = 0;

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            int v = dense[i * n_cols + j];
            if (v != 0) {
                a.values[k] = v;
                a.col_idx[k] = j;
                k++;
            }
        }
        a.row_ptr[i + 1] = k;
    }

    return a;
}

static ccs_t build_ccs_from_dense(const int *dense, int n_rows, int n_cols) {
    ccs_t a;
    a.n_rows = n_rows;
    a.n_cols = n_cols;
    a.nnz = count_nnz_dense(dense, n_rows, n_cols);

    a.values = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.row_idx = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.col_ptr = (int *)malloc((size_t)(n_cols + 1) * sizeof(int));

    if (!a.values || !a.row_idx || !a.col_ptr) {
        fprintf(stderr, "ccs malloc failed\n");
        free_ccs(&a);
        return a;
    }

    int k = 0;
    a.col_ptr[0] = 0;

    for (int j = 0; j < n_cols; j++) {
        for (int i = 0; i < n_rows; i++) {
            int v = dense[i * n_cols + j];
            if (v != 0) {
                a.values[k] = v;
                a.row_idx[k] = i;
                k++;
            }
        }
        a.col_ptr[j + 1] = k;
    }

    return a;
}

static jds_t build_jds_from_dense(const int *dense, int n_rows, int n_cols) {
    jds_t a;
    a.n_rows = n_rows;
    a.n_cols = n_cols;
    a.nnz = count_nnz_dense(dense, n_rows, n_cols);

    a.num_jd = 0;
    a.jdiag = NULL;
    a.col_idx = NULL;
    a.perm = NULL;
    a.jdiag_ptr = NULL;

    int *row_nnz = (int *)malloc((size_t)n_rows * sizeof(int));
    int *order = (int *)malloc((size_t)n_rows * sizeof(int));
    if (!row_nnz || !order) {
        fprintf(stderr, "jds malloc failed\n");
        free(row_nnz);
        free(order);
        return a;
    }

    for (int i = 0; i < n_rows; i++) {
        int c = 0;
        for (int j = 0; j < n_cols; j++) {
            if (dense[i * n_cols + j] != 0) c++;
        }
        row_nnz[i] = c;
        order[i] = i;
        if (c > a.num_jd) a.num_jd = c;
    }

    for (int i = 0; i < n_rows; i++) {
        for (int j = i + 1; j < n_rows; j++) {
            if (row_nnz[order[j]] > row_nnz[order[i]]) {
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }

    a.perm = (int *)malloc((size_t)n_rows * sizeof(int));
    a.jdiag = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.col_idx = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.jdiag_ptr = (int *)malloc((size_t)(a.num_jd + 1) * sizeof(int));
    if (!a.perm || !a.jdiag || !a.col_idx || !a.jdiag_ptr) {
        fprintf(stderr, "jds malloc failed\n");
        free(row_nnz);
        free(order);
        free_jds(&a);
        return a;
    }

    for (int r = 0; r < n_rows; r++) a.perm[r] = order[r];

    int *packed_val = (int *)calloc((size_t)n_rows * (size_t)a.num_jd, sizeof(int));
    int *packed_col = (int *)calloc((size_t)n_rows * (size_t)a.num_jd, sizeof(int));
    if (!packed_val || !packed_col) {
        fprintf(stderr, "jds packed malloc failed\n");
        free(row_nnz);
        free(order);
        free(packed_val);
        free(packed_col);
        free_jds(&a);
        return a;
    }

    for (int r = 0; r < n_rows; r++) {
        int orig_row = a.perm[r];
        int p = 0;
        for (int j = 0; j < n_cols; j++) {
            int v = dense[orig_row * n_cols + j];
            if (v != 0) {
                packed_val[r * a.num_jd + p] = v;
                packed_col[r * a.num_jd + p] = j;
                p++;
            }
        }
    }

    int k = 0;
    a.jdiag_ptr[0] = 0;
    for (int d = 0; d < a.num_jd; d++) {
        for (int r = 0; r < n_rows; r++) {
            int orig_row = a.perm[r];
            if (d < row_nnz[orig_row]) {
                a.jdiag[k] = packed_val[r * a.num_jd + d];
                a.col_idx[k] = packed_col[r * a.num_jd + d];
                k++;
            }
        }
        a.jdiag_ptr[d + 1] = k;
    }

    free(row_nnz);
    free(order);
    free(packed_val);
    free(packed_col);

    return a;
}

static tjds_t build_tjds_from_dense(const int *dense, int n_rows, int n_cols) {
    tjds_t a;
    a.n_rows = n_rows;
    a.n_cols = n_cols;
    a.nnz = count_nnz_dense(dense, n_rows, n_cols);

    a.num_tjd = 0;
    a.tjd = NULL;
    a.row_idx = NULL;
    a.perm = NULL;
    a.tjd_ptr = NULL;

    int *col_nnz = (int *)malloc((size_t)n_cols * sizeof(int));
    int *order = (int *)malloc((size_t)n_cols * sizeof(int));
    if (!col_nnz || !order) {
        fprintf(stderr, "tjds malloc failed\n");
        free(col_nnz);
        free(order);
        return a;
    }

    for (int j = 0; j < n_cols; j++) {
        int c = 0;
        for (int i = 0; i < n_rows; i++) {
            if (dense[i * n_cols + j] != 0) c++;
        }
        col_nnz[j] = c;
        order[j] = j;
        if (c > a.num_tjd) a.num_tjd = c;
    }

    for (int i = 0; i < n_cols; i++) {
        for (int j = i + 1; j < n_cols; j++) {
            if (col_nnz[order[j]] > col_nnz[order[i]]) {
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }

    a.perm = (int *)malloc((size_t)n_cols * sizeof(int));
    a.tjd = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.row_idx = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.tjd_ptr = (int *)malloc((size_t)(a.num_tjd + 1) * sizeof(int));
    if (!a.perm || !a.tjd || !a.row_idx || !a.tjd_ptr) {
        fprintf(stderr, "tjds malloc failed\n");
        free(col_nnz);
        free(order);
        free_tjds(&a);
        return a;
    }

    for (int c = 0; c < n_cols; c++) a.perm[c] = order[c];

    int *packed_val = (int *)calloc((size_t)n_cols * (size_t)a.num_tjd, sizeof(int));
    int *packed_row = (int *)calloc((size_t)n_cols * (size_t)a.num_tjd, sizeof(int));
    if (!packed_val || !packed_row) {
        fprintf(stderr, "tjds packed malloc failed\n");
        free(col_nnz);
        free(order);
        free(packed_val);
        free(packed_row);
        free_tjds(&a);
        return a;
    }

    for (int c = 0; c < n_cols; c++) {
        int orig_col = a.perm[c];
        int p = 0;
        for (int i = 0; i < n_rows; i++) {
            int v = dense[i * n_cols + orig_col];
            if (v != 0) {
                packed_val[c * a.num_tjd + p] = v;
                packed_row[c * a.num_tjd + p] = i;
                p++;
            }
        }
    }

    int k = 0;
    a.tjd_ptr[0] = 0;
    for (int d = 0; d < a.num_tjd; d++) {
        for (int c = 0; c < n_cols; c++) {
            int orig_col = a.perm[c];
            if (d < col_nnz[orig_col]) {
                a.tjd[k] = packed_val[c * a.num_tjd + d];
                a.row_idx[k] = packed_row[c * a.num_tjd + d];
                k++;
            }
        }
        a.tjd_ptr[d + 1] = k;
    }

    free(col_nnz);
    free(order);
    free(packed_val);
    free(packed_row);

    return a;
}

/* builds CRS directly from triplets  no dense */
static int build_crs_from_triplets(int n_rows, int n_cols, const triplet_t *t, int nnz, crs_t *out) {
    crs_t a;
    a.n_rows = n_rows;
    a.n_cols = n_cols;
    a.nnz = nnz;
    a.values = (int *)malloc((size_t)nnz * sizeof(int));
    a.col_idx = (int *)malloc((size_t)nnz * sizeof(int));
    a.row_ptr = (int *)calloc((size_t)n_rows + 1, sizeof(int));
    if (!a.values || !a.col_idx || !a.row_ptr) {
        free_crs(&a);
        return 0;
    }

    for (int k = 0; k < nnz; k++) {
        int i = t[k].i;
        if (i < 0 || i >= n_rows) {
            free_crs(&a);
            return 0;
        }
        a.row_ptr[i + 1]++;
    }
    /* lowkey guha the goat for showing me the prefix sum */
    for (int i = 0; i < n_rows; i++) {
        a.row_ptr[i + 1] += a.row_ptr[i];
    }

    int *next = (int *)malloc((size_t)n_rows * sizeof(int));
    if (!next) {
        free_crs(&a);
        return 0;
    }
    for (int i = 0; i < n_rows; i++)
		next[i] = a.row_ptr[i];

    for (int k = 0; k < nnz; k++) {
        int i = t[k].i;
        int j = t[k].j;
        int pos = next[i]++;
        a.values[pos] = t[k].v;
        a.col_idx[pos] = j;
    }

    free(next);
    *out = a;
    return 1;
}

/* builds CCS directly from triplets (this is basically CSC) */
static int build_ccs_from_triplets(int n_rows, int n_cols, const triplet_t *t, int nnz, ccs_t *out) {
    ccs_t a;
    a.n_rows = n_rows;
    a.n_cols = n_cols;
    a.nnz = nnz;
    a.values = (int *)malloc((size_t)nnz * sizeof(int));
    a.row_idx = (int *)malloc((size_t)nnz * sizeof(int));
    a.col_ptr = (int *)calloc((size_t)n_cols + 1, sizeof(int));
    if (!a.values || !a.row_idx || !a.col_ptr) {
        free_ccs(&a);
        return 0;
    }

    for (int k = 0; k < nnz; k++) {
        int j = t[k].j;
        if (j < 0 || j >= n_cols) {
            free_ccs(&a);
            return 0;
        }
        a.col_ptr[j + 1]++;
    }

    for (int j = 0; j < n_cols; j++) {
        a.col_ptr[j + 1] += a.col_ptr[j];
    }

    int *next = (int *)malloc((size_t)n_cols * sizeof(int));
    if (!next) {
        free_ccs(&a);
        return 0;
    }
    for (int j = 0; j < n_cols; j++) next[j] = a.col_ptr[j];

    for (int k = 0; k < nnz; k++) {
        int i = t[k].i;
        int j = t[k].j;
        int pos = next[j]++;
        a.values[pos] = t[k].v;
        a.row_idx[pos] = i;
    }

    free(next);
    *out = a;
    return 1;
}

static int build_crs_from_triplets_double(int n_rows, int n_cols,
                                         const triplet_d_t *t, int nnz, crs_d_t *out) {
    crs_d_t a;
    a.n_rows = n_rows;
    a.n_cols = n_cols;
    a.nnz = nnz;

    a.values = (double *)malloc((size_t)nnz * sizeof(double));
    a.col_idx = (int *)malloc((size_t)nnz * sizeof(int));
    a.row_ptr = (int *)calloc((size_t)n_rows + 1, sizeof(int));
    if (!a.values || !a.col_idx || !a.row_ptr) { free_crs_d(&a); return 0; }

    for (int k = 0; k < nnz; k++) {
        int i = t[k].i;
        if (i < 0 || i >= n_rows) { free_crs_d(&a); return 0; }
        a.row_ptr[i + 1]++;
    }

    for (int i = 0; i < n_rows; i++) a.row_ptr[i + 1] += a.row_ptr[i];

    int *next = (int *)malloc((size_t)n_rows * sizeof(int));
    if (!next) { free_crs_d(&a); return 0; }
    for (int i = 0; i < n_rows; i++) next[i] = a.row_ptr[i];

    for (int k = 0; k < nnz; k++) {
        int i = t[k].i;
        int j = t[k].j;
        int pos = next[i]++;
        a.values[pos] = t[k].v;
        a.col_idx[pos] = j;
    }

    free(next);
    *out = a;
    return 1;
}

static int build_ccs_from_triplets_double(int n_rows, int n_cols,
                                         const triplet_d_t *t, int nnz, ccs_d_t *out) {
    ccs_d_t a;
    a.n_rows = n_rows;
    a.n_cols = n_cols;
    a.nnz = nnz;

    a.values = (double *)malloc((size_t)nnz * sizeof(double));
    a.row_idx = (int *)malloc((size_t)nnz * sizeof(int));
    a.col_ptr = (int *)calloc((size_t)n_cols + 1, sizeof(int));
    if (!a.values || !a.row_idx || !a.col_ptr) { free_ccs_d(&a); return 0; }

    for (int k = 0; k < nnz; k++) {
        int j = t[k].j;
        if (j < 0 || j >= n_cols) { free_ccs_d(&a); return 0; }
        a.col_ptr[j + 1]++;
    }

    for (int j = 0; j < n_cols; j++) a.col_ptr[j + 1] += a.col_ptr[j];

    int *next = (int *)malloc((size_t)n_cols * sizeof(int));
    if (!next) { free_ccs_d(&a); return 0; }
    for (int j = 0; j < n_cols; j++) next[j] = a.col_ptr[j];

    for (int k = 0; k < nnz; k++) {
        int i = t[k].i;
        int j = t[k].j;
        int pos = next[j]++;
        a.values[pos] = t[k].v;
        a.row_idx[pos] = i;
    }

    free(next);
    *out = a;
    return 1;
}


/* builds TJDS from CCSfor memplus because we already have columns */
static tjds_t build_tjds_from_ccs(const ccs_t *c) {
    tjds_t a;
    a.n_rows = c->n_rows;
    a.n_cols = c->n_cols;
    a.nnz = c->nnz;

    a.num_tjd = 0;
    a.tjd = NULL;
    a.row_idx = NULL;
    a.perm = NULL;
    a.tjd_ptr = NULL;

    int n_cols = c->n_cols;

    int *col_nnz = (int *)malloc((size_t)n_cols * sizeof(int));
    int *order = (int *)malloc((size_t)n_cols * sizeof(int));
    if (!col_nnz || !order) {
        free(col_nnz);
        free(order);
        return a;
    }

    for (int j = 0; j < n_cols; j++) {
        int count = c->col_ptr[j + 1] - c->col_ptr[j];
        col_nnz[j] = count;
        order[j] = j;
        if (count > a.num_tjd) a.num_tjd = count;
    }

    for (int i = 0; i < n_cols; i++) {
        for (int j = i + 1; j < n_cols; j++) {
            if (col_nnz[order[j]] > col_nnz[order[i]]) {
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }

    a.perm = (int *)malloc((size_t)n_cols * sizeof(int));
    a.tjd = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.row_idx = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.tjd_ptr = (int *)malloc((size_t)(a.num_tjd + 1) * sizeof(int));
    if (!a.perm || !a.tjd || !a.row_idx || !a.tjd_ptr) {
        free(col_nnz);
        free(order);
        free_tjds(&a);
        return a;
    }

    for (int cidx = 0; cidx < n_cols; cidx++) a.perm[cidx] = order[cidx];

    int k = 0;
    a.tjd_ptr[0] = 0;

    for (int d = 0; d < a.num_tjd; d++) {
        for (int cidx = 0; cidx < n_cols; cidx++) {
            int orig_col = a.perm[cidx];
            if (d < col_nnz[orig_col]) {
                int base = c->col_ptr[orig_col];
                a.tjd[k] = c->values[base + d];
                a.row_idx[k] = c->row_idx[base + d];
                k++;
            }
        }
        a.tjd_ptr[d + 1] = k;
    }

    free(col_nnz);
    free(order);

    return a;
}

static tjds_d_t build_tjds_from_ccs_double(const ccs_d_t *c) {
    tjds_d_t a;
    a.n_rows = c->n_rows;
    a.n_cols = c->n_cols;
    a.nnz = c->nnz;

    a.num_tjd = 0;
    a.tjd = NULL;
    a.row_idx = NULL;
    a.perm = NULL;
    a.tjd_ptr = NULL;

    int n_cols = c->n_cols;

    int *col_nnz = (int *)malloc((size_t)n_cols * sizeof(int));
    int *order = (int *)malloc((size_t)n_cols * sizeof(int));
    if (!col_nnz || !order) { free(col_nnz); free(order); return a; }

    for (int j = 0; j < n_cols; j++) {
        int count = c->col_ptr[j + 1] - c->col_ptr[j];
        col_nnz[j] = count;
        order[j] = j;
        if (count > a.num_tjd) a.num_tjd = count;
    }

    for (int i = 0; i < n_cols; i++) {
        for (int j = i + 1; j < n_cols; j++) {
            if (col_nnz[order[j]] > col_nnz[order[i]]) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
        }
    }

    a.perm = (int *)malloc((size_t)n_cols * sizeof(int));
    a.tjd = (double *)malloc((size_t)a.nnz * sizeof(double));
    a.row_idx = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.tjd_ptr = (int *)malloc((size_t)(a.num_tjd + 1) * sizeof(int));
    if (!a.perm || !a.tjd || !a.row_idx || !a.tjd_ptr) {
        free(col_nnz); free(order);
        free_tjds_d(&a);
        return a;
    }

    for (int cidx = 0; cidx < n_cols; cidx++) a.perm[cidx] = order[cidx];

    int k = 0;
    a.tjd_ptr[0] = 0;

    for (int d = 0; d < a.num_tjd; d++) {
        for (int cidx = 0; cidx < n_cols; cidx++) {
            int orig_col = a.perm[cidx];
            if (d < col_nnz[orig_col]) {
                int base = c->col_ptr[orig_col];
                a.tjd[k] = c->values[base + d];
                a.row_idx[k] = c->row_idx[base + d];
                k++;
            }
        }
        a.tjd_ptr[d + 1] = k;
    }

    free(col_nnz);
    free(order);
    return a;
}

static void crs_spmv_double(const crs_d_t *a, const double *x, double *y) {
    for (int i = 0; i < a->n_rows; i++) {
        double sum = 0.0;
        for (int k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++) {
            sum += a->values[k] * x[a->col_idx[k]];
        }
        y[i] = sum;
    }
}

static void tjds_spmv_double(const tjds_d_t *a, const double *x, double *y) {
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


static void dense_spmv(const int *a, int n_rows, int n_cols, const int *x, int *y) {
    for (int i = 0; i < n_rows; i++) {
        int sum = 0;
        for (int j = 0; j < n_cols; j++) {
            sum += a[i * n_cols + j] * x[j];
        }
        y[i] = sum;
    }
}

static void crs_spmv(const crs_t *a, const int *x, int *y) {
    for (int i = 0; i < a->n_rows; i++) {
        int sum = 0;
        for (int k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++) {
            sum += a->values[k] * x[a->col_idx[k]];
        }
        y[i] = sum;
    }
}

static void ccs_spmv(const ccs_t *a, const int *x, int *y) {
    for (int i = 0; i < a->n_rows; i++) y[i] = 0;

    for (int j = 0; j < a->n_cols; j++) {
        for (int k = a->col_ptr[j]; k < a->col_ptr[j + 1]; k++) {
            int i = a->row_idx[k];
            y[i] += a->values[k] * x[j];
        }
    }
}

static void jds_spmv(const jds_t *a, const int *x, int *y) {
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

static void tjds_spmv(const tjds_t *a, const int *x, int *y) {
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

static void print_crs_hw(const crs_t *a) {
    printf("CRS 0-based\n");
    printf("nRows = %d\n", a->n_rows);
    printf("nCols = %d\n", a->n_cols);
    printf("nnz = %d\n", a->nnz);
    print_int_array("values", a->values, a->nnz);
    print_int_array("column_index", a->col_idx, a->nnz);
    print_int_array("row_pointers", a->row_ptr, a->n_rows + 1);
}

static void print_ccs_hw(const ccs_t *a) {
    printf("CcS 0-based\n");
    printf("nRows = %d\n", a->n_rows);
    printf("nCols = %d\n", a->n_cols);
    printf("nnz = %d\n", a->nnz);
    print_int_array("values", a->values, a->nnz);
    print_int_array("row_index", a->row_idx, a->nnz);
    print_int_array("column_pointers", a->col_ptr, a->n_cols + 1);
}

static void print_jds_hw(const jds_t *a) {
    printf("JDS 0-based\n");
    printf("nRows = %d\n", a->n_rows);
    printf("nCols = %d\n", a->n_cols);
    printf("nnz = %d\n", a->nnz);
    printf("numJd = %d\n", a->num_jd);
    print_int_array("jdiag", a->jdiag, a->nnz);
    print_int_array("column_index", a->col_idx, a->nnz);
    print_int_array("perm", a->perm, a->n_rows);
    print_int_array("jdiag_ptr", a->jdiag_ptr, a->num_jd + 1);
}

static void print_tjds_hw(const tjds_t *a) {
    printf("TJDS 0-based\n");
    printf("nRows = %d\n", a->n_rows);
    printf("nCols = %d\n", a->n_cols);
    printf("nnz = %d\n", a->nnz);
    printf("numTjd = %d\n", a->num_tjd);
    print_int_array("tjd", a->tjd, a->nnz);
    print_int_array("row_index", a->row_idx, a->nnz);
    print_int_array("perm", a->perm, a->n_cols);
    print_int_array("tjd_ptr", a->tjd_ptr, a->num_tjd + 1);
}

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void bench_dense(const int *a, int n_rows, int n_cols, const int *x, int *y, int iters) {
    long long t0 = now_ns();
    long long check = 0;
    for (int k = 0; k < iters; k++) {
        dense_spmv(a, n_rows, n_cols, x, y);
        check += y[0];
    }
    long long t1 = now_ns();
    printf("dense iters = %d time_ns = %lld check = %lld\n", iters, (t1 - t0), check);
}

static void bench_crs(const crs_t *a, const int *x, int *y, int iters) {
    long long t0 = now_ns();
    long long check = 0;
    for (int k = 0; k < iters; k++) {
        crs_spmv(a, x, y);
        check += y[0];
    }
    long long t1 = now_ns();
    printf("crs iters = %d time_ns = %lld check = %lld\n", iters, (t1 - t0), check);
}

static void bench_tjds(const tjds_t *a, const int *x, int *y, int iters) {
    long long t0 = now_ns();
    long long check = 0;
    for (int k = 0; k < iters; k++) {
        tjds_spmv(a, x, y);
        check += y[0];
    }
    long long t1 = now_ns();
    printf("tjds iters = %d time_ns = %lld check = %lld\n", iters, (t1 - t0), check);
}

static long long checksum_vec(const int *y, int n) {
    long long s = 0;
    for (int i = 0; i < n; i++) {
        s += (long long)y[i] * (long long)(i + 1);
    }
    return s;
}

static void demo_q1(void) {
    int a6[6][6] = {
        {10,0,0,0,-2,0},
        {3,9,0,0,0,0},
        {0,0,8,7,3,0},
        {0,0,8,7,0,0},
        {0,8,0,9,9,0},
        {0,5,0,0,2,-1}
    };

    printf("=== Q1 sanity matrix A 6x6 ===\n");
    print_matrix_6x6(a6);
    printf("\n");

    int flat[36];
    flatten_6x6(a6, flat);

    crs_t crs = build_crs_from_dense(flat, 6, 6);
    ccs_t ccs = build_ccs_from_dense(flat, 6, 6);
    jds_t jds = build_jds_from_dense(flat, 6, 6);
    tjds_t tjds = build_tjds_from_dense(flat, 6, 6);

    print_crs_hw(&crs);
    printf("\n");
    print_ccs_hw(&ccs);
    printf("\n");
    print_jds_hw(&jds);
    printf("\n");
    print_tjds_hw(&tjds);
    printf("\n");

    int x[6] = {1,1,1,1,1,1};
    int y_dense[6];
    int y_crs[6];
    int y_ccs[6];
    int y_jds[6];
    int y_tjds[6];

    dense_spmv(flat, 6, 6, x, y_dense);
    crs_spmv(&crs, x, y_crs);
    ccs_spmv(&ccs, x, y_ccs);
    jds_spmv(&jds, x, y_jds);
    tjds_spmv(&tjds, x, y_tjds);

    print_vec("x", x, 6);
    print_vec("y dense", y_dense, 6);
    print_vec("y crs", y_crs, 6);
    print_vec("y ccs", y_ccs, 6);
    print_vec("y jds", y_jds, 6);
    print_vec("y tjds", y_tjds, 6);
    printf("\n");

    for (int i = 0; i < 6; i++) {
        if (y_dense[i] != y_crs[i] || y_dense[i] != y_ccs[i] ||
            y_dense[i] != y_jds[i] || y_dense[i] != y_tjds[i]) {
            printf("bruh mismatch at i=%d\n\n", i);
            break;
        }
    }

    free_crs(&crs);
    free_ccs(&ccs);
    free_jds(&jds);
    free_tjds(&tjds);
}

static void run_ibm32_dense(const char *mtx_path) {
    int *a = NULL;
    int n_rows = 0, n_cols = 0, nnz = 0;

    if (!mm_read_dense_int(mtx_path, &a, &n_rows, &n_cols, &nnz)) {
        printf("=== Q2/Q3 IBM32 ===\n");
        printf("couldn't open %s (skipping)\n\n", mtx_path);
        return;
    }

    printf("=== Q2/Q3 IBM32 loaded ===\n");
    printf("file = %s\n", mtx_path);
    printf("nRows = %d\n", n_rows);
    printf("nCols = %d\n", n_cols);
    printf("nnz  = %d\n\n", nnz);

    int *x = (int *)malloc((size_t)n_cols * sizeof(int));
    int *y = (int *)malloc((size_t)n_rows * sizeof(int));
    if (!x || !y) {
        fprintf(stderr, "malloc failed\n");
        free(a);
        free(x);
        free(y);
        return;
    }
    for (int i = 0; i < n_cols; i++) x[i] = 1;

    bench_dense(a, n_rows, n_cols, x, y, 1000);
    bench_dense(a, n_rows, n_cols, x, y, 10000);

    crs_t crs = build_crs_from_dense(a, n_rows, n_cols);
    bench_crs(&crs, x, y, 1000);
    bench_crs(&crs, x, y, 10000);

    tjds_t tjds = build_tjds_from_dense(a, n_rows, n_cols);
    bench_tjds(&tjds, x, y, 1000);
    bench_tjds(&tjds, x, y, 10000);

    tjds_spmv(&tjds, x, y);
    printf("\nquick y peek: y[0]=%d", y[0]);
    if (n_rows > 1) printf(" y[1]=%d", y[1]);
    if (n_rows > 2) printf(" y[2]=%d", y[2]);
    printf("\n\n");

    free_crs(&crs);
    free_tjds(&tjds);
    free(a);
    free(x);
    free(y);
}

/* static void run_memplus_sparse(const char *mtx_path) { */
/*     triplet_t *t = NULL; */
/*     int n_rows = 0, n_cols = 0, nnz = 0; */
/**/
/*     if (!mm_read_triplets_int(mtx_path, &t, &n_rows, &n_cols, &nnz)) { */
/*         printf("=== Q4 MEMPLUS ===\n"); */
/*         printf("couldn't open %s (skipping)\n\n", mtx_path); */
/*         return; */
/*     } */
/**/
/*     printf("=== Q4 MEMPLUS loaded ===\n"); */
/*     printf("file = %s\n", mtx_path); */
/*     printf("nRows = %d\n", n_rows); */
/*     printf("nCols = %d\n", n_cols); */
/*     printf("nnz  = %d\n\n", nnz); */
/**/
/*     crs_t crs; */
/*     if (!build_crs_from_triplets(n_rows, n_cols, t, nnz, &crs)) { */
/*         fprintf(stderr, "failed building crs from triplets\n"); */
/*         free(t); */
/*         return; */
/*     } */
/**/
/*     ccs_t ccs; */
/*     if (!build_ccs_from_triplets(n_rows, n_cols, t, nnz, &ccs)) { */
/*         fprintf(stderr, "failed building ccs from triplets\n"); */
/*         free_crs(&crs); */
/*         free(t); */
/*         return; */
/*     } */
/**/
/*     tjds_t tjds = build_tjds_from_ccs(&ccs); */
/**/
/*     int *x = (int *)malloc((size_t)n_cols * sizeof(int)); */
/*     int *y_crs = (int *)malloc((size_t)n_rows * sizeof(int)); */
/*     int *y_tjds = (int *)malloc((size_t)n_rows * sizeof(int)); */
/*     if (!x || !y_crs || !y_tjds) { */
/*         fprintf(stderr, "malloc failed\n"); */
/*         free(x); */
/*         free(y_crs); */
/*         free(y_tjds); */
/*         free_tjds(&tjds); */
/*         free_ccs(&ccs); */
/*         free_crs(&crs); */
/*         free(t); */
/*         return; */
/*     } */
/**/
/*     for (int i = 0; i < n_cols; i++) x[i] = 1; */
/**/
/*     bench_crs(&crs, x, y_crs, 1000); */
/*     bench_crs(&crs, x, y_crs, 10000); */
/**/
/*     bench_tjds(&tjds, x, y_tjds, 1000); */
/*     bench_tjds(&tjds, x, y_tjds, 10000); */
/**/
/*     /* compare checksums + a few entries  */
/*     crs_spmv(&crs, x, y_crs); */
/*     tjds_spmv(&tjds, x, y_tjds); */
/**/
/*     long long s1 = checksum_vec(y_crs, n_rows); */
/*     long long s2 = checksum_vec(y_tjds, n_rows); */
/**/
/*     printf("\nverify:\n"); */
/*     printf("  checksum crs  = %lld\n", s1); */
/*     printf("  checksum tjds = %lld\n", s2); */
/*     printf("  y[0]=%d vs %d\n", y_crs[0], y_tjds[0]); */
/*     if (n_rows > 1) printf("  y[1]=%d vs %d\n", y_crs[1], y_tjds[1]); */
/*     if (n_rows > 2) printf("  y[2]=%d vs %d\n", y_crs[2], y_tjds[2]); */
/**/
/*     if (s1 != s2) { */
/*         printf("  bruh: checksum mismatch (tjds bug)\n"); */
/*     } else { */
/*         printf("  ok: checksums match\n"); */
/*     } */
/*     printf("\n"); */
/**/
/*     free(x); */
/*     free(y_crs); */
/*     free(y_tjds); */
/*     free_tjds(&tjds); */
/*     free_ccs(&ccs); */
/*     free_crs(&crs); */
/*     free(t); */
/* } */

static double checksum_vec_double(const double *y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += y[i] * (double)(i + 1);
    return s;
}

static void bench_crs_double(const crs_d_t *a, const double *x, double *y, int iters) {
    long long t0 = now_ns();
    double check = 0.0;
    for (int k = 0; k < iters; k++) {
        crs_spmv_double(a, x, y);
        check += y[k % a->n_rows]; /* avoids “y[0] is 0” useless check */
    }
    long long t1 = now_ns();
    printf("crs iters = %d time_ns = %lld check = %.6g\n", iters, (t1 - t0), check);
}

static void bench_tjds_double(const tjds_d_t *a, const double *x, double *y, int iters) {
    long long t0 = now_ns();
    double check = 0.0;
    for (int k = 0; k < iters; k++) {
        tjds_spmv_double(a, x, y);
        check += y[k % a->n_rows];
    }
    long long t1 = now_ns();
    printf("tjds iters = %d time_ns = %lld check = %.6g\n", iters, (t1 - t0), check);
}

static void run_memplus_sparse(const char *mtx_path) {
    triplet_d_t *t = NULL;
    int n_rows = 0, n_cols = 0, nnz = 0;

    if (!mm_read_triplets_double(mtx_path, &t, &n_rows, &n_cols, &nnz)) {
        printf("=== Q4 MEMPLUS ===\n");
        printf("couldn't open %s (skipping)\n\n", mtx_path);
        return;
    }

    printf("=== Q4 MEMPLUS loaded (double) ===\n");
    printf("file = %s\n", mtx_path);
    printf("nRows = %d\n", n_rows);
    printf("nCols = %d\n", n_cols);
    printf("nnz  = %d\n\n", nnz);

    crs_d_t crs;
    if (!build_crs_from_triplets_double(n_rows, n_cols, t, nnz, &crs)) {
        fprintf(stderr, "failed building crs(double)\n");
        free(t);
        return;
    }

    ccs_d_t ccs;
    if (!build_ccs_from_triplets_double(n_rows, n_cols, t, nnz, &ccs)) {
        fprintf(stderr, "failed building ccs(double)\n");
        free_crs_d(&crs);
        free(t);
        return;
    }

    tjds_d_t tjds = build_tjds_from_ccs_double(&ccs);

    double *x = (double *)malloc((size_t)n_cols * sizeof(double));
    double *y_crs = (double *)malloc((size_t)n_rows * sizeof(double));
    double *y_tjds = (double *)malloc((size_t)n_rows * sizeof(double));
    if (!x || !y_crs || !y_tjds) {
        fprintf(stderr, "malloc failed\n");
        free(x); free(y_crs); free(y_tjds);
        free_tjds_d(&tjds);
        free_ccs_d(&ccs);
        free_crs_d(&crs);
        free(t);
        return;
    }

    for (int i = 0; i < n_cols; i++) x[i] = 1.0;

    bench_crs_double(&crs, x, y_crs, 1000);
    bench_crs_double(&crs, x, y_crs, 10000);

    bench_tjds_double(&tjds, x, y_tjds, 1000);
    bench_tjds_double(&tjds, x, y_tjds, 10000);

    /* verify */
    crs_spmv_double(&crs, x, y_crs);
    tjds_spmv_double(&tjds, x, y_tjds);

    double s1 = checksum_vec_double(y_crs, n_rows);
    double s2 = checksum_vec_double(y_tjds, n_rows);

    printf("\nverify:\n");
    printf("  checksum crs  = %.12g\n", s1);
    printf("  checksum tjds = %.12g\n", s2);
    printf("  y[0]=%.6g vs %.6g\n", y_crs[0], y_tjds[0]);
    if (n_rows > 1) printf("  y[1]=%.6g vs %.6g\n", y_crs[1], y_tjds[1]);
    if (n_rows > 2) printf("  y[2]=%.6g vs %.6g\n", y_crs[2], y_tjds[2]);

    double diff = s1 - s2;
    if (diff < 0) diff = -diff;
    if (diff > 1e-8) printf("  bruh: checksum mismatch (tjds bug)\n");
    else printf("  ok: checksums match\n");
    printf("\n");

    free(x);
    free(y_crs);
    free(y_tjds);
    free_tjds_d(&tjds);
    free_ccs_d(&ccs);
    free_crs_d(&crs);
    free(t);
}


int main(void) {
    demo_q1();
    run_ibm32_dense("ibm32.mtx");
    run_memplus_sparse("memplus.mtx");
    return 0;
}


