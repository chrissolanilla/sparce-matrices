#include <stdio.h>
#include <stdlib.h>

#include "formats.h"
#include "util.h"

//for quick sort
typedef struct {
	int idx; int nnz;
} nnz_pair_t;

static int cmp_nnz_desc(const void *a, const void *b) {
    const nnz_pair_t *pa = (const nnz_pair_t *)a;
    const nnz_pair_t *pb = (const nnz_pair_t *)b;
    if (pa->nnz != pb->nnz)
		return (pb->nnz - pa->nnz);
    return (pa->idx - pb->idx);
}

void free_crs(crs_t *a) {
    if (!a) return;
    free(a->values);
    free(a->col_idx);
    free(a->row_ptr);
    a->values = NULL;
    a->col_idx = NULL;
    a->row_ptr = NULL;
    a->nnz = 0;
}

void free_ccs(ccs_t *a) {
    if (!a) return;
    free(a->values);
    free(a->row_idx);
    free(a->col_ptr);
    a->values = NULL;
    a->row_idx = NULL;
    a->col_ptr = NULL;
    a->nnz = 0;
}

void free_jds(jds_t *a) {
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

void free_tjds(tjds_t *a) {
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

void free_crs_d(crs_d_t *a) {
    if (!a) return;
    free(a->values);
    free(a->col_idx);
    free(a->row_ptr);
    a->values = NULL;
    a->col_idx = NULL;
    a->row_ptr = NULL;
    a->nnz = 0;
}

void free_ccs_d(ccs_d_t *a) {
    if (!a) return;
    free(a->values);
    free(a->row_idx);
    free(a->col_ptr);
    a->values = NULL;
    a->row_idx = NULL;
    a->col_ptr = NULL;
    a->nnz = 0;
}

void free_tjds_d(tjds_d_t *a) {
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

crs_t build_crs_from_dense(const int *dense, int n_rows, int n_cols) {
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

ccs_t build_ccs_from_dense(const int *dense, int n_rows, int n_cols) {
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

jds_t build_jds_from_dense(const int *dense, int n_rows, int n_cols) {
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
        for (int j = 0; j < n_cols; j++)
			if (dense[i * n_cols + j] != 0)
				c++;
        row_nnz[i] = c;
        order[i] = i;
        if (c > a.num_jd) a.num_jd = c;
    }

	nnz_pair_t *pairs = malloc((size_t)n_rows * sizeof(nnz_pair_t));
	if (!pairs) {
		fprintf(stderr, "jds malloc failed\n");
		free(row_nnz);
		free(order);
		return a;
	}

	for (int i = 0; i < n_rows; i++) {
		pairs[i].idx = i;
		pairs[i].nnz = row_nnz[i];
	}

	qsort(pairs, (size_t)n_rows, sizeof(nnz_pair_t), cmp_nnz_desc);

	for (int i = 0; i < n_rows; i++)
		order[i] = pairs[i].idx;

	free(pairs);

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

    for (int r = 0; r < n_rows; r++)
		a.perm[r] = order[r];

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

tjds_t build_tjds_from_dense(const int *dense, int n_rows, int n_cols) {
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
        for (int i = 0; i < n_rows; i++)
			if (dense[i * n_cols + j] != 0)
				c++;
        col_nnz[j] = c;
        order[j] = j;
        if (c > a.num_tjd) a.num_tjd = c;
    }

	nnz_pair_t *pairs = malloc((size_t)n_cols * sizeof(nnz_pair_t));
	if (!pairs) {
		fprintf(stderr, "tjds malloc failed\n");
		free(col_nnz);
		free(order);
		return a;
	}

	for (int j = 0; j < n_cols; j++) {
		pairs[j].idx = j;
		pairs[j].nnz = col_nnz[j];
	}

	qsort(pairs, (size_t)n_cols, sizeof(nnz_pair_t), cmp_nnz_desc);

	for (int j = 0; j < n_cols; j++)
		order[j] = pairs[j].idx;

	free(pairs);

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

    for (int c = 0; c < n_cols; c++)
		a.perm[c] = order[c];

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

int build_crs_from_triplets(int n_rows, int n_cols, const triplet_t *t, int nnz, crs_t *out) {
    crs_t a;
    a.n_rows = n_rows;
    a.n_cols = n_cols;
    a.nnz = nnz;

    a.values = (int *)malloc((size_t)nnz * sizeof(int));
    a.col_idx = (int *)malloc((size_t)nnz * sizeof(int));
    a.row_ptr = (int *)calloc((size_t)n_rows + 1, sizeof(int));
    if (!a.values || !a.col_idx || !a.row_ptr){
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
    for (int i = 0; i < n_rows; i++)
		a.row_ptr[i + 1] += a.row_ptr[i];

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

int build_ccs_from_triplets(int n_rows, int n_cols, const triplet_t *t, int nnz, ccs_t *out) {
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
    for (int j = 0; j < n_cols; j++)
		a.col_ptr[j + 1] += a.col_ptr[j];

    int *next = (int *)malloc((size_t)n_cols * sizeof(int));
    if (!next) { free_ccs(&a);
		return 0;
	}

    for (int j = 0; j < n_cols; j++)
		next[j] = a.col_ptr[j];

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

int build_crs_from_triplets_double(int n_rows, int n_cols, const triplet_d_t *t, int nnz, crs_d_t *out) {
    crs_d_t a;
    a.n_rows = n_rows;
    a.n_cols = n_cols;
    a.nnz = nnz;

    a.values = (double *)malloc((size_t)nnz * sizeof(double));
    a.col_idx = (int *)malloc((size_t)nnz * sizeof(int));
    a.row_ptr = (int *)calloc((size_t)n_rows + 1, sizeof(int));
    if (!a.values || !a.col_idx || !a.row_ptr) {
		free_crs_d(&a);
		return 0;
	}

    for (int k = 0; k < nnz; k++) {
        int i = t[k].i;
        if (i < 0 || i >= n_rows) { free_crs_d(&a);
			return 0;
		}

        a.row_ptr[i + 1]++;
    }

    for (int i = 0; i < n_rows; i++)
		a.row_ptr[i + 1] += a.row_ptr[i];

    int *next = (int *)malloc((size_t)n_rows * sizeof(int));
    if (!next) {
		free_crs_d(&a);
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

int build_ccs_from_triplets_double(int n_rows, int n_cols, const triplet_d_t *t, int nnz, ccs_d_t *out) {
    ccs_d_t a;
    a.n_rows = n_rows;
    a.n_cols = n_cols;
    a.nnz = nnz;

    a.values = (double *)malloc((size_t)nnz * sizeof(double));
    a.row_idx = (int *)malloc((size_t)nnz * sizeof(int));
    a.col_ptr = (int *)calloc((size_t)n_cols + 1, sizeof(int));
    if (!a.values || !a.row_idx || !a.col_ptr) {
		free_ccs_d(&a);
		return 0;
	}

    for (int k = 0; k < nnz; k++) {
        int j = t[k].j;
        if (j < 0 || j >= n_cols) {
			free_ccs_d(&a);
			return 0;
		}

        a.col_ptr[j + 1]++;
    }

    for (int j = 0; j < n_cols; j++)
		a.col_ptr[j + 1] += a.col_ptr[j];

    int *next = (int *)malloc((size_t)n_cols * sizeof(int));
    if (!next) {
		free_ccs_d(&a);
		return 0;
	}

    for (int j = 0; j < n_cols; j++)
		next[j] = a.col_ptr[j];

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

tjds_t build_tjds_from_ccs(const ccs_t *c) {
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
		free(col_nnz); free(order); return a;
	}

    for (int j = 0; j < n_cols; j++) {
        int count = c->col_ptr[j + 1] - c->col_ptr[j];
        col_nnz[j] = count;
        order[j] = j;
        if (count > a.num_tjd) a.num_tjd = count;
    }

	nnz_pair_t *pairs = malloc((size_t)n_cols * sizeof(nnz_pair_t));
	if (!pairs) { free(col_nnz); free(order); return a; }

	for (int j = 0; j < n_cols; j++) {
		pairs[j].idx = j;
		pairs[j].nnz = col_nnz[j];
	}

	qsort(pairs, (size_t)n_cols, sizeof(nnz_pair_t), cmp_nnz_desc);

	for (int j = 0; j < n_cols; j++)
		order[j] = pairs[j].idx;

	free(pairs);


    a.perm = (int *)malloc((size_t)n_cols * sizeof(int));
    a.tjd = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.row_idx = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.tjd_ptr = (int *)malloc((size_t)(a.num_tjd + 1) * sizeof(int));
    if (!a.perm || !a.tjd || !a.row_idx || !a.tjd_ptr) {
        free(col_nnz); free(order);
        free_tjds(&a);
        return a;
    }

    for (int cidx = 0; cidx < n_cols; cidx++)
		a.perm[cidx] = order[cidx];

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

tjds_d_t build_tjds_from_ccs_double(const ccs_d_t *c) {
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
    if (!col_nnz || !order) {
		free(col_nnz); free(order); return a;
	}

    for (int j = 0; j < n_cols; j++) {
        int count = c->col_ptr[j + 1] - c->col_ptr[j];
        col_nnz[j] = count;
        order[j] = j;
        if (count > a.num_tjd) a.num_tjd = count;
    }

	nnz_pair_t *pairs = malloc((size_t)n_cols * sizeof(nnz_pair_t));
	if (!pairs) { free(col_nnz); free(order); return a; }

	for (int j = 0; j < n_cols; j++) {
		pairs[j].idx = j;
		pairs[j].nnz = col_nnz[j];
	}

	qsort(pairs, (size_t)n_cols, sizeof(nnz_pair_t), cmp_nnz_desc);

	for (int j = 0; j < n_cols; j++)
		order[j] = pairs[j].idx;

	free(pairs);

    a.perm = (int *)malloc((size_t)n_cols * sizeof(int));
    a.tjd = (double *)malloc((size_t)a.nnz * sizeof(double));
    a.row_idx = (int *)malloc((size_t)a.nnz * sizeof(int));
    a.tjd_ptr = (int *)malloc((size_t)(a.num_tjd + 1) * sizeof(int));
    if (!a.perm || !a.tjd || !a.row_idx || !a.tjd_ptr) {
        free(col_nnz); free(order);
        free_tjds_d(&a);
        return a;
    }

    for (int cidx = 0; cidx < n_cols; cidx++)
		a.perm[cidx] = order[cidx];

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

void print_crs_hw(const crs_t *a) {
    printf("CRS 0-based\n");
    printf("nRows = %d\n", a->n_rows);
    printf("nCols = %d\n", a->n_cols);
    printf("nnz = %d\n", a->nnz);
    print_int_array("values", a->values, a->nnz);
    print_int_array("column_index", a->col_idx, a->nnz);
    print_int_array("row_pointers", a->row_ptr, a->n_rows + 1);
}

void print_ccs_hw(const ccs_t *a) {
    printf("CcS 0-based\n");
    printf("nRows = %d\n", a->n_rows);
    printf("nCols = %d\n", a->n_cols);
    printf("nnz = %d\n", a->nnz);
    print_int_array("values", a->values, a->nnz);
    print_int_array("row_index", a->row_idx, a->nnz);
    print_int_array("column_pointers", a->col_ptr, a->n_cols + 1);
}

void print_jds_hw(const jds_t *a) {
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

void print_tjds_hw(const tjds_t *a) {
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

