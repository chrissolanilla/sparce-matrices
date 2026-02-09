#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include "matrix_multiply_io.h"
#include "formats.h"
#include "spmv.h"
#include "bench.h"
#include "util.h"

void demo_q1(void);
void run_ibm32_dense(const char *path);
void run_memplus_sparse(const char *path);

void demo_q1(void) {
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

void run_ibm32_dense(const char *mtx_path) {
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
    for (int i = 0; i < n_cols; i++)
		x[i] = 1;

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
    if (n_rows > 1)
		printf(" y[1]=%d", y[1]);
    if (n_rows > 2)
		printf(" y[2]=%d", y[2]);
    printf("\n\n");

    free_crs(&crs);
    free_tjds(&tjds);
    free(a);
    free(x);
    free(y);
}

void run_memplus_sparse(const char *mtx_path) {
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

    for (int i = 0; i < n_cols; i++)
		x[i] = 1.0;

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
    if (n_rows > 1)
		printf("  y[1]=%.6g vs %.6g\n", y_crs[1], y_tjds[1]);
    if (n_rows > 2)
		printf("  y[2]=%.6g vs %.6g\n", y_crs[2], y_tjds[2]);

    double diff = s1 - s2;
    if (diff < 0)
		diff = -diff;
    if (diff > 1e-8)
		printf("  bruh: checksum mismatch (tjds bug)\n");
    else
		printf("  ok: checksums match\n");
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

