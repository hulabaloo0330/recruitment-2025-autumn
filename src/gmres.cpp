#include "gmres.hpp" // 包含官方头文件
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>

// CUDA Headers
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

const int RESTART_TIMES = 20;         // 禁止修改
const double REL_RESID_LIMIT = 1e-6;  // 禁止修改
const int ITERATION_LIMIT = 10000;    // 禁止修改

const int H_NUM_ROWS = RESTART_TIMES + 1;

// --- CUDA API Error Checking Macro ---
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// --- cuBLAS API Error Checking Macro ---
#define CUBLAS_CHECK(err) { \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// --- cuSPARSE API Error Checking Macro ---
#define CUSPARSE_CHECK(err) { \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE Error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// =========================================================================
// 内部辅助函数，声明为 static，作用域限制在本文件内
// =========================================================================

static void applyRotation(double &dx, double &dy, double &cs, double &sn) {
    double temp = cs * dx + sn * dy;
    dy = (-sn) * dx + cs * dy;
    dx = temp;
}

static void generateRotation(double &dx, double &dy, double &cs, double &sn) {
    if (dx == 0.0) {
        cs = 0.0;
        sn = 1.0;
    } else {
        double scale = std::abs(dx) + std::abs(dy);
        double norm = scale * std::sqrt((dx / scale) * (dx / scale) + (dy / scale) * (dy / scale));
        double alpha = dx > 0 ? 1.0 : -1.0;
        cs = std::abs(dx) / norm;
        sn = alpha * dy / norm;
    }
}

static void rotation2(uint Am, double *H, double *cs, double *sn, double *s, uint i) {
    for (uint k = 0; k < i; k++) {
        applyRotation(H[i * H_NUM_ROWS + k], H[i * H_NUM_ROWS + (k + 1)], cs[k], sn[k]);
    }
    generateRotation(H[i * H_NUM_ROWS + i], H[i * H_NUM_ROWS + (i + 1)], cs[i], sn[i]);
    applyRotation(H[i * H_NUM_ROWS + i], H[i * H_NUM_ROWS + (i + 1)], cs[i], sn[i]);
    applyRotation(s[i], s[i + 1], cs[i], sn[i]);
}

static void sovlerTri(int Am, int i, double *H, double *s) {
    for (int j = i; j >= 0; j--) {
        s[j] /= H[j * H_NUM_ROWS + j];
        for (int k = j - 1; k >= 0; k--) {
            s[k] -= H[j * H_NUM_ROWS + k] * s[j];
        }
    }
}

static void spmv_cpu(const uint *rowPtr, const uint *colInd, const double *values,
          const double *x, double *y, uint numRows) {
    for (uint i = 0; i < numRows; ++i) {
        double sum = 0.0;
        for (uint j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
            sum += values[j] * x[colInd[j]];
        }
        y[i] = sum;
    }
}

static double calculateNorm_cpu(const double *vec, uint N) {
    double sum = 0.0;
    for (uint i = 0; i < N; ++i) {
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}

// =========================================================================
// gmres.hpp 中声明的函数实现
// =========================================================================

void initialize(SpM<double> *A, double *x, double *b) {
    int N = A->nrows;
    std::vector<double> temp_x(N);

    for (int i = 0; i < N; i++) {
        temp_x[i] = sin(i);
    }

    double beta = calculateNorm_cpu(temp_x.data(), N);
    for (uint i = 0; i < N; i++) {
        temp_x[i] /= beta;
    }

    spmv_cpu(A->rows, A->cols, A->vals, temp_x.data(), b, N);

    for (uint i = 0; i < N; i++) x[i] = 0.0;
}

RESULT gmres(SpM<double> *A_h, double *x_h, double *b_h) {
    const uint N = A_h->nrows;
    const uint NNZ = A_h->nnz;

    std::vector<double> s(RESTART_TIMES + 1, 0.0);
    std::vector<double> H(H_NUM_ROWS * RESTART_TIMES, 0.0);
    std::vector<double> cs(RESTART_TIMES, 0.0);
    std::vector<double> sn(RESTART_TIMES, 0.0);
    
    // 在计时器外声明句柄和设备指针
   
    auto start = std::chrono::high_resolution_clock::now();  // 禁止修改
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    cusparseSpMatDescr_t matA;
    void *d_rows, *d_cols, *d_vals;
    double *x_d, *b_d, *r_d, *V_d;
    void *d_buffer = nullptr;
    cusparseDnVecDescr_t vecX_descr, vecR_descr;

    int i, j, k;
    double resid;
    int iteration = 0;
    double init_res;

    // --- 初始化代码移入计时器内部 ---
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));

    CUDA_CHECK(cudaMalloc(&d_rows, (N + 1) * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_cols, NNZ * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_vals, NNZ * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_rows, A_h->rows, (N + 1) * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, A_h->cols, NNZ * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, A_h->vals, NNZ * sizeof(double), cudaMemcpyHostToDevice));

    CUSPARSE_CHECK(cusparseCreateCsr(&matA, N, N, NNZ, d_rows, d_cols, d_vals,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    
    CUDA_CHECK(cudaMalloc((void **)&x_d, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&b_d, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&r_d, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&V_d, (RESTART_TIMES + 1) * N * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(x_d, x_h, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b_h, N * sizeof(double), cudaMemcpyHostToDevice));
    
    size_t buffer_size = 0;
    const double gpu_alpha_one = 1.0;
    const double gpu_beta_zero = 0.0;

    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX_descr, N, x_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecR_descr, N, r_d, CUDA_R_64F));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &gpu_alpha_one, matA, vecX_descr, &gpu_beta_zero, vecR_descr, 
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));

    double beta;
    CUBLAS_CHECK(cublasDnrm2(cublas_handle, N, b_d, 1, &beta));
    double RESID_LIMIT = REL_RESID_LIMIT * beta;
    init_res = beta;
    // --- 初始化结束 ---

    do {
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecX_descr, x_d));
        CUSPARSE_CHECK(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &gpu_alpha_one, matA, vecX_descr, &gpu_beta_zero, vecR_descr, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
        
        const double gpu_minus_one = -1.0;
        CUBLAS_CHECK(cublasDaxpy(cublas_handle, N, &gpu_minus_one, b_d, 1, r_d, 1));
        
        CUBLAS_CHECK(cublasDnrm2(cublas_handle, N, r_d, 1, &beta));
        
        const double alpha_scal = -1.0 / beta;
        CUBLAS_CHECK(cublasDscal(cublas_handle, N, &alpha_scal, r_d, 1));
        
        CUBLAS_CHECK(cublasDcopy(cublas_handle, N, r_d, 1, V_d, 1));

        fill(s.begin(), s.end(), 0.0);
        s[0] = beta;
        resid = std::abs(beta);
        i = -1;

        if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT) {
            break;
        }
        do {
            i++;
            iteration++;
            
            CUSPARSE_CHECK(cusparseDnVecSetValues(vecX_descr, V_d + i * N));
            CUSPARSE_CHECK(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &gpu_alpha_one, matA, vecX_descr,
                                        &gpu_beta_zero, vecR_descr, CUDA_R_64F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));

            for (k = 0; k <= i; k++) {
                double h_val;
                CUBLAS_CHECK(cublasDdot(cublas_handle, N, r_d, 1, V_d + k * N, 1, &h_val));
                H[i * H_NUM_ROWS + k] = h_val;
                
                const double alpha_daxpy = -H[i * H_NUM_ROWS + k];
                CUBLAS_CHECK(cublasDaxpy(cublas_handle, N, &alpha_daxpy, V_d + k * N, 1, r_d, 1));
            }
            
            double h_norm;
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, N, r_d, 1, &h_norm));
            H[i * H_NUM_ROWS + (i + 1)] = h_norm;

            if(h_norm != 0.0) {
                const double alpha_dscal_2 = 1.0 / h_norm;
                CUBLAS_CHECK(cublasDscal(cublas_handle, N, &alpha_dscal_2, r_d, 1));
            }
            
            CUBLAS_CHECK(cublasDcopy(cublas_handle, N, r_d, 1, V_d + (i + 1) * N, 1));

            rotation2(RESTART_TIMES, H.data(), cs.data(), sn.data(), s.data(), i);

            resid = std::abs(s[i + 1]);

            if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT) {
                break;
            }
        } while (i + 1 < RESTART_TIMES && iteration <= ITERATION_LIMIT);

        sovlerTri(RESTART_TIMES, i, H.data(), s.data());

        for (j = 0; j <= i; j++) {
            CUBLAS_CHECK(cublasDaxpy(cublas_handle, N, &s[j], V_d + j * N, 1, x_d, 1));
        }
    } while (resid > RESID_LIMIT && iteration <= ITERATION_LIMIT);
    CUDA_CHECK(cudaMemcpy(x_h, x_d, N * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_buffer);
    cusparseDestroyDnVec(vecX_descr);
    cusparseDestroyDnVec(vecR_descr);
    cusparseDestroySpMat(matA);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_vals);
    cudaFree(x_d);
    cudaFree(b_d);
    cudaFree(r_d);
    cudaFree(V_d);
    cusparseDestroy(cusparse_handle);
    cublasDestroy(cublas_handle);
    auto stop = std::chrono::high_resolution_clock::now();  // 禁止修改
    std::chrono::duration<float, std::milli> duration = stop - start; // 禁止修改
    float test_time = duration.count();  // 禁止修改

    

    return std::make_tuple(iteration, test_time, resid / init_res);  // 禁止修改
}