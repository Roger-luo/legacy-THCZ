#ifndef THCZ_BLAS_INC
#define THCZ_BLAS_INC

#include "THCZGeneral.h"
#include "THC/THCBlas.h"

/* Level 1 */
THC_API cuComplex THCudaBlas_Cdotc(THCState *state, int64_t n, cuComplex *x, int64_t incx, cuComplex *y, int64_t incy);
THC_API cuDoubleComplex THCudaBlas_Zdotc(THCState *state, int64_t n, cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy);

/* Level 2 */
THC_API void THCudaBlas_Cgemv(THCState *state, char trans, int64_t m, int64_t n, cuComplex alpha, cuComplex *a, int64_t lda, cuComplex *x, int64_t incx, cuComplex beta, cuComplex *y, int64_t incy);
THC_API void THCudaBlas_Zgemv(THCState *state, char trans, int64_t m, int64_t n, cuDoubleComplex alpha, cuDoubleComplex *a, int64_t lda, cuDoubleComplex *x, int64_t incx, cuDoubleComplex beta, cuDoubleComplex *y, int64_t incy);
THC_API void THCudaBlas_Cgerc(THCState *state, int64_t m, int64_t n, cuComplex alpha, cuComplex *x, int64_t incx, cuComplex *y, int64_t incy, cuComplex *a, int64_t lda);
THC_API void THCudaBlas_Zgerc(THCState *state, int64_t m, int64_t n, cuDoubleComplex alpha, cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy, cuDoubleComplex *a, int64_t lda);

/* Level 3 */
THC_API void THCudaBlas_Cgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, cuComplex alpha, cuComplex *a, int64_t lda, cuComplex *b, int64_t ldb, cuComplex beta, cuComplex *c, int64_t ldc);
THC_API void THCudaBlas_Zgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, cuDoubleComplex alpha, cuDoubleComplex *a, int64_t lda, cuDoubleComplex *b, int64_t ldb, cuDoubleComplex beta, cuDoubleComplex *c, int64_t ldc);

THC_API void THCudaBlas_CgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     cuComplex alpha, const cuComplex *a[], int64_t lda, const cuComplex *b[], int64_t ldb,
                                     cuComplex beta, cuComplex *c[], int64_t ldc, int64_t batchCount);
THC_API void THCudaBlas_ZgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     cuDoubleComplex alpha, const cuDoubleComplex *a[], int64_t lda, const cuDoubleComplex *b[], int64_t ldb,
                                     cuDoubleComplex beta, cuDoubleComplex *c[], int64_t ldc, int64_t batchCount);

#if CUDA_VERSION >= 8000
THC_API void THCudaBlas_CgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     cuComplex alpha, const cuComplex *a, int64_t lda, int64_t strideA, const cuComplex *b, int64_t ldb, int64_t strideB,
                                     cuComplex beta, cuComplex *c, int64_t ldc, int64_t strideC, int64_t batchCount);
THC_API void THCudaBlas_ZgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     cuDoubleComplex alpha, const cuDoubleComplex *a, int64_t lda, int64_t strideA, const cuDoubleComplex *b, int64_t ldb, int64_t strideB, 
                                     cuDoubleComplex beta, cuDoubleComplex *c, int64_t ldc, int64_t strideC, int64_t batchCount);
#endif

/* Inverse */
THC_API void THCudaBlas_Cgetrf(THCState *state, int n, cuComplex **a, int lda, int *pivot, int *info, int batchSize);
THC_API void THCudaBlas_Zgetrf(THCState *state, int n, cuDoubleComplex **a, int lda, int *pivot, int *info, int batchSize);

THC_API void THCudaBlas_Cgetrs(THCState *state, char transa, int n, int nrhs, const cuComplex **a, int lda, int *pivot, cuComplex **b, int ldb, int *info, int batchSize);
THC_API void THCudaBlas_Zgetrs(THCState *state, char transa, int n, int nrhs, const cuDoubleComplex **a, int lda, int *pivot, cuDoubleComplex **b, int ldb, int *info, int batchSize);

THC_API void THCudaBlas_Cgetri(THCState *state, int n, const cuComplex **a, int lda, int *pivot, cuComplex **c, int ldc, int *info, int batchSize);
THC_API void THCudaBlas_Zgetri(THCState *state, int n, const cuDoubleComplex **a, int lda, int *pivot, cuDoubleComplex **c, int ldc, int *info, int batchSize);

#endif