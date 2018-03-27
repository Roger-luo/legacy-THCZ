#include "THCZBlas.h"
#include "THCZGeneral.h"

cuComplex THCudaBlas_Cdotc(THCState *state, int64_t n, cuComplex *x, int64_t incx, cuComplex *y, int64_t incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    cuComplex result;
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasCdotc(handle, i_n, x, i_incx, y, i_incy, &result));
    return result;
  }

  THError("Cublas_Cdot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return cuComplex();
}


cuDoubleComplex THCudaBlas_Zdotc(THCState *state, int64_t n, cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    cuDoubleComplex result;
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasZdotc(handle, i_n, x, i_incx, y, i_incy, &result));
    return result;
  }

  THError("Cublas_Zdot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return cuDoubleComplex();
}


/* Level 2 */
void THCudaBlas_Cgemv(THCState *state, char trans, int64_t m, int64_t n, cuComplex alpha, cuComplex *a, int64_t lda, cuComplex *x, int64_t incx, cuComplex beta, cuComplex *y, int64_t incy)
{
  if(n == 1)
    lda = m;

  cublasOperation_t op;
  if (trans == 't') op = CUBLAS_OP_T;
  else if (trans == 'n') op = CUBLAS_OP_N;
  else if (trans == 'c') op = CUBLAS_OP_C;

  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasCgemv(handle, op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));
    return;
  }
  THError("Cublas_Cgemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}


void THCudaBlas_Zgemv(THCState *state, char trans, int64_t m, int64_t n, cuDoubleComplex alpha, cuDoubleComplex *a, int64_t lda, cuDoubleComplex *x, int64_t incx, cuDoubleComplex beta, cuDoubleComplex *y, int64_t incy)
{
  if(n == 1)
    lda = m;

  cublasOperation_t op;
  if (trans == 't') op = CUBLAS_OP_T;
  else if (trans == 'n') op = CUBLAS_OP_N;
  else if (trans == 'c') op = CUBLAS_OP_C;

  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasZgemv(handle, op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));
    return;
  }
  THError("Cublas_Zgemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

void THCudaBlas_Cgerc(THCState *state, int64_t m, int64_t n, cuComplex alpha, cuComplex *x, int64_t incx, cuComplex *y, int64_t incy, cuComplex *a, int64_t lda)
{
  if(n == 1)
    lda = m;

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
      cublasSetStream(handle, THCState_getCurrentStream(state));
      THCublasCheck(cublasCgerc(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_Cgerc only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}


void THCudaBlas_Zgerc(THCState *state, int64_t m, int64_t n, cuDoubleComplex alpha, cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy, cuDoubleComplex *a, int64_t lda)
{
  if(n == 1)
    lda = m;

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
      cublasSetStream(handle, THCState_getCurrentStream(state));
      THCublasCheck(cublasZgerc(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_Zgerc only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}


/* Level 3 */
void THCudaBlas_Cgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, cuComplex alpha, cuComplex *a, int64_t lda, cuComplex *b, int64_t ldb, cuComplex beta, cuComplex *c, int64_t ldc)
{
  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasCgemm(handle, opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Cublas_Cgemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}


void THCudaBlas_Zgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, cuDoubleComplex alpha, cuDoubleComplex *a, int64_t lda, cuDoubleComplex *b, int64_t ldb, cuDoubleComplex beta, cuDoubleComplex *c, int64_t ldc)
{
  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasZgemm(handle, opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Cublas_Zgemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}


void THCudaBlas_CgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                           cuComplex alpha, const cuComplex *a[], int64_t lda, const cuComplex *b[], int64_t ldb,
                           cuComplex beta, cuComplex *c[], int64_t ldc, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_CgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasCgemmBatched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
                                   (int)batchCount));
}


#if CUDA_VERSION >= 8000
void THCudaBlas_CgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                           cuComplex alpha, const cuComplex *a, int64_t lda, int64_t strideA, const cuComplex *b, int64_t ldb, int64_t strideB,
                           cuComplex beta, cuComplex *c, int64_t ldc, int64_t strideC, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_CgemmStridedBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasCgemmStridedBatched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, strideA, b, (int)ldb, strideB, &beta, c, (int)ldc, strideC,
                                   (int)batchCount));
}
#endif


void THCudaBlas_ZgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                             cuDoubleComplex alpha, const cuDoubleComplex *a[], int64_t lda, const cuDoubleComplex *b[], int64_t ldb,
                             cuDoubleComplex beta, cuDoubleComplex *c[], int64_t ldc, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_ZgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasZgemmBatched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
                                   (int)batchCount));
}

#if CUDA_VERSION >= 8000
void THCudaBlas_ZgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                 cuDoubleComplex alpha, const cuDoubleComplex *a, int64_t lda, int64_t strideA, const cuDoubleComplex *b, int64_t ldb, int64_t strideB, 
                                 cuDoubleComplex beta, cuDoubleComplex *c, int64_t ldc, int64_t strideC, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_ZgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasZgemmStridedBatched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, strideA, b, (int)ldb, strideB, &beta, c, (int)ldc, strideC, 
                                   (int)batchCount));
}
#endif

/* Inverse */
void THCudaBlas_Cgetrf(THCState *state, int n, cuComplex **a, int lda, int *pivot, int *info, int batchSize) {
  if( (n >= INT_MAX) || (lda >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Cgetrf only supports n, lda, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasCgetrfBatched(handle, n, a, lda, pivot, info, batchSize));
}

void THCudaBlas_Zgetrf(THCState *state, int n, cuDoubleComplex **a, int lda, int *pivot, int *info, int batchSize) {
  if( (n >= INT_MAX) || (lda >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Zgetrf only supports n, lda, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasZgetrfBatched(handle, n, a, lda, pivot, info, batchSize));
}

void THCudaBlas_Cgetrs(THCState *state, char transa, int n, int nrhs, const cuComplex **a, int lda, int *pivot, cuComplex **b, int ldb, int *info, int batchSize)
{
  if( (n >= INT_MAX) || (nrhs >= INT_MAX) || (lda >= INT_MAX) || (ldb >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Cgetrs only supports n, nrhs, lda, ldb, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }

  // no need to adjust leading dimensions, since matrices are square
  cublasOperation_t opa = convertTransToCublasOperation(transa);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasCgetrsBatched(handle, opa, n, nrhs, a, lda, pivot, b, ldb, info, batchSize));
}


void THCudaBlas_Zgetrs(THCState *state, char transa, int n, int nrhs, const cuDoubleComplex **a, int lda, int *pivot, cuDoubleComplex **b, int ldb, int *info, int batchSize)
{
  if( (n >= INT_MAX) || (nrhs >= INT_MAX) || (lda >= INT_MAX) || (ldb >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Zgetrs only supports n, nrhs, lda, ldb, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }

  // no need to adjust leading dimensions, since matrices are square
  cublasOperation_t opa = convertTransToCublasOperation(transa);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasZgetrsBatched(handle, opa, n, nrhs, a, lda, pivot, b, ldb, info, batchSize));
}

void THCudaBlas_Cgetri(THCState *state, int n, const cuComplex **a, int lda, int *pivot, cuComplex **c, int ldc, int *info, int batchSize) {

  if( (n >= INT_MAX) || (lda >= INT_MAX)|| (ldc >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Cgetri only supports n, lda, ldc, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasCgetriBatched(handle, n, a, lda, pivot, c, ldc, info, batchSize));
}

void THCudaBlas_Zgetri(THCState *state, int n, const cuDoubleComplex **a, int lda, int *pivot, cuDoubleComplex **c, int ldc, int *info, int batchSize) {

  if( (n >= INT_MAX) || (lda >= INT_MAX)|| (ldc >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Zgetri only supports n, lda, ldc, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasZgetriBatched(handle, n, a, lda, pivot, c, ldc, info, batchSize));
}

