#include "THCZTensor.h"
#include "THCZTensorTypeUtils.cuh"

#include "THC/THCTensorCopy.cuh"
#include "THC/THCApply.cuh"
#include "THC/THCHalf.h"
#include "THCZNumerics.cuh"

// Copy operator for the pointwise apply kernel
template <typename TypeDst>
struct CopyOp<TypeDst, thrust::complex<float>> {
  __device__ __forceinline__ void operator()(TypeDst* dst, thrust::complex<float>* src) {
#if __CUDA_ARCH__ >= 350
    cuComplex temp = __ldg(reinterpret_cast<cuComplex*>(src));
    *dst = ScalarConvert<thrust::complex<float>, TypeDst>::to(thrust::complex<float>(temp.x, temp.y));
#else
    *dst = ScalarConvert<thrust::complex<float>, TypeDst>::to(*src);
#endif
  }
};

template <typename TypeDst>
struct CopyOp<TypeDst, thrust::complex<double>> {
  __device__ __forceinline__ void operator()(TypeDst* dst, thrust::complex<double>* src) {
#if __CUDA_ARCH__ >= 350
    cuDoubleComplex temp = __ldg(reinterpret_cast<cuDoubleComplex*>(src));
    *dst = ScalarConvert<thrust::complex<double>, TypeDst>::to(thrust::complex<double>(temp.x, temp.y));
#else
    *dst = ScalarConvert<thrust::complex<double>, TypeDst>::to(*src);
#endif
  }
};


#include "generic/THCZTensorCopy.cu"
#include "THCZGenerateAllTypes.h"
