#ifndef THCZ_NUMERICS_INC
#define THCZ_NUMERICS_INC

#include "THC/THCNumerics.cuh"
#include <thrust/complex.h>

template <typename DType>
struct THCNumerics<thrust::complex<DType>> {

  using complex = thrust::complex<DType>;

  static inline __host__ __device__ bool eq(complex a, complex b) { return a == b; }
  static inline __host__ __device__ bool ne(complex a, complex b) { return a != b; }

  static inline __host__ __device__  complex exp  (complex a) { return   thrust::exp(a); }
  static inline __host__ __device__  complex log  (complex a) { return   thrust::log(a); }
  // TODO: implement log1p
  static inline __host__ __device__  complex log1p(complex a) { return thrust::log(1+a); }
  static inline __host__ __device__  complex cos  (complex a) { return thrust::cos(a); }
  static inline __host__ __device__  complex sin  (complex a) { return thrust::sin(a); }
  static inline __host__ __device__  complex sqrt (complex a) { return thrust::sqrt(a); }
  static inline __host__ __device__  complex rsqrt(complex a) { return 1.0/thrust::sqrt(a); }
  static inline __host__ __device__  complex neg  (complex a) { return        -a; }
  static inline __host__ __device__  complex acos (complex a) { return  thrust::acos(a); }
  static inline __host__ __device__  complex cosh (complex a) { return  thrust::cosh(a); }
  static inline __host__ __device__  complex acosh(complex a) { return thrust::acosh(a); }
  static inline __host__ __device__  complex asin (complex a) { return  thrust::asin(a); }
  static inline __host__ __device__  complex sinh (complex a) { return  thrust::sinh(a); }
  static inline __host__ __device__  complex asinh(complex a) { return thrust::asinh(a); }
  static inline __host__ __device__  complex tan  (complex a) { return   thrust::tan(a); }
  static inline __host__ __device__  complex atan (complex a) { return  thrust::atan(a); }
  static inline __host__ __device__  complex tanh (complex a) { return  thrust::tanh(a); }
  static inline __host__ __device__  complex abs  (complex a) { return   thrust::abs(a); }
  static inline __host__ __device__  complex cinv (complex a) { return 1.0 / a; }
  static inline __host__ __device__  complex add  (complex a, complex b) { return a + b; }
  static inline __host__ __device__  complex div  (complex a, complex b) { return a / b; }
  static inline __host__ __device__  complex mul  (complex a, complex b) { return a * b; }
  static inline __host__ __device__  complex sub  (complex a, complex b) { return a - b; }
  static inline __host__ __device__  complex pow  (complex a, complex b) { return thrust::pow(a, b); }
};

template <typename OutDType, typename InDType>
struct ScalarConvert<thrust::complex<InDType>, thrust::complex<OutDType>> {
    static __host__ __device__ thrust::complex<OutDType> to(const thrust::complex<InDType> v) {
        return thrust::complex<OutDType>((OutDType)v.real(), (OutDType)v.imag());
    }
};

#endif