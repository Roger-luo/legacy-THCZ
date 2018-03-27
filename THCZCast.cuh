#ifndef THCZ_CAST_INC
#define THCZ_CAST_INC

#include <thrust/complex.h>
#include <cuComplex.h>

typedef thrust::complex<float> complex64;
typedef thrust::complex<double> complex128;


inline complex128 complex_cast(cuDoubleComplex x) {
  return complex128(cuCreal(x), cuCimag(x));
}

inline complex64 complex_cast(cuComplex x) {
  return complex64(cuCrealf(x), cuCimagf(x));
}

inline cuComplex complex_cast(complex64 x) {
  return make_cuComplex(x.real(), x.imag());
}

inline cuDoubleComplex complex_cast(complex128 x) {
  return make_cuDoubleComplex(x.real(), x.imag());
}

inline complex64 *complex_cast(cuComplex *x) {
  return reinterpret_cast<complex64*>(x);
}

inline complex128 *complex_cast(cuDoubleComplex *x) {
  return reinterpret_cast<complex128*>(x);
}

inline cuComplex *complex_cast(complex64 *x) {
  return reinterpret_cast<cuComplex*>(x);
}

inline cuDoubleComplex *complex_cast(complex128 *x) {
  return reinterpret_cast<cuDoubleComplex*>(x);
}

inline cuComplex **complex_cast(complex64 **x) {
  return reinterpret_cast<cuComplex**>(x);
}

inline cuDoubleComplex **complex_cast(complex128 **x) {
  return reinterpret_cast<cuDoubleComplex**>(x);
}

#endif // THCZ_CAST_INC