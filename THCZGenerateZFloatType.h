#ifndef THCZ_GENERIC_FILE
#error "You must define THCZ_GENERIC_FILE before including THCZGenerateFloatType.h"
#endif

#define ntype thrust::complex<float>
/* FIXME: fp64 has bad performance on some platforms; avoid using it unless
   we opt into it? */
#define accntype thrust::complex<float>
#define NType ZFloat
#define CNType CudaZFloat
#define THCZ_NTYPE_IS_ZFLOAT
#line 1 THCZ_GENERIC_FILE
#include THCZ_GENERIC_FILE
#undef ntype
#undef accntype
#undef NType
#undef CNType
#undef THCZ_NTYPE_IS_ZFLOAT

#ifndef THCZGenerateAllTypes
#ifndef THCZGenerateFloatTypes
#undef THCZ_GENERIC_FILE
#endif
#endif
