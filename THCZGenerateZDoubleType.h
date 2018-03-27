#ifndef THCZ_GENERIC_FILE
#error "You must define THCZ_GENERIC_FILE before including THCZGenerateDoubleType.h"
#endif

#define ntype thrust::complex<double>
#define accntype thrust::complex<double>
#define NType ZDouble
#define CNType CudaZDouble
#define THCZ_NTYPE_IS_ZDOUBLE
#line 1 THCZ_GENERIC_FILE
#include THCZ_GENERIC_FILE
#undef ntype
#undef accntype
#undef NType
#undef CNType
#undef THCZ_NTYPE_IS_DOUBLE

#ifndef THCZGenerateAllTypes
#ifndef THCZGenerateFloatTypes
#undef THCZ_GENERIC_FILE
#endif
#endif
