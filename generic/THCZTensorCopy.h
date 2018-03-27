#ifndef THCZ_GENERIC_FILE
#define THCZ_GENERIC_FILE "generic/THCZTensorCopy.h"
#else

THC_API void THCZTensor_(copy)(THCState *state, THCZTensor *self, THCZTensor *src);
THC_API void THCZTensor_(copyIgnoringOverlaps)(THCState *state, THCZTensor *self, THCZTensor *src);
THC_API void THCZTensor_(copyByte)(THCState *state, THCZTensor *self, THByteTensor *src);
THC_API void THCZTensor_(copyChar)(THCState *state, THCZTensor *self, THCharTensor *src);
THC_API void THCZTensor_(copyShort)(THCState *state, THCZTensor *self, THShortTensor *src);
THC_API void THCZTensor_(copyInt)(THCState *state, THCZTensor *self, THIntTensor *src);
THC_API void THCZTensor_(copyLong)(THCState *state, THCZTensor *self, THLongTensor *src);
THC_API void THCZTensor_(copyFloat)(THCState *state, THCZTensor *self, THFloatTensor *src);
THC_API void THCZTensor_(copyDouble)(THCState *state, THCZTensor *self, THDoubleTensor *src);
THC_API void THCZTensor_(copyZFloat)(THCState *state, THCZTensor *self, THZFloatTensor *src);
THC_API void THCZTensor_(copyZDouble)(THCState *state, THCZTensor *self, THZDoubleTensor *src);
THC_API void THCZTensor_(copyHalf)(THCState *state, THCZTensor *self, struct THHalfTensor *src);

THC_API void THCZTensor_(copyCudaByte)(THCState *state, THCZTensor *dst, struct THCudaByteTensor *src);
THC_API void THCZTensor_(copyCudaChar)(THCState *state, THCZTensor *dst, struct THCudaCharTensor *src);
THC_API void THCZTensor_(copyCudaShort)(THCState *state, THCZTensor *dst, struct THCudaShortTensor *src);
THC_API void THCZTensor_(copyCudaInt)(THCState *state, THCZTensor *dst, struct THCudaIntTensor *src);
THC_API void THCZTensor_(copyCudaLong)(THCState *state, THCZTensor *dst, struct THCudaLongTensor *src);
THC_API void THCZTensor_(copyCudaFloat)(THCState *state, THCZTensor *dst, struct THCudaTensor *src);
THC_API void THCZTensor_(copyCudaDouble)(THCState *state, THCZTensor *dst, struct THCudaDoubleTensor *src);
THC_API void THCZTensor_(copyCudaZFloat)(THCState *state, THCZTensor *dst, struct THCudaZFloatTensor *src);
THC_API void THCZTensor_(copyCudaZDouble)(THCState *state, THCZTensor *dst, struct THCudaZDoubleTensor *src);
#ifdef CUDA_HALF_TENSOR
THC_API void THCZTensor_(copyCudaHalf)(THCState *state, THCZTensor *dst, struct THCudaHalfTensor *src);
#endif

THC_API void TH_CONCAT_2(THByteTensor_copyCuda  , NType)  (THCState *state, THByteTensor *self, THCZTensor *src);
THC_API void TH_CONCAT_2(THCharTensor_copyCuda  , NType)  (THCState *state, THCharTensor *self, THCZTensor *src);
THC_API void TH_CONCAT_2(THShortTensor_copyCuda , NType)  (THCState *state, THShortTensor *self, THCZTensor *src);
THC_API void TH_CONCAT_2(THIntTensor_copyCuda   , NType)  (THCState *state, THIntTensor *self, THCZTensor *src);
THC_API void TH_CONCAT_2(THLongTensor_copyCuda  , NType)  (THCState *state, THLongTensor *self, THCZTensor *src);
THC_API void TH_CONCAT_2(THFloatTensor_copyCuda , NType)  (THCState *state, THFloatTensor *self, THCZTensor *src);
THC_API void TH_CONCAT_2(THDoubleTensor_copyCuda, NType)  (THCState *state, THDoubleTensor *self, THCZTensor *src);
THC_API void TH_CONCAT_2(THZFloatTensor_copyCuda , NType)  (THCState *state, THZFloatTensor *self, THCZTensor *src);
THC_API void TH_CONCAT_2(THZDoubleTensor_copyCuda, NType)  (THCState *state, THZDoubleTensor *self, THCZTensor *src);
THC_API void TH_CONCAT_2(THHalfTensor_copyCuda, NType)    (THCState *state, THHalfTensor *self, THCZTensor *src);
THC_API void THCZTensor_(copyCuda) (THCState *state, THCZTensor *self, THCZTensor *src);

THC_API void THZTensor_(copyCuda) (THCState *state, THZTensor *self, THCZTensor *src);
THC_API void THCZTensor_(copyCPU) (THCState *state, THCZTensor *self, THZTensor *src);

THC_API void THCZTensor_(copyAsyncCPU)(THCState *state, THCZTensor *self, THZTensor *src);
THC_API void THZTensor_(copyAsyncCuda)(THCState *state, THZTensor *self, THCZTensor *src);

#endif
