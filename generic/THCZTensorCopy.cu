#ifndef THCZ_GENERIC_FILE
#define THCZ_GENERIC_FILE "generic/THCZTensorCopy.cu"
#else

THC_API void
THCZTensor_(copy)(THCState* state, THCZTensor* dst, THCZTensor* src) {
  if (dst == src) return;
  THC_copyTensor<THCZTensor, THCZTensor>(state, dst, src);
}

THC_API void
THCZTensor_(copyIgnoringOverlaps)(THCState* state, THCZTensor* dst, THCZTensor* src) {
  // Called when we are copying into an overlapping index `dst`, but
  // we don't care which writer wins. Hacky but it works.
  // This is itself invoked by pointwiseApply2 / THCZTensor_copy in
  // case that there are write overlaps.
  // FIXME: really, overlapping writes should be illegal/an error in Torch
  THC_pointwiseApply2(
    state, dst, src,
    CopyOp<typename TensorUtils<THCZTensor>::DataType,
           typename TensorUtils<THCZTensor>::DataType>(),
    ReadOnly, /* ignore overwrites */
    ReadOnly);
}

#define IMPLEMENT_THC_CUDA_TENSOR_COPY(TYPEC, TYPECUDA)                 \
  THC_API void                                                          \
  THCZTensor_(copyCuda##TYPEC)(THCState *state,                          \
                              THCZTensor *self,                          \
                              THCuda##TYPECUDA##Tensor *src) {          \
    THC_copyTensor<THCZTensor, THCuda##TYPECUDA##Tensor>(state, self, src); \
  }

IMPLEMENT_THC_CUDA_TENSOR_COPY(Byte, Byte)
IMPLEMENT_THC_CUDA_TENSOR_COPY(Char, Char)
IMPLEMENT_THC_CUDA_TENSOR_COPY(Short, Short)
IMPLEMENT_THC_CUDA_TENSOR_COPY(Int, Int)
IMPLEMENT_THC_CUDA_TENSOR_COPY(Long, Long)
// THCudaTensor aka the non-existent THCudaFloatTensor
IMPLEMENT_THC_CUDA_TENSOR_COPY(Float, )
IMPLEMENT_THC_CUDA_TENSOR_COPY(Double, Double)
IMPLEMENT_THC_CUDA_TENSOR_COPY(ZFloat, ZFloat)
IMPLEMENT_THC_CUDA_TENSOR_COPY(ZDouble, ZDouble)
#ifdef CUDA_HALF_TENSOR
IMPLEMENT_THC_CUDA_TENSOR_COPY(Half, Half)
#endif

#undef IMPLEMENT_THC_CUDA_TENSOR_COPY

#endif
