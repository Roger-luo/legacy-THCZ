#ifndef THCZ_GENERIC_FILE
#define THCZ_GENERIC_FILE "generic/THCZTensorCopy.cpp"
#else

/* specific methods */

void THCZTensor_(copyCPU)(THCState *state, THCZTensor *self, struct THZTensor *src)
{
  THArgCheck(THCZTensor_(nElement)(state, self) == THZTensor_(nElement)(src), 2, "sizes do not match");

  {
    THCZTensor *selfc = THCZTensor_(newContiguous)(state, self);
    src = THZTensor_(newContiguous)(src);

    cudaStream_t stream = THCState_getCurrentStream(state);
    THCudaCheck(cudaMemcpyAsync(THCZTensor_(data)(state,selfc),
                                THZTensor_(data)(src),
                                THZTensor_(nElement)(src) * sizeof(ntype),
                                cudaMemcpyHostToDevice,
                                stream));
    THCudaCheck(cudaStreamSynchronize(stream));

    THZTensor_(free)(src);
    THCZTensor_(freeCopyTo)(state, selfc, self);
  }
}

#define IMPLEMENT_TH_CUDA_TENSOR_COPY(TYPEC)                            \
void THCZTensor_(copy##TYPEC)(THCState *state, THCZTensor *self, struct TH##TYPEC##Tensor *src)                \
{                                                                       \
  THArgCheck(THCZTensor_(nElement)(state, self) == TH##TYPEC##Tensor_nElement(src), 2, "sizes do not match"); \
  if(THCZTypeIdx_(NType) == THCZTypeIdx_(TYPEC)) {               \
    THCZTensor_(copyCPU)(state, self, (THZTensor*) src);  /* cast just removes warnings */                     \
  } else {                                                              \
    THLongStorage *size = TH##TYPEC##Tensor_newSizeOf(src);             \
    THZTensor *srcf = THZTensor_(newWithSize)(size, NULL);                \
                                                                        \
    THZTensor_(copy##TYPEC)(srcf, src);                                  \
    THCZTensor_(copyCPU)(state, self, srcf);                             \
                                                                        \
    THLongStorage_free(size);                                           \
    THZTensor_(free)(srcf);                                              \
  }                                                                     \
}

IMPLEMENT_TH_CUDA_TENSOR_COPY(Byte)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Char)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Short)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Int)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Long)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Float)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Double)
IMPLEMENT_TH_CUDA_TENSOR_COPY(ZFloat)
IMPLEMENT_TH_CUDA_TENSOR_COPY(ZDouble)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Half)

/* copyCuda */

void THZTensor_(copyCuda)(THCState *state, THZTensor *self, struct THCZTensor *src)
{
  THArgCheck(THZTensor_(nElement)(self) == THCZTensor_(nElement)(state, src), 2, "sizes do not match");

  {
    THZTensor *selfc = THZTensor_(newContiguous)(self);
    src = THCZTensor_(newContiguous)(state, src);

    cudaStream_t stream = THCState_getCurrentStream(state);
    THCudaCheck(cudaMemcpyAsync(THZTensor_(data)(selfc),
                                THCZTensor_(data)(state, src),
                                THCZTensor_(nElement)(state, src) * sizeof(ntype),
                                cudaMemcpyDeviceToHost,
                                stream));
    THCudaCheck(cudaStreamSynchronize(stream));

    THCZTensor_(free)(state, src);
    THZTensor_(freeCopyTo)(selfc, self);
  }
}

#define IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(TYPEC)                           \
  void TH_CONCAT_4(TH,TYPEC,Tensor_copyCuda,NType)(THCState *state, TH##TYPEC##Tensor *self, struct THCZTensor *src) \
  {                                                                       \
    THArgCheck(TH##TYPEC##Tensor_nElement(self) == THCZTensor_(nElement)(state, src), 2, "sizes do not match");       \
    if(THCZTypeIdx_(NType) == THCZTypeIdx_(TYPEC)) {   \
      THZTensor_(copyCuda)(state, (THZTensor*) self, src);  /* cast just removes compiler warning */                   \
    } else {                                                              \
      THLongStorage *size = THCZTensor_(newSizeOf)(state, src);            \
      THZTensor *srcf = THZTensor_(newWithSize)(size, NULL);                \
                                                                          \
      THZTensor_(copyCuda)(state, srcf, src);                              \
      TH_CONCAT_4(TH,TYPEC,Tensor_copy,NType)(self, srcf);                 \
                                                                          \
      THLongStorage_free(size);                                           \
      THZTensor_(free)(srcf);                                              \
    }                                                                     \
  }

IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Byte)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Char)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Short)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Int)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Long)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Float)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Double)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(ZFloat)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(ZDouble)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Half)

void THCZTensor_(copyCuda)(THCState *state, THCZTensor *self, THCZTensor *src)
{
  THCZTensor_(copy)(state, self, src);
}

void THCZTensor_(copyAsyncCPU)(THCState *state, THCZTensor *self, struct THZTensor *src)
{
  THArgCheck(THCZTensor_(nElement)(state, self) == THZTensor_(nElement)(src), 2, "sizes do not match");
  THArgCheck(THCZTensor_(isContiguous)(state, self), 2, "Target tensor must be contiguous");
  THArgCheck(THZTensor_(isContiguous)(src), 3, "Source tensor must be contiguous");

  if (THCZTensor_(nElement)(state, self) == 0) return;

  // Perform the copy wrt the current stream on the CudaTensor's device.
  int tensorDevice = THCZTensor_(getDevice)(state, self);
  int currentDevice;
  THCudaCheck(cudaGetDevice(&currentDevice));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(tensorDevice));
  }

  THCStream *stream  = THCState_getStream(state);
  THCudaCheck(cudaMemcpyAsync(THCZTensor_(data)(state, self),
                              THZTensor_(data)(src),
                              THZTensor_(nElement)(src) * sizeof(ntype),
                              cudaMemcpyHostToDevice,
                              stream->stream));

  THCudaCheck(THCCachingHostAllocator_recordEvent(src->storage->data, stream));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(currentDevice));
  }
}

void THZTensor_(copyAsyncCuda)(THCState *state, THZTensor *self, struct THCZTensor *src)
{
  THArgCheck(THZTensor_(nElement)(self) == THCZTensor_(nElement)(state, src), 2, "sizes do not match");
  THArgCheck(THZTensor_(isContiguous)(self), 2, "Target tensor must be contiguous");
  THArgCheck(THCZTensor_(isContiguous)(state, src), 3, "Source tensor must be contiguous");

  if (THZTensor_(nElement)(self) == 0) return;

  // Perform the copy wrt the current stream on the CudaTensor's device.
  int tensorDevice = THCZTensor_(getDevice)(state, src);
  int currentDevice;
  THCudaCheck(cudaGetDevice(&currentDevice));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(tensorDevice));
  }

  THCStream *stream = THCState_getStream(state);
  THCudaCheck(cudaMemcpyAsync(THZTensor_(data)(self),
                              THCZTensor_(data)(state, src),
                              THCZTensor_(nElement)(state, src) * sizeof(ntype),
                              cudaMemcpyDeviceToHost,
                              stream->stream));

  THCudaCheck(THCCachingHostAllocator_recordEvent(src->storage->data, stream));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(currentDevice));
  }
}

#undef IMPLEMENT_TH_CUDA_TENSOR_COPY
#undef IMPLEMENT_TH_CUDA_TENSOR_COPY_TO

#endif
