#ifndef THCZ_GENERIC_FILE
#define THCZ_GENERIC_FILE "generic/THCZTensor.cu"
#else

cudaTextureObject_t THCZTensor_(getTextureObject)(THCState *state, THCZTensor *self)
{
  THCAssertSameGPU(THCZTensor_(checkGPU)(state, 1, self));
  cudaTextureObject_t texObj;
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = THCZTensor_(data)(state, self);
  resDesc.res.linear.sizeInBytes = THCZTensor_(nElement)(state, self) * 4;
  resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0,
                                                  cudaChannelFormatKindFloat);
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess) {
    if (THCZTensor_(nElement)(state, self) > 2>>27)
      THError("Failed to create texture object, "
              "nElement:%ld exceeds 27-bit addressing required for tex1Dfetch. Cuda Error: %s",
              THCZTensor_(nElement)(state, self), cudaGetErrorString(errcode));
    else
      THError("Failed to create texture object: %s", cudaGetErrorString(errcode));
  }
  return texObj;
}

THC_API int THCZTensor_(getDevice)(THCState* state, const THCZTensor* tensor) {
  if (!tensor->storage) return -1;
  return THCZStorage_(getDevice)(state, tensor->storage);
}

#endif
