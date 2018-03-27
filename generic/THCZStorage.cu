#ifndef THCZ_GENERIC_FILE
#define THCZ_GENERIC_FILE "generic/THCZStorage.cu"
#else

void THCZStorage_(fill)(THCState *state, THCZStorage *self, ntype value)
{
  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<ntype> self_data(self->data);
  thrust::fill(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
    self_data, self_data+self->size, value);
}

void THCZStorage_(resize)(THCState *state, THCZStorage *self, ptrdiff_t size)
{
  THArgCheck(size >= 0, 2, "invalid size");
  THAssert(self->allocator != NULL);
  int device;
  THCudaCheck(cudaGetDevice(&device));

  if(!(self->flag & TH_STORAGE_RESIZABLE))
    THError("Trying to resize storage that is not resizable");

  if (self->allocator->realloc) {
    cudaError_t err = (*self->allocator->realloc)(
      self->allocatorContext,
      (void**)&(self->data),
      self->size * sizeof(ntype),
      size * sizeof(ntype), THCState_getCurrentStream(state));
    if (err != cudaSuccess) {
      THCudaCheck(err);
    }
    self->size = size;
    self->device = device;
    return;
  }

  if(size == 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      THCudaCheck(
        (*self->allocator->free)(self->allocatorContext, self->data));
    }
    self->data = NULL;
    self->size = 0;
    self->device = device;
  }
  else
  {
    ntype *data = NULL;
    cudaError_t err =
      (*self->allocator->malloc)(self->allocatorContext,
                                 (void**)&(data),
                                 size * sizeof(ntype),
                                 THCState_getCurrentStream(state));
    THCudaCheck(err);

    if (self->data) {
      // Enable p2p access when the memcpy is across devices
      THCState_getPeerToPeerAccess(state, device, self->device);

      THCudaCheck(cudaMemcpyAsync(data,
                                  self->data,
                                  THMin(self->size, size) * sizeof(ntype),
                                  cudaMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
      if(self->flag & TH_STORAGE_FREEMEM) {
        THCudaCheck(
          (*self->allocator->free)(self->allocatorContext, self->data));
      }
    }

    self->data = data;
    self->size = size;
    self->device = device;
  }
}

THC_API int THCZStorage_(getDevice)(THCState* state, const THCZStorage* storage) {
  return storage->device;
}

#endif
