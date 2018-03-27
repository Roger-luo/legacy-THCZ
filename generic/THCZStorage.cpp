#ifndef THCZ_GENERIC_FILE
#define THCZ_GENERIC_FILE "generic/THCZStorage.cpp"
#else

#pragma message("processing THCZStorage")

ntype* THCZStorage_(data)(THCState *state, const THCZStorage *self)
{
  return self->data;
}

ptrdiff_t THCZStorage_(size)(THCState *state, const THCZStorage *self)
{
  return self->size;
}

int THCZStorage_(elementSize)(THCState *state)
{
  return sizeof(ntype);
}

void THCZStorage_(set)(THCState *state, THCZStorage *self, ptrdiff_t index, ntype value)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  cudaStream_t stream = THCState_getCurrentStream(state);
  THCudaCheck(cudaMemcpyAsync(self->data + index, &value, sizeof(ntype),
                              cudaMemcpyHostToDevice,
                              stream));
  THCudaCheck(cudaStreamSynchronize(stream));
}

ntype THCZStorage_(get)(THCState *state, const THCZStorage *self, ptrdiff_t index)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  ntype value;
  cudaStream_t stream = THCState_getCurrentStream(state);
  THCudaCheck(cudaMemcpyAsync(&value, self->data + index, sizeof(ntype),
                              cudaMemcpyDeviceToHost, stream));
  THCudaCheck(cudaStreamSynchronize(stream));
  return value;
}

THCZStorage* THCZStorage_(new)(THCState *state)
{
  return THCZStorage_(newWithSize)(state, 0);
}

THCZStorage* THCZStorage_(newWithSize)(THCState *state, ptrdiff_t size)
{
  return THCZStorage_(newWithAllocator)(
    state, size,
    state->cudaDeviceAllocator,
    state->cudaDeviceAllocator->state);
}

THCZStorage* THCZStorage_(newWithAllocator)(THCState *state, ptrdiff_t size,
                                          THCDeviceAllocator* allocator,
                                          void* allocatorContext)
{
  THArgCheck(size >= 0, 2, "invalid size");
  int device;
  THCudaCheck(cudaGetDevice(&device));

  THCZStorage *storage = (THCZStorage*)THAlloc(sizeof(THCZStorage));
  memset(storage, 0, sizeof(THCZStorage));
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocator = allocator;
  storage->allocatorContext = allocatorContext;
  storage->size = size;
  storage->device = device;

  if(size > 0)
  {
    // update heap *before* attempting malloc, to free space for the malloc
    cudaError_t err =
      (*allocator->malloc)(allocatorContext,
                           (void**)&(storage->data),
                           size * sizeof(ntype),
                           THCState_getCurrentStream(state));
    if(err != cudaSuccess){
      free(storage);
    }
    THCudaCheck(err);
  } else {
    storage->data = NULL;
  }
  return storage;
}

THCZStorage* THCZStorage_(newWithSize1)(THCState *state, ntype data0)
{
  THCZStorage *self = THCZStorage_(newWithSize)(state, 1);
  THCZStorage_(set)(state, self, 0, data0);
  return self;
}

THCZStorage* THCZStorage_(newWithSize2)(THCState *state, ntype data0, ntype data1)
{
  THCZStorage *self = THCZStorage_(newWithSize)(state, 2);
  THCZStorage_(set)(state, self, 0, data0);
  THCZStorage_(set)(state, self, 1, data1);
  return self;
}

THCZStorage* THCZStorage_(newWithSize3)(THCState *state, ntype data0, ntype data1, ntype data2)
{
  THCZStorage *self = THCZStorage_(newWithSize)(state, 3);
  THCZStorage_(set)(state, self, 0, data0);
  THCZStorage_(set)(state, self, 1, data1);
  THCZStorage_(set)(state, self, 2, data2);
  return self;
}

THCZStorage* THCZStorage_(newWithSize4)(THCState *state, ntype data0, ntype data1, ntype data2, ntype data3)
{
  THCZStorage *self = THCZStorage_(newWithSize)(state, 4);
  THCZStorage_(set)(state, self, 0, data0);
  THCZStorage_(set)(state, self, 1, data1);
  THCZStorage_(set)(state, self, 2, data2);
  THCZStorage_(set)(state, self, 3, data3);
  return self;
}

THCZStorage* THCZStorage_(newWithMapping)(THCState *state, const char *fileName, ptrdiff_t size, int isShared)
{
  THError("not available yet for THCZStorage");
  return NULL;
}

THCZStorage* THCZStorage_(newWithData)(THCState *state, ntype *data, ptrdiff_t size)
{
  return THCZStorage_(newWithDataAndAllocator)(state, data, size,
                                              state->cudaDeviceAllocator,
                                              state->cudaDeviceAllocator->state);
}

THCZStorage* THCZStorage_(newWithDataAndAllocator)(
  THCState *state, ntype *data, ptrdiff_t size,
  THCDeviceAllocator *allocator, void *allocatorContext) {
  THCZStorage *storage = (THCZStorage*)THAlloc(sizeof(THCZStorage));
  memset(storage, 0, sizeof(THCZStorage));
  storage->data = data;
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocator = allocator;
  storage->allocatorContext = allocatorContext;
  int device;
  if (data) {
    struct cudaPointerAttributes attr;
    THCudaCheck(cudaPointerGetAttributes(&attr, data));
    device = attr.device;
  } else {
    THCudaCheck(cudaGetDevice(&device));
  }
  storage->device = device;
  return storage;
}

void THCZStorage_(setFlag)(THCState *state, THCZStorage *storage, const char flag)
{
  storage->flag |= flag;
}

void THCZStorage_(clearFlag)(THCState *state, THCZStorage *storage, const char flag)
{
  storage->flag &= ~flag;
}

void THCZStorage_(retain)(THCState *state, THCZStorage *self)
{
  if(self && (self->flag & TH_STORAGE_REFCOUNTED))
    THAtomicIncrementRef(&self->refcount);
}

void THCZStorage_(free)(THCState *state, THCZStorage *self)
{
  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (THAtomicDecrementRef(&self->refcount))
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      THCudaCheck(
        (*self->allocator->free)(self->allocatorContext, self->data));
    }
    if(self->flag & TH_STORAGE_VIEW) {
      THCZStorage_(free)(state, self->view);
    }
    THFree(self);
  }
}
#endif
