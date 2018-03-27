#ifndef THCZ_GENERIC_FILE
#define THCZ_GENERIC_FILE "generic/THCZStorage.h"
#else

#define TH_STORAGE_REFCOUNTED 1
#define TH_STORAGE_RESIZABLE  2
#define TH_STORAGE_FREEMEM    4

typedef struct THCZStorage
{
    ntype *data;
    ptrdiff_t size;
    int refcount;
    char flag;
    THCDeviceAllocator *allocator;
    void *allocatorContext;
    struct THCZStorage *view;
    int device;
} THCZStorage;


THC_API ntype* THCZStorage_(data)(THCState *state, const THCZStorage*);
THC_API ptrdiff_t THCZStorage_(size)(THCState *state, const THCZStorage*);
THC_API int THCZStorage_(elementSize)(THCState *state);

/* slow access -- checks everything */
THC_API void THCZStorage_(set)(THCState *state, THCZStorage*, ptrdiff_t, ntype);
THC_API ntype THCZStorage_(get)(THCState *state, const THCZStorage*, ptrdiff_t);

THC_API THCZStorage* THCZStorage_(new)(THCState *state);
THC_API THCZStorage* THCZStorage_(newWithSize)(THCState *state, ptrdiff_t size);
THC_API THCZStorage* THCZStorage_(newWithSize1)(THCState *state, ntype);
THC_API THCZStorage* THCZStorage_(newWithSize2)(THCState *state, ntype, ntype);
THC_API THCZStorage* THCZStorage_(newWithSize3)(THCState *state, ntype, ntype, ntype);
THC_API THCZStorage* THCZStorage_(newWithSize4)(THCState *state, ntype, ntype, ntype, ntype);
THC_API THCZStorage* THCZStorage_(newWithMapping)(THCState *state, const char *filename, ptrdiff_t size, int shared);

/* takes ownership of data */
THC_API THCZStorage* THCZStorage_(newWithData)(THCState *state, ntype *data, ptrdiff_t size);

THC_API THCZStorage* THCZStorage_(newWithAllocator)(
  THCState *state, ptrdiff_t size,
  THCDeviceAllocator* allocator,
  void *allocatorContext);
THC_API THCZStorage* THCZStorage_(newWithDataAndAllocator)(
  THCState *state, ntype* data, ptrdiff_t size,
  THCDeviceAllocator* allocator,
  void *allocatorContext);

THC_API void THCZStorage_(setFlag)(THCState *state, THCZStorage *storage, const char flag);
THC_API void THCZStorage_(clearFlag)(THCState *state, THCZStorage *storage, const char flag);
THC_API void THCZStorage_(retain)(THCState *state, THCZStorage *storage);

THC_API void THCZStorage_(free)(THCState *state, THCZStorage *storage);
THC_API void THCZStorage_(resize)(THCState *state, THCZStorage *storage, ptrdiff_t size);
THC_API void THCZStorage_(fill)(THCState *state, THCZStorage *storage, ntype value);

THC_API int THCZStorage_(getDevice)(THCState* state, const THCZStorage* storage);

#endif
