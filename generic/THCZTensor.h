#ifndef THCZ_GENERIC_FILE
#define THCZ_GENERIC_FILE "generic/THCZTensor.h"
#else

#define TH_TENSOR_REFCOUNTED 1

typedef struct THCZTensor
{
    int64_t *size;
    int64_t *stride;
    int nDimension;

    THCZStorage *storage;
    ptrdiff_t storageOffset;
    int refcount;

    char flag;

} THCZTensor;


/**** access methods ****/
THC_API THCZStorage* THCZTensor_(storage)(THCState *state, const THCZTensor *self);
THC_API ptrdiff_t THCZTensor_(storageOffset)(THCState *state, const THCZTensor *self);
THC_API int THCZTensor_(nDimension)(THCState *state, const THCZTensor *self);
THC_API int64_t THCZTensor_(size)(THCState *state, const THCZTensor *self, int dim);
THC_API int64_t THCZTensor_(stride)(THCState *state, const THCZTensor *self, int dim);
THC_API THLongStorage *THCZTensor_(newSizeOf)(THCState *state, THCZTensor *self);
THC_API THLongStorage *THCZTensor_(newStrideOf)(THCState *state, THCZTensor *self);
THC_API ntype *THCZTensor_(data)(THCState *state, const THCZTensor *self);

THC_API void THCZTensor_(setFlag)(THCState *state, THCZTensor *self, const char flag);
THC_API void THCZTensor_(clearFlag)(THCState *state, THCZTensor *self, const char flag);


/**** creation methods ****/
THC_API THCZTensor *THCZTensor_(new)(THCState *state);
THC_API THCZTensor *THCZTensor_(newWithTensor)(THCState *state, THCZTensor *tensor);
/* stride might be NULL */
THC_API THCZTensor *THCZTensor_(newWithStorage)(THCState *state, THCZStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THC_API THCZTensor *THCZTensor_(newWithStorage1d)(THCState *state, THCZStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_);
THC_API THCZTensor *THCZTensor_(newWithStorage2d)(THCState *state, THCZStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_);
THC_API THCZTensor *THCZTensor_(newWithStorage3d)(THCState *state, THCZStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_,
                                int64_t size2_, int64_t stride2_);
THC_API THCZTensor *THCZTensor_(newWithStorage4d)(THCState *state, THCZStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_,
                                int64_t size2_, int64_t stride2_,
                                int64_t size3_, int64_t stride3_);

/* stride might be NULL */
THC_API THCZTensor *THCZTensor_(newWithSize)(THCState *state, THLongStorage *size_, THLongStorage *stride_);
THC_API THCZTensor *THCZTensor_(newWithSize1d)(THCState *state, int64_t size0_);
THC_API THCZTensor *THCZTensor_(newWithSize2d)(THCState *state, int64_t size0_, int64_t size1_);
THC_API THCZTensor *THCZTensor_(newWithSize3d)(THCState *state, int64_t size0_, int64_t size1_, int64_t size2_);
THC_API THCZTensor *THCZTensor_(newWithSize4d)(THCState *state, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);

THC_API THCZTensor *THCZTensor_(newClone)(THCState *state, THCZTensor *self);
THC_API THCZTensor *THCZTensor_(newContiguous)(THCState *state, THCZTensor *tensor);
THC_API THCZTensor *THCZTensor_(newSelect)(THCState *state, THCZTensor *tensor, int dimension_, int64_t sliceIndex_);
THC_API THCZTensor *THCZTensor_(newNarrow)(THCState *state, THCZTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_);
THC_API THCZTensor *THCZTensor_(newTranspose)(THCState *state, THCZTensor *tensor, int dimension1_, int dimension2_);
THC_API THCZTensor *THCZTensor_(newUnfold)(THCState *state, THCZTensor *tensor, int dimension_, int64_t size_, int64_t step_);
THC_API THCZTensor *THCZTensor_(newView)(THCState *state, THCZTensor *tensor, THLongStorage *size);
THC_API THCZTensor *THCZTensor_(newExpand)(THCState *state, THCZTensor *tensor, THLongStorage *size);

THC_API void THCZTensor_(expand)(THCState *state, THCZTensor *r, THCZTensor *tensor, THLongStorage *sizes);
THC_API void THCZTensor_(expandNd)(THCState *state, THCZTensor **rets, THCZTensor **ops, int count);

THC_API void THCZTensor_(resize)(THCState *state, THCZTensor *tensor, THLongStorage *size, THLongStorage *stride);
THC_API void THCZTensor_(resizeAs)(THCState *state, THCZTensor *tensor, THCZTensor *src);
THC_API void THCZTensor_(resize1d)(THCState *state, THCZTensor *tensor, int64_t size0_);
THC_API void THCZTensor_(resize2d)(THCState *state, THCZTensor *tensor, int64_t size0_, int64_t size1_);
THC_API void THCZTensor_(resize3d)(THCState *state, THCZTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_);
THC_API void THCZTensor_(resize4d)(THCState *state, THCZTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);
THC_API void THCZTensor_(resize5d)(THCState *state, THCZTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_, int64_t size4_);
THC_API void THCZTensor_(resizeNd)(THCState *state, THCZTensor *tensor, int nDimension, int64_t *size, int64_t *stride);

THC_API void THCZTensor_(set)(THCState *state, THCZTensor *self, THCZTensor *src);
THC_API void THCZTensor_(setStorage)(THCState *state, THCZTensor *self, THCZStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THC_API void THCZTensor_(setStorageNd)(THCState *state, THCZTensor *self, THCZStorage *storage, ptrdiff_t storageOffset, int nDimension, int64_t *size, int64_t *stride);
THC_API void THCZTensor_(setStorage1d)(THCState *state, THCZTensor *self, THCZStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_);
THC_API void THCZTensor_(setStorage2d)(THCState *state, THCZTensor *self, THCZStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_);
THC_API void THCZTensor_(setStorage3d)(THCState *state, THCZTensor *self, THCZStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_,
                                    int64_t size2_, int64_t stride2_);
THC_API void THCZTensor_(setStorage4d)(THCState *state, THCZTensor *self, THCZStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_,
                                    int64_t size2_, int64_t stride2_,
                                    int64_t size3_, int64_t stride3_);

THC_API void THCZTensor_(narrow)(THCState *state, THCZTensor *self, THCZTensor *src, int dimension_, int64_t firstIndex_, int64_t size_);
THC_API void THCZTensor_(select)(THCState *state, THCZTensor *self, THCZTensor *src, int dimension_, int64_t sliceIndex_);
THC_API void THCZTensor_(transpose)(THCState *state, THCZTensor *self, THCZTensor *src, int dimension1_, int dimension2_);
THC_API void THCZTensor_(unfold)(THCState *state, THCZTensor *self, THCZTensor *src, int dimension_, int64_t size_, int64_t step_);

THC_API void THCZTensor_(squeeze)(THCState *state, THCZTensor *self, THCZTensor *src);
THC_API void THCZTensor_(squeeze1d)(THCState *state, THCZTensor *self, THCZTensor *src, int dimension_);
THC_API void THCZTensor_(unsqueeze1d)(THCState *state, THCZTensor *self, THCZTensor *src, int dimension_);

THC_API int THCZTensor_(isContiguous)(THCState *state, const THCZTensor *self);
THC_API int THCZTensor_(isSameSizeAs)(THCState *state, const THCZTensor *self, const THCZTensor *src);
THC_API int THCZTensor_(isSetTo)(THCState *state, const THCZTensor *self, const THCZTensor *src);
THC_API int THCZTensor_(isSize)(THCState *state, const THCZTensor *self, const THLongStorage *dims);
THC_API ptrdiff_t THCZTensor_(nElement)(THCState *state, const THCZTensor *self);

THC_API void THCZTensor_(retain)(THCState *state, THCZTensor *self);
THC_API void THCZTensor_(free)(THCState *state, THCZTensor *self);
THC_API void THCZTensor_(freeCopyTo)(THCState *state, THCZTensor *self, THCZTensor *dst);

/* Slow access methods [check everything] */
THC_API void THCZTensor_(set1d)(THCState *state, THCZTensor *tensor, int64_t x0, ntype value);
THC_API void THCZTensor_(set2d)(THCState *state, THCZTensor *tensor, int64_t x0, int64_t x1, ntype value);
THC_API void THCZTensor_(set3d)(THCState *state, THCZTensor *tensor, int64_t x0, int64_t x1, int64_t x2, ntype value);
THC_API void THCZTensor_(set4d)(THCState *state, THCZTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, ntype value);

THC_API ntype THCZTensor_(get1d)(THCState *state, const THCZTensor *tensor, int64_t x0);
THC_API ntype THCZTensor_(get2d)(THCState *state, const THCZTensor *tensor, int64_t x0, int64_t x1);
THC_API ntype THCZTensor_(get3d)(THCState *state, const THCZTensor *tensor, int64_t x0, int64_t x1, int64_t x2);
THC_API ntype THCZTensor_(get4d)(THCState *state, const THCZTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3);

/* CUDA-specific functions */
THC_API cudaTextureObject_t THCZTensor_(getTextureObject)(THCState *state, THCZTensor *self);
THC_API int THCZTensor_(getDevice)(THCState *state, const THCZTensor *self);
THC_API int THCZTensor_(checkGPU)(THCState *state, unsigned int nTensors, ...);

/* debug methods */
THC_API THCDescBuff THCZTensor_(sizeDesc)(THCState *state, const THCZTensor *tensor);

#endif
