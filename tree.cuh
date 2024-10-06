#include"buffer.cuh"
#include<limits>
#include<vector>
#include<iostream>
#include<memory>
#include<unordered_map>
namespace Sloth
{
#define SLOTH_DEBUG_ENABLED

    namespace TreeInternalKernels
    {

        static constexpr int iMin = std::numeric_limits<int>::min();
        static constexpr int iMax = std::numeric_limits<int>::max();



        // each task is 1 block of threads
        // each chunk can have multiple tasks
        static constexpr int taskThreads = 128;
        static constexpr int nodeElements = 128;
        static constexpr int nodeMaxDepth = 10;
        static constexpr int numChildNodesPerParent =4;
        
       

        template<typename Type>
        struct Reducer
        {
            __device__ Type BlockSum(const int id, const Type val, Type* sm)
            {
                sm[id] = val;
                __syncthreads();
                // shared reduction
                for (unsigned int i = ((unsigned int)taskThreads) >> 1; i > 16; i >>= 1)
                {
                    if (id < i)
                        sm[id] += sm[id + i];
                    __syncthreads();
                }
                Type result;
                if (id < 32)
                {
                    result = sm[id];
                    // warp reduction
                    for (unsigned int i = 16; i > 0; i >>= 1)
                    {
                        result += __shfl_sync(0xffffffff, result, i + (id & 31));
                    }
                }
                __syncthreads();
                if (id == 0)
                    sm[0] = result;
                __syncthreads();
                result = sm[0];
                __syncthreads();
                return result;
            }

            __device__ Type BlockMin(const int id, const Type val, Type* sm)
            {
                sm[id] = val;
                __syncthreads();
                for (unsigned int i = ((unsigned int)taskThreads) >> 1; i > 0; i >>= 1)
                {
                    if (id < i)
                    {
                        const Type a = sm[id];
                        const Type b = sm[id + i];
                        if (a > b)
                        {
                            sm[id] = b;
                            sm[id + i] = a;
                        }
                    }
                    __syncthreads();
                }
                Type result = sm[0];
                __syncthreads();
                return result;
            }

            __device__ Type BlockMax(const int id, const Type val, Type* sm)
            {
                sm[id] = val;
                __syncthreads();
                for (unsigned int i = ((unsigned int)taskThreads) >> 1; i > 0; i >>= 1)
                {
                    if (id < i)
                    {
                        const Type a = sm[id];
                        const Type b = sm[id + i];
                        if (a < b)
                        {
                            sm[id] = b;
                            sm[id + i] = a;
                        }
                    }
                    __syncthreads();
                }
                Type result = sm[0];
                __syncthreads();
                return result;
            }
        };

        template<typename Type, typename TypeMask>
        struct StreamCompacter
        {
            __device__ Type BlockCompact(Type val, bool mask, Type* sm, Type* sm2, TypeMask* smMask, const int id)
            {

                sm[id] = val;
                smMask[id + 1] = mask;
                if (id == 0)
                {
                    smMask[0] = 0;
                    smMask[taskThreads + 1] = 0;
                }
                __syncthreads();
                int gatherDistance = 1;

                while (gatherDistance < taskThreads)
                {
                    TypeMask msk = smMask[id + 1];
                    if (id + 1 - gatherDistance >= 0)
                        msk += smMask[id + 1 - gatherDistance];
                    __syncthreads();
                    smMask[id + 1] = msk;
                    gatherDistance <<= 1;
                    __syncthreads();
                }

                __syncthreads();

                if (smMask[id] == smMask[id + 1] - 1)
                {
                    sm2[smMask[id + 1] - 1] = sm[id];
                }

                __syncthreads();
                Type result = sm2[id];
                __syncthreads();
                return result;
            }

            template<typename Type2>
            __device__ Type BlockCompactKeyValue(
                Type key, Type value, 
                bool mask, 
                Type* smKey, Type* smKey2, 
                Type* smValue, Type* smValue2,
                TypeMask* smMask, 
                const int id,
                Type * resultKey,
                Type2 * resultVal)
            {

                smKey[id] = key;
                smValue[id] = value;
                smMask[id + 1] = mask;
                if (id == 0)
                {
                    smMask[0] = 0;
                    smMask[taskThreads + 1] = 0;
                }
                __syncthreads();
                int gatherDistance = 1;

                while (gatherDistance < taskThreads)
                {
                    TypeMask msk = smMask[id + 1];
                    if (id + 1 - gatherDistance >= 0)
                        msk += smMask[id + 1 - gatherDistance];
                    __syncthreads();
                    smMask[id + 1] = msk;
                    gatherDistance <<= 1;
                    __syncthreads();
                }

                __syncthreads();

                if (smMask[id] == smMask[id + 1] - 1)
                {
                    smKey2[smMask[id + 1] - 1] = smKey[id];
                    smValue2[smMask[id + 1] - 1] = smValue[id];
                }

                __syncthreads();
                *resultKey = smKey2[id];
                *resultVal = smValue2[id];
                __syncthreads();

            }

            template<typename Type2>
            __device__ void BlockCompact2(
                Type val, Type2 val0, 
                bool mask, 
                Type* sm, Type2* sm0, 
                Type* sm2, Type2* sm20,
                TypeMask* smMask, 
                const int id, 
                Type* out1, Type2* out2, 
                char* smChunkTracking, char* smChunkTracking2, 
                char chunkId, char* outChunk)
            {

                sm[id] = val;
                sm0[id] = val0;
                smMask[id + 1] = mask;
                smChunkTracking[id] = chunkId;
                if (id == 0)
                {
                    smMask[0] = 0;
                    smMask[taskThreads + 1] = 0;
                }
                __syncthreads();
                int gatherDistance = 1;

                while (gatherDistance < taskThreads)
                {
                    TypeMask msk = smMask[id + 1];
                    if (id + 1 - gatherDistance >= 0)
                        msk += smMask[id + 1 - gatherDistance];
                    __syncthreads();
                    smMask[id + 1] = msk;
                    gatherDistance <<= 1;
                    __syncthreads();
                }

                __syncthreads();

                const int target = smMask[id + 1] - 1;
                if (smMask[id] == target)
                {
                    sm2[target] = sm[id];
                    sm20[target] = sm0[id];
                    smChunkTracking2[target] = smChunkTracking[id];
                }

                __syncthreads();
                Type result = sm2[id];
                Type2 result2 = sm20[id];
                char resultChunk = smChunkTracking2[id];
                __syncthreads();
                *out1 = result;
                *out2 = result2;
                *outChunk = resultChunk;
            }
        };





        // work: single integer to indicate current chunk to compute
        template<typename KeyType>
        __global__ void minMaxReduction(KeyType* key, int* work, const int numKeys, KeyType* minMaxData)
        {
            const int tid = threadIdx.x;
            const int gid = blockIdx.x;
            const int gs = blockDim.x;
            KeyType keyMin = iMax;
            KeyType keyMax = iMin;
            __shared__ int wo;
            while (true)
            {
                __syncthreads();
                if (tid == 0)
                    wo = atomicAdd(&work[0], gs);
                __syncthreads();
                const int workOffset = wo;
                if (workOffset >= numKeys)
                    break;

                if (workOffset + tid < numKeys)
                {
                    const KeyType keyData = key[workOffset + tid];
                    if (keyMin > keyData)
                        keyMin = keyData;
                    if (keyMax < keyData)
                        keyMax = keyData;
                }
            }
            __shared__ KeyType sm[taskThreads];
            Reducer<KeyType> reducer;
            const KeyType minData = reducer.BlockMin(tid, keyMin, sm);
            const KeyType maxData = reducer.BlockMax(tid, keyMax, sm);
            if (tid == 0)
            {
                atomicMin(&minMaxData[0], minData);
                atomicMax(&minMaxData[1], maxData);
            }
        }

        // single global load operation
        // broadcasted from shared memory to all threads in block
        template<typename Type>
        __device__ Type loadSingle(Type * ptr,Type * ptrSm, const int id)
        {
            if (id == 0)
            {
                *ptrSm = *ptr;
            }
            __syncthreads();
            const Type result = *ptrSm;
            __syncthreads();
            return result;
        }


        template<typename KeyType, typename ValueType>
        __global__ void resetChunkLength(
            int* taskIdIn,
            int* taskParallelismIn,
            int* taskInCounter, int* taskInChunkId, char * chunkDepth,
            int* chunkLength, int* chunkProgress, int* chunkOffset,
            KeyType* chunkRangeMin, KeyType* chunkRangeMax, int* chunkCounter,char * chunkType,
            KeyType* keyIn, ValueType* valueIn, KeyType* keyOut, ValueType* valueOut,
            int* taskOutCounter,
            int* debugBuffer
        )
        {

            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int bs = blockDim.x;
            const int gs = gridDim.x;
            const int thread = tid + bid * bs;
            __shared__ int smLoadInt[1];
            __shared__ KeyType smLoadKey[1];
            __shared__ char smLoadChar[1];

            const int chunkTasks = loadSingle(taskParallelismIn + bid, smLoadInt, tid); // 1 block per task
            const int chunkId = loadSingle(taskInChunkId + bid, smLoadInt, tid);
            const int totalWorkSize = loadSingle(chunkLength + chunkId, smLoadInt, tid);
            const int chunkOfs = loadSingle(chunkOffset + chunkId, smLoadInt, tid); 
            const KeyType chunkMin = loadSingle(chunkRangeMin + chunkId, smLoadKey, tid);
            const KeyType chunkMax = loadSingle(chunkRangeMax + chunkId, smLoadKey, tid);
            const char chkDepth = loadSingle(chunkDepth + chunkId, smLoadChar, tid);

            // if parent range is [0,640] and if there are 64 children per node, then each child will have 10 sized range, [0,9],[10,19],...
            const int chunkChildRange = ((double)chunkMax - (double)chunkMin) / numChildNodesPerParent;

    

            // chunk is processed in multiple leaps of this stride 
            const int chunkStride = bs * chunkTasks;
            const int strideThreadId = tid + taskIndex * bs; // this allows variable amount of tasks per chunk within same kernel
            const int numStrides = 1 + (totalWorkSize - 1) / chunkStride;
            
            const int childChunkIndexStart = chunkId * numChildNodesPerParent + 1;
            // if a chunk is being processed, it is at least a node (value of 2 means leaf node)
            if (strideThreadId == 0)
            {
                
                chunkType[chunkId] = 1;
                for (int j = 0; j < numChildNodesPerParent; j++)
                {
                    const int childChunkIndex = childChunkIndexStart + j;
                    chunkLength[childChunkIndex] = 0; 
                }
            }
        }


        /* 
            pseudo-allocation on array requires a measurement
            counts children elements required to copy
        */ 
        template<typename KeyType, typename ValueType>
        __global__ void computeChildChunkAllocationRequirements(
            int* taskIdIn,
            int* taskParallelismIn,
            int * taskInCounter, int * taskInChunkId, char* chunkDepth,
            int * chunkLength, int * chunkProgress,  int * chunkOffset,
            KeyType * chunkRangeMin, KeyType * chunkRangeMax, int* chunkCounter, char* chunkType,
            KeyType * keyIn, ValueType * valueIn, KeyType * keyOut, ValueType * valueOut,
            int * taskOutCounter,
            int * debugBuffer
        )
        {
            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int bs = blockDim.x;
            const int gs = gridDim.x;
            const int thread = tid + bid * bs;
            __shared__ int smLoadInt[1];
            __shared__ KeyType smLoadKey[1];
            __shared__ char smLoadChar[1];

            // task index is linearly increasing within each chunk. maximum number of blocks per chunk
            const int taskIndex = loadSingle(taskIdIn + bid, smLoadInt, tid); // 1 block per task
            const int chunkTasks = loadSingle(taskParallelismIn + bid, smLoadInt, tid); // 1 block per task
            const int chunkId = loadSingle(taskInChunkId + bid, smLoadInt, tid);
            const int totalWorkSize = loadSingle(chunkLength + chunkId, smLoadInt, tid);
            const int chunkOfs = loadSingle(chunkOffset + chunkId, smLoadInt, tid);
            const KeyType chunkMin = loadSingle(chunkRangeMin + chunkId, smLoadKey, tid);
            const KeyType chunkMax = loadSingle(chunkRangeMax + chunkId, smLoadKey, tid);
            const char chkDepth = loadSingle(chunkDepth + chunkId, smLoadChar, tid);

            // if parent range is [0,640] and if there are 64 children per node, then each child will have 10 sized range, [0,9],[10,19],...
            const int chunkChildRange = ((double)chunkMax - (double)chunkMin) / numChildNodesPerParent;


            // if ranges are too small or if it is a leaf-node, do not create child nodes
            if ((chunkChildRange < 1) || (totalWorkSize <= nodeElements) || (chkDepth > nodeMaxDepth))
                return;

            // chunk is processed in multiple leaps of this stride 
            const int chunkStride = bs * chunkTasks;
            const int strideThreadId = tid + taskIndex * bs; // this allows variable amount of tasks per chunk within same kernel
            const int numStrides = 1 + (totalWorkSize - 1) / chunkStride;


            // each child-node has its own key range to compare so every deeper node will have a closer number to target when searching a number
            // inclusive values
            KeyType childNodeRangeMin[numChildNodesPerParent];
            KeyType childNodeRangeMax[numChildNodesPerParent];
            // preparing min-max range per child
            KeyType curBegin = chunkMin;

            for (int i = 0; i < numChildNodesPerParent; i++)
            {
                int curEnd = curBegin + chunkChildRange;
                if (i == numChildNodesPerParent - 1)
                    curEnd = chunkMax;
                childNodeRangeMin[i] = curBegin;
                childNodeRangeMax[i] = curEnd;
                curBegin = curEnd + 1;
            }

            
            Reducer<int> reducerInt;
            __shared__ int smReductionInt[taskThreads];

            
            // sum in registers
            int newElementFound[numChildNodesPerParent];
            for (int j = 0; j < numChildNodesPerParent; j++)
                newElementFound[j] = 0;

            for (int i = 0; i < numStrides; i++)
            {
                const int currentId = strideThreadId + i * chunkStride;
                KeyType key;
                if (currentId < totalWorkSize)
                {
                    key = keyIn[currentId + chunkOfs];
                    for (int j = 0; j < numChildNodesPerParent; j++)
                        newElementFound[j] += (key >= childNodeRangeMin[j] && key <= childNodeRangeMax[j]);
                }
            }

            // uses complete-tree definition to traverse tree without gaps. it has all children or none.
            const int childChunkIndexStart = chunkId * numChildNodesPerParent + 1;
            for (int j = 0; j < numChildNodesPerParent; j++)
            {
                // chunkId: zero based. tree-traversal: one based +1 -1
                const int childChunkIndex = childChunkIndexStart + j;
                // shared reduction
                const int sum = reducerInt.BlockSum(tid, newElementFound[j], smReductionInt);

                if (tid == 0)
                {
                    // global reduction
                    atomicAdd(&chunkLength[childChunkIndex], sum);

                }

                // only 1 thread of chunk is resetting this value for later kernels
                if (strideThreadId == 0)
                {
                    chunkCounter[childChunkIndex] = 0;
                }
            }
        }


        template<typename KeyType, typename ValueType>
        __global__ void computeChildChunkOffset(
            int* taskIdIn,
            int* taskParallelismIn,
            int* taskInCounter, int* taskInChunkId, char* chunkDepth,
            int* chunkLength, int* chunkProgress,  int* chunkOffset,
            KeyType* chunkRangeMin, KeyType* chunkRangeMax, int * chunkCounter , char* chunkType,
            KeyType* keyIn, ValueType* valueIn, KeyType* keyOut, ValueType* valueOut,
            int* taskOutCounter,
            int* debugBuffer
        )
        {

            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int bs = blockDim.x;
            const int gs = gridDim.x;
            const int thread = tid + bid * bs;
            __shared__ int smLoadInt[1];
            __shared__ KeyType smLoadKey[1];
            __shared__ char smLoadChar[1];

            // task index is linearly increasing within each chunk. maximum number of blocks per chunk
            const int taskIndex = loadSingle(taskIdIn + bid, smLoadInt, tid); // 1 block per task
            const int chunkTasks = loadSingle(taskParallelismIn + bid, smLoadInt, tid); // 1 block per task
            const int chunkId = loadSingle(taskInChunkId + bid, smLoadInt, tid);
            const int totalWorkSize = loadSingle(chunkLength + chunkId, smLoadInt, tid);
            const int chunkOfs = loadSingle(chunkOffset + chunkId, smLoadInt, tid);
            const KeyType chunkMin = loadSingle(chunkRangeMin + chunkId, smLoadKey, tid);
            const KeyType chunkMax = loadSingle(chunkRangeMax + chunkId, smLoadKey, tid);
            const char chkDepth = loadSingle(chunkDepth + chunkId, smLoadChar, tid);

            // if parent range is [0,640] and if there are 64 children per node, then each child will have 10 sized range, [0,9],[10,19],...
            const int chunkChildRange = ((double)chunkMax - (double)chunkMin) / numChildNodesPerParent;

            // if ranges are too small or if it is a leaf-node, do not create child nodes
            if ((chunkChildRange < 1) || (totalWorkSize <= nodeElements) || (chkDepth > nodeMaxDepth))
                return;

            // chunk is processed in multiple leaps of this stride 
            const int chunkStride = bs * chunkTasks;
            const int strideThreadId = tid + taskIndex * bs; // this allows variable amount of tasks per chunk within same kernel
            const int numStrides = 1 + (totalWorkSize - 1) / chunkStride;


            // calculate offsets of child chunks  from their sizes, within parent's region
            int runningSum = 0;
            // uses complete-tree definition to traverse tree without gaps. it has all children or none.
            const int childChunkIndexStart = chunkId * numChildNodesPerParent + 1;
            if (strideThreadId == 0)
            {
                for (int j = 0; j < numChildNodesPerParent; j++)
                {
                    // chunkId: zero based. tree-traversal: one based +1 -1
                    const int childChunkIndex = childChunkIndexStart + j;
                    // computed child offset + parent offset = absolute child offset
                    chunkOffset[childChunkIndex] = runningSum + chunkOfs;
                    runningSum += chunkLength[childChunkIndex]; // todo: optimize this with shared memory or registers
                }
            }
        }


        template<typename KeyType, typename ValueType>
        __global__ void allocateChildChunkAndCopy(
            int* taskIdIn,
            int* taskParallelismIn,
            int* taskInCounter, int* taskInChunkId, char* chunkDepth,
            int* chunkLength, int* chunkProgress, int* chunkOffset,
            KeyType* chunkRangeMin, KeyType* chunkRangeMax, int * chunkCounter, char * chunkType,
            KeyType* keyIn, ValueType* valueIn, KeyType* keyOut, ValueType* valueOut,
            int* taskOutCounter,
            int* debugBuffer
        )
        {
          
            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int bs = blockDim.x;
            const int gs = gridDim.x;
            const int thread = tid + bid * bs;
            __shared__ int smLoadInt[1];
            __shared__ KeyType smLoadKey[1];
            __shared__ char smLoadChar[1];

            // task index is linearly increasing within each chunk. maximum number of blocks per chunk
            const int taskIndex = loadSingle(taskIdIn + bid, smLoadInt, tid); // 1 block per task
            const int chunkTasks = loadSingle(taskParallelismIn + bid, smLoadInt, tid); // 1 block per task
            const int chunkId = loadSingle(taskInChunkId + bid, smLoadInt, tid);
            const int totalWorkSize = loadSingle(chunkLength + chunkId, smLoadInt, tid);
            const int chunkOfs = loadSingle(chunkOffset + chunkId, smLoadInt, tid);
            const KeyType chunkMin = loadSingle(chunkRangeMin + chunkId, smLoadKey, tid);
            const KeyType chunkMax = loadSingle(chunkRangeMax + chunkId, smLoadKey, tid);
            const char chkDepth = loadSingle(chunkDepth + chunkId, smLoadChar, tid);

            // if parent range is [0,640] and if there are 64 children per node, then each child will have 10 sized range, [0,9],[10,19],...
            const int chunkChildRange = ((double)chunkMax - (double)chunkMin) / numChildNodesPerParent;


            
            // if ranges are too small or if it is a leaf-node, do not create child nodes
            if ((chunkChildRange < 1) || (totalWorkSize <= nodeElements) || (chkDepth > nodeMaxDepth))
                return;

            // chunk is processed in multiple leaps of this stride 
            const int chunkStride = bs * chunkTasks;
            const int strideThreadId = tid + taskIndex * bs; // this allows variable amount of tasks per chunk within same kernel
            const int numStrides = 1 + (totalWorkSize - 1) / chunkStride;


            // each child-node has its own key range to compare so every deeper node will have a closer number to target when searching a number
            // inclusive values
            KeyType childNodeRangeMin[numChildNodesPerParent];
            KeyType childNodeRangeMax[numChildNodesPerParent];
            // preparing min-max range per child
            KeyType curBegin = chunkMin;

            for (int i = 0; i < numChildNodesPerParent; i++)
            {
                int curEnd = curBegin + chunkChildRange;
                if (i == numChildNodesPerParent - 1)
                    curEnd = chunkMax;
                childNodeRangeMin[i] = curBegin;
                childNodeRangeMax[i] = curEnd;
                curBegin = curEnd + 1;
            }

            int absoluteChildChunkOffsets[numChildNodesPerParent];
            const int childChunkIndexStart = chunkId * numChildNodesPerParent + 1;
            for (int j = 0; j < numChildNodesPerParent; j++)
            {
                // chunkId: zero based. tree-traversal: one based +1 -1
                const int childChunkIndex = childChunkIndexStart + j;
                absoluteChildChunkOffsets[j] = chunkOffset[childChunkIndex];
            }

            Reducer<int> reducerInt;
            __shared__ int smReductionInt[taskThreads];

            StreamCompacter<KeyType, int> compacter;
            __shared__ KeyType smCompactionKey[taskThreads + 2];
            __shared__ ValueType smCompactionValue[taskThreads + 2];
            __shared__ KeyType smCompactionKey2[taskThreads + 2];
            __shared__ ValueType smCompactionValue2[taskThreads + 2];
            __shared__ int smMaskCompaction[taskThreads + 2];

            __shared__ int smTargetIndex;

            for (int i = 0; i < numStrides; i++)
            {
                const int currentId = strideThreadId + i * chunkStride;
                int newElementFound[numChildNodesPerParent];
                for (int j = 0; j < numChildNodesPerParent; j++)
                    newElementFound[j] = 0;
                KeyType key;
                ValueType value;
                if (currentId < totalWorkSize)
                {
                    key = keyIn[currentId + chunkOfs];
                    value = valueIn[currentId + chunkOfs];

                    for (int j = 0; j < numChildNodesPerParent; j++)
                    {
                        if (key >= childNodeRangeMin[j] && key <= childNodeRangeMax[j])
                            newElementFound[j] = (key >= chunkMin && key <= chunkMax);
                    }
                }
             
                for (int j = 0; j < numChildNodesPerParent; j++)
                {
                    const int nCompacted = reducerInt.BlockSum(tid, newElementFound[j], smReductionInt);


                    KeyType keyCompacted;
                    ValueType valueCompacted;
                    compacter.BlockCompactKeyValue(
                        key, value,
                        newElementFound[j],
                        smCompactionKey, smCompactionKey2,
                        smCompactionValue, smCompactionValue2,
                        smMaskCompaction,
                        tid,
                        &keyCompacted,
                        &valueCompacted);


                    const int targetOffset = absoluteChildChunkOffsets[j];
                    const int childChunkIndex = childChunkIndexStart + j;
                    if (tid == 0)
                    {
                        smTargetIndex = atomicAdd(&chunkCounter[childChunkIndex], nCompacted); // reducing global atomics by nCompacted times
                    }
                    __syncthreads();
                    const int targetIndex = smTargetIndex;

                    if (tid < nCompacted)
                    {                        
                       keyOut[tid + targetOffset + targetIndex]=keyCompacted;
                       valueOut[tid + targetOffset + targetIndex]=valueCompacted;
                    }
                }

               
            }

            // if made it this far, it is a node with children
            if (strideThreadId == 0)
            {
                chunkType[chunkId] = 2;
            }
            
        }

        template<typename KeyType, typename ValueType>
        __global__ void copyChunkBack(           
            int* taskIdIn,
            int* taskParallelismIn,
            int* taskInCounter, int* taskInChunkId, char* chunkDepth,
            int* chunkLength, int* chunkProgress,  int* chunkOffset,
            KeyType* chunkRangeMin, KeyType* chunkRangeMax, int* chunkCounter, char* chunkType,
            KeyType* keyIn, ValueType* valueIn, KeyType* keyOut, ValueType* valueOut,
            int* taskOutCounter,
            int* debugBuffer
        )
        {
            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int bs = blockDim.x;
            const int gs = gridDim.x;
            const int thread = tid + bid * bs;
            __shared__ int smLoadInt[1];
            __shared__ KeyType smLoadKey[1];
            __shared__ char smLoadChar[1];

            // task index is linearly increasing within each chunk. maximum number of blocks per chunk
            const int taskIndex = loadSingle(taskIdIn + bid, smLoadInt, tid); // 1 block per task
            const int chunkTasks = loadSingle(taskParallelismIn + bid, smLoadInt, tid); // 1 block per task
            const int chunkId = loadSingle(taskInChunkId + bid, smLoadInt, tid);
            const int totalWorkSize = loadSingle(chunkLength + chunkId, smLoadInt, tid);
            const int chunkOfs = loadSingle(chunkOffset + chunkId, smLoadInt, tid);
            const KeyType chunkMin = loadSingle(chunkRangeMin + chunkId, smLoadKey, tid);
            const KeyType chunkMax = loadSingle(chunkRangeMax + chunkId, smLoadKey, tid);
            const char chkDepth = loadSingle(chunkDepth + chunkId, smLoadChar, tid);

            const int chunkLen = loadSingle(chunkLength + chunkId, smLoadInt, tid);

            // if parent range is [0,640] and if there are 64 children per node, then each child will have 10 sized range, [0,9],[10,19],...
            const int chunkChildRange = ((double)chunkMax - (double)chunkMin) / numChildNodesPerParent;



            // if ranges are too small or if it is a leaf-node, do not create child nodes
            if ((chunkChildRange < 1) || (totalWorkSize <= nodeElements) || (chkDepth > nodeMaxDepth))
                return;

            // chunk is processed in multiple leaps of this stride 
            const int chunkStride = bs * chunkTasks;
            const int strideThreadId = tid + taskIndex * bs; // this allows variable amount of tasks per chunk within same kernel
            const int numStrides = 1 + (totalWorkSize - 1) / chunkStride;

            if (strideThreadId == 0)
            {
                const int childChunkIndexStart = chunkId * numChildNodesPerParent + 1;
                
                for (int i = 0; i < numChildNodesPerParent; i++)
                {
                    const int childChunkIndex = childChunkIndexStart + i;
                    const int copyStart = chunkOffset[childChunkIndex];
                    const int copyLen = chunkLength[childChunkIndex];
                    for (int j = 0; j < copyLen; j++)
                    {
                        keyIn[copyStart + j] = keyOut[copyStart + j];
                        valueIn[copyStart + j] = valueOut[copyStart + j];
                        //atomicAdd(&debugBuffer[0], 1);
                    }
                }


            }
        }


        // for each children created, add 1 task. adds to task counter atomically so tasks are not sorted but they keep id values to reach their target
        template<typename KeyType, typename ValueType>
        __global__ void createTask(
            int* taskIdIn,
            int * taskParallelismIn,
            int* taskInCounter, int* taskInChunkId, char* chunkDepth,
            int* chunkLength, int* chunkProgress,  int* chunkOffset,
            KeyType* chunkRangeMin, KeyType* chunkRangeMax, int* chunkCounter, char* chunkType,
            KeyType* keyIn, ValueType* valueIn, KeyType* keyOut, ValueType* valueOut,
            int * taskOutCounter,
            int * taskOutChunkId,
            int * taskParallelismOut,
            int * taskIndexOut,
            int* debugBuffer
        )
        {

            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int bs = blockDim.x;
            const int gs = gridDim.x;
            const int thread = tid + bid * bs;
            __shared__ int smLoadInt[1];
            __shared__ KeyType smLoadKey[1];
            __shared__ char smLoadChar[1];

            // task index is linearly increasing within each chunk. maximum number of blocks per chunk
            const int taskIndex = loadSingle(taskIdIn + bid, smLoadInt, tid); // 1 block per task
            const int chunkTasks = loadSingle(taskParallelismIn + bid, smLoadInt, tid); // 1 block per task
            const int chunkId = loadSingle(taskInChunkId + bid, smLoadInt, tid);
            const int totalWorkSize = loadSingle(chunkLength + chunkId, smLoadInt, tid);
            const int chunkOfs = loadSingle(chunkOffset + chunkId, smLoadInt, tid);
            const KeyType chunkMin = loadSingle(chunkRangeMin + chunkId, smLoadKey, tid);
            const KeyType chunkMax = loadSingle(chunkRangeMax + chunkId, smLoadKey, tid);
            const char chkDepth = loadSingle(chunkDepth + chunkId, smLoadChar, tid);

            // if parent range is [0,640] and if there are 64 children per node, then each child will have 10 sized range, [0,9],[10,19],...
            const int chunkChildRange = ((double)chunkMax - (double)chunkMin) / numChildNodesPerParent;



            // if ranges are too small or if it is a leaf-node, do not create child nodes
            if ((chunkChildRange < 1) || (totalWorkSize <= nodeElements) || (chkDepth > nodeMaxDepth))
                return;

            // chunk is processed in multiple leaps of this stride 
            const int chunkStride = bs * chunkTasks;
            const int strideThreadId = tid + taskIndex * bs; // this allows variable amount of tasks per chunk within same kernel
            const int numStrides = 1 + (totalWorkSize - 1) / chunkStride;

            // each child-node has its own key range to compare so every deeper node will have a closer number to target when searching a number
            // inclusive values
            KeyType childNodeRangeMin[numChildNodesPerParent];
            KeyType childNodeRangeMax[numChildNodesPerParent];
            // preparing min-max range per child
            KeyType curBegin = chunkMin;

            for (int i = 0; i < numChildNodesPerParent; i++)
            {
                int curEnd = curBegin + chunkChildRange;
                if (i == numChildNodesPerParent - 1)
                    curEnd = chunkMax;
                childNodeRangeMin[i] = curBegin;
                childNodeRangeMax[i] = curEnd;
                curBegin = curEnd + 1;
            }

            const int childChunkIndexStart = chunkId * numChildNodesPerParent + 1;

            // create tasks for children
            if (strideThreadId == 0)
            {
                const int newTaskIdOfs = atomicAdd(&taskOutCounter[0], numChildNodesPerParent);
                for (int i = 0; i < numChildNodesPerParent; i++)
                {
                    const int childChunkIndex = childChunkIndexStart + i;

                    const int newTaskId = i + newTaskIdOfs;
                    taskOutChunkId[newTaskId] = childChunkIndex;
                    taskParallelismOut[newTaskId] = 1; // todo: loadbalance this with decreasing on more depth
                    taskIndexOut[newTaskId] = i;
                    chunkDepth[childChunkIndex] = chkDepth + 1;
                    chunkRangeMin[childChunkIndex] = childNodeRangeMin[i];
                    chunkRangeMax[childChunkIndex] = childNodeRangeMax[i];
                    chunkCounter[childChunkIndex] = 0;
                    chunkType[childChunkIndex] = 1;
                     
                    //atomicAdd(&debugBuffer[0], chunkLength[childChunkIndex]);
                    
                }
            }
        }

      


        __global__ void resetDebugBuffer(int * debugBuffer)
        {
            debugBuffer[threadIdx.x + blockIdx.x * blockDim.x] = 0;
        }

  
    }


    template<typename KeyType, typename ValueType>
	struct Tree
	{
	private:

        // 1 task = 1 block of threads
        // all in same kernel
        // 1 chunk = N tasks when N tasks have same id
        // chunk = tree node
        // keyin/out valuein/out are elements
        std::shared_ptr<Sloth::Buffer<int>> taskInCounter;
        std::shared_ptr<Sloth::Buffer<int>> taskIndexIn;
        std::shared_ptr<Sloth::Buffer<int>> taskIndexOut;
        std::shared_ptr<Sloth::Buffer<int>> taskInChunkId;
        std::shared_ptr<Sloth::Buffer<int>> taskOutCounter;
        std::shared_ptr<Sloth::Buffer<int>> taskOutChunkId;
        std::shared_ptr<Sloth::Buffer<int>> taskParallelismIn;
        std::shared_ptr<Sloth::Buffer<int>> taskParallelismOut;
       


        // 0: empty, 1: leaf node, 2: node with children
        std::shared_ptr<Sloth::Buffer<char>> chunkType;
        std::shared_ptr<Sloth::Buffer<char>> chunkDepth; // can't have more than 127 depth in any case
        std::shared_ptr<Sloth::Buffer<int>> chunkCounter;

        std::shared_ptr<Sloth::Buffer<int>> chunkOffset;
        std::shared_ptr<Sloth::Buffer<int>> chunkLength;
        std::shared_ptr<Sloth::Buffer<int>> chunkProgress; // tasks atomically use this to do work-stealing between each other for dynamic-load-balancing
        std::shared_ptr<Sloth::Buffer<KeyType>> chunkRangeMin;
        std::shared_ptr<Sloth::Buffer<KeyType>> chunkRangeMax;


        std::shared_ptr<Sloth::Buffer<KeyType>> keyIn;
        std::shared_ptr<Sloth::Buffer<ValueType>> valueIn;

        std::shared_ptr<Sloth::Buffer<KeyType>> keyOut;
        std::shared_ptr<Sloth::Buffer<ValueType>> valueOut;


        std::shared_ptr<Sloth::Buffer<int>> minMaxWorkCounter;
        std::shared_ptr<Sloth::Buffer<int>> inputMinMax;
        std::shared_ptr<Sloth::Buffer<int>> debugBuffer;
        int maxTasks;
        int maxChunks;
        int lastInputSize;
        int allocSize;
        int debugBufferSize;
	public:
		Tree()
		{
            maxTasks = 1024 * 1024;
            debugBufferSize = 0;
            maxChunks = 1 + std::pow(TreeInternalKernels::numChildNodesPerParent, TreeInternalKernels::nodeMaxDepth);
            cudaSetDevice(0);

            lastInputSize = 0;

            inputMinMax = std::make_shared< Sloth::Buffer<int>>("inputMinMax",2, 0, false);
            minMaxWorkCounter = std::make_shared< Sloth::Buffer<int>>("minMaxCounter", 1, 0, false);

            taskInCounter = std::make_shared< Sloth::Buffer<int>>("taskInCounter", 1, 0, false);
            taskOutCounter = std::make_shared< Sloth::Buffer<int>>("taskOutCounter", 1, 0, false);

            taskIndexIn = std::make_shared< Sloth::Buffer<int>>("taskIndexIn", maxTasks, 0, false);;
            taskIndexOut = std::make_shared< Sloth::Buffer<int>>("taskIndexOut", maxTasks, 0, false);;

            taskInChunkId = std::make_shared< Sloth::Buffer<int>>("taskInChunkId", maxTasks, 0, false);
            taskOutChunkId = std::make_shared< Sloth::Buffer<int>>("taskOutChunkId", maxTasks, 0, false);
            taskParallelismIn = std::make_shared< Sloth::Buffer<int>>("taskParallelismIn", maxTasks, 0, false);
            taskParallelismOut = std::make_shared< Sloth::Buffer<int>>("taskParallelismOut", maxTasks, 0, false);


            chunkType = std::make_shared< Sloth::Buffer<char>>("chunkType", maxChunks, 0, false);            
            chunkDepth = std::make_shared< Sloth::Buffer<char>>("chunkDepth", maxChunks, 0, false);
            chunkCounter = std::make_shared< Sloth::Buffer<int>>("chunkCounter", maxChunks, 0, false);

            chunkOffset = std::make_shared< Sloth::Buffer<int>>("chunkOffset", maxChunks, 0, false);
            chunkLength = std::make_shared< Sloth::Buffer<int>>("chunkLength", maxChunks, 0, false);
            chunkProgress = std::make_shared< Sloth::Buffer<int>>("chunkProgress", maxChunks, 0, false);
            chunkRangeMin = std::make_shared< Sloth::Buffer<KeyType>>("chunkRangeMin", maxChunks, 0, false);
            chunkRangeMax = std::make_shared< Sloth::Buffer<KeyType>>("chunkRangeMax", maxChunks, 0, false);


		}

        void Build(std::vector<KeyType>& keys, std::vector<ValueType>& values)
        {
            const int inputSize = keys.size();
#ifdef SLOTH_DEBUG_ENABLED
      
            if (debugBufferSize == 0)
            {
                debugBufferSize = 1024 * 1024 * 100;
                debugBuffer = std::make_shared< Sloth::Buffer<int>>("debugBuffer", debugBufferSize, 0, false);
            }
            TreeInternalKernels::resetDebugBuffer << <debugBufferSize / TreeInternalKernels::taskThreads, TreeInternalKernels::taskThreads >> > (debugBuffer->Data());
            cudaDeviceSynchronize();
#else
            if (debugBufferSize == 0)
            {
                debugBuffer = std::make_shared< Sloth::Buffer<int>>("debugBuffer", 1, 0, false);
                debugBufferSize = 1;
            }
#endif
            size_t t;
            {
                Sloth::Bench bench(&t);
                if (lastInputSize < inputSize)
                {
                    keyIn = std::make_shared< Sloth::Buffer<KeyType>>("keyIn", inputSize, 0, false);
                    keyOut = std::make_shared< Sloth::Buffer<KeyType>>("keyOut", inputSize, 0, false);
                    valueIn = std::make_shared< Sloth::Buffer<ValueType>>("valueIn", inputSize, 0, false);
                    valueOut = std::make_shared< Sloth::Buffer<ValueType>>("valueOut", inputSize, 0, false);
                    lastInputSize = inputSize;
                }
                keyIn->CopyFrom(keys.data(), inputSize);
                valueIn->CopyFrom(values.data(), inputSize);
                // starting n tasks depending on chunk length
                int nTasks = 1 + (inputSize / ( 16*TreeInternalKernels::taskThreads));

                minMaxWorkCounter->Set(0, 0);
                // compute min-max range for first step
                inputMinMax->Set(0, TreeInternalKernels::iMax);
                inputMinMax->Set(1, TreeInternalKernels::iMin);
                TreeInternalKernels::minMaxReduction << <nTasks, TreeInternalKernels::taskThreads >> > (keyIn->Data(), minMaxWorkCounter->Data(), inputSize, inputMinMax->Data());
                cudaDeviceSynchronize();

                // first chunk = input array
                // counter is to coordinate target writing form multiple inputs (multiple threads)
                chunkCounter->Set(0, 0);
                chunkDepth->Set(0, 0);// root is depth 0
                chunkRangeMin->Set(0, inputMinMax->Get(0));
                chunkRangeMax->Set(0, inputMinMax->Get(1));
                taskParallelismIn->Set(0, nTasks);
                chunkLength->Set(0, inputSize);
                chunkOffset->Set(0, 0);// chunk starts at 0th element
                chunkProgress->Set(0, 0); // tasks coordinate with this atomically


                // number of tasks launched
                taskInCounter->Set(0, nTasks);
                std::vector<int> tasks(nTasks);
                for (int i = 0; i < nTasks; i++)
                {                    
                    tasks[i] = 0;
                    
                }
                taskInChunkId->CopyFrom(tasks.data(),nTasks);
                

                bool working = true;
                int maxDebugCtr = 30;
                while (working)
                {
                    if (maxDebugCtr-- == 0)
                        break;
                    taskOutCounter->Set(0, 0);


                    TreeInternalKernels::resetChunkLength << <nTasks, TreeInternalKernels::taskThreads >> > (
                        taskIndexIn->Data(),
                        taskParallelismIn->Data(),
                        taskInCounter->Data(), taskInChunkId->Data(), chunkDepth->Data(),
                        chunkLength->Data(), chunkProgress->Data(), chunkOffset->Data(),
                        chunkRangeMin->Data(), chunkRangeMax->Data(), chunkCounter->Data(), chunkType->Data(),
                        keyIn->Data(), valueIn->Data(), keyOut->Data(), valueOut->Data(),
                        taskOutCounter->Data(),
                        debugBuffer->Data()
                        );
                    TreeInternalKernels::computeChildChunkAllocationRequirements << <nTasks, TreeInternalKernels::taskThreads >> > (
                        taskIndexIn->Data(),
                        taskParallelismIn->Data(),
                        taskInCounter->Data(), taskInChunkId->Data(), chunkDepth->Data(),
                        chunkLength->Data(), chunkProgress->Data(),  chunkOffset->Data(),
                        chunkRangeMin->Data(), chunkRangeMax->Data(), chunkCounter->Data(), chunkType->Data(),
                        keyIn->Data(), valueIn->Data(), keyOut->Data(), valueOut->Data(),
                        taskOutCounter->Data(),
                        debugBuffer->Data()
                        );

                    TreeInternalKernels::computeChildChunkOffset << <nTasks, TreeInternalKernels::taskThreads >> > (
                        taskIndexIn->Data(),
                        taskParallelismIn->Data(),
                        taskInCounter->Data(), taskInChunkId->Data(), chunkDepth->Data(),
                        chunkLength->Data(), chunkProgress->Data(),  chunkOffset->Data(),
                        chunkRangeMin->Data(), chunkRangeMax->Data(), chunkCounter->Data(), chunkType->Data(),
                        keyIn->Data(), valueIn->Data(), keyOut->Data(), valueOut->Data(),
                        taskOutCounter->Data(),
                        debugBuffer->Data()
                        );

                    TreeInternalKernels::allocateChildChunkAndCopy << <nTasks, TreeInternalKernels::taskThreads >> > (
                        taskIndexIn->Data(),
                        taskParallelismIn->Data(),
                        taskInCounter->Data(), taskInChunkId->Data(), chunkDepth->Data(),
                        chunkLength->Data(), chunkProgress->Data(),  chunkOffset->Data(),
                        chunkRangeMin->Data(), chunkRangeMax->Data(), chunkCounter->Data(), chunkType->Data(),
                        keyIn->Data(), valueIn->Data(), keyOut->Data(), valueOut->Data(),
                        taskOutCounter->Data(),
                        debugBuffer->Data()
                        );


                    TreeInternalKernels::copyChunkBack << <nTasks, TreeInternalKernels::taskThreads >> > (
                        taskIndexIn->Data(),
                        taskParallelismIn->Data(),
                        taskInCounter->Data(), taskInChunkId->Data(), chunkDepth->Data(),
                        chunkLength->Data(), chunkProgress->Data(),  chunkOffset->Data(),
                        chunkRangeMin->Data(), chunkRangeMax->Data(), chunkCounter->Data(), chunkType->Data(),
                        keyIn->Data(), valueIn->Data(), keyOut->Data(), valueOut->Data(),
                        taskOutCounter->Data(),
                        debugBuffer->Data()
                        );


                    // recursion part                    
                    TreeInternalKernels::createTask << <nTasks, TreeInternalKernels::taskThreads >> > (
                        taskIndexIn->Data(),
                        taskParallelismIn->Data(),
                        taskInCounter->Data(), taskInChunkId->Data(), chunkDepth->Data(),
                        chunkLength->Data(), chunkProgress->Data(),  chunkOffset->Data(),
                        chunkRangeMin->Data(), chunkRangeMax->Data(), chunkCounter->Data(), chunkType->Data(),
                        keyIn->Data(), valueIn->Data(), keyOut->Data(), valueOut->Data(),
                        taskOutCounter->Data(),
                        taskOutChunkId->Data(),
                        taskParallelismOut->Data(),
                        taskIndexOut->Data(),
                        debugBuffer->Data()
                        );

                    gpuErrchk(cudaDeviceSynchronize());

                    const int numNewTasks = taskOutCounter->Get(0);
                    taskOutCounter->DeviceCopyTo(taskInCounter->Data(), 1);
                    taskOutChunkId->DeviceCopyTo(taskInChunkId->Data(), numNewTasks);
                    taskParallelismOut->DeviceCopyTo(taskParallelismIn->Data(), numNewTasks);
                    taskIndexOut->DeviceCopyTo(taskIndexIn->Data(), numNewTasks);
                    taskOutCounter->Set(0, 0);


#ifdef SLOTH_DEBUG_ENABLED
                    std::cout << "num tasks = " << numNewTasks << std::endl;
                    std::cout << "debug: " << debugBuffer->Get(0) << std::endl;
                    debugBuffer->Set(0,0);
#endif
                    nTasks = numNewTasks;
                    working = (numNewTasks > 0);
                }


#ifdef SLOTH_DEBUG_ENABLED
                std::cout << "input range min: " << inputMinMax->Get(0) << std::endl;
                std::cout << "input range max: " << inputMinMax->Get(1) << std::endl;
                std::cout << "debug: " << debugBuffer->Get(0) << std::endl;
                int sum = 0;
                for (int i = 0; i < TreeInternalKernels::numChildNodesPerParent; i++)
                {
                    sum += chunkCounter->Get(1 + i);
                    std::cout << " total processed: " << chunkCounter->Get(1 + i) << std::endl;
                }
                std::cout << "sum = " << sum << std::endl;
                std::cout << "-------------------------------" << std::endl;
#endif
            }
            std::cout << "gpu: " << t / 1000000000.0 << "s" << std::endl;

          

        }

		~Tree()
		{

		}
	

	};
}
