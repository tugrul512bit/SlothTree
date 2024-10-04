#include"buffer.cuh"
#include<limits>
#include<vector>
#include<iostream>
#include<memory>
#include<unordered_map>
namespace Sloth
{
    namespace TreeInternalKernels
    {

        static constexpr int iMin = std::numeric_limits<int>::min();
        static constexpr int iMax = std::numeric_limits<int>::max();



        // each node of tree is handled by 1 block of threads
        static constexpr int nodeThreads = 64;
        static constexpr int nodeElements = 64;
        static constexpr int nodeMaxDepth = 20;
        static constexpr int numChildNodesPerParent =2;
        static constexpr int numTaskParameters = 7 + 2 * numChildNodesPerParent; // offsets + atomic counters


        static constexpr int nodeTreeHeader = 2;
        static constexpr int nodeSize = 5;

        // node structure (array of structs for simplicity, no compressed structure) 
        /*
            tree[0]: allocator counter for nodes
            tree[1]: allocator counter for elements
            int-0: number of elements if its a leaf node
            int-1: min key
            int-2: max key
            int-3: element begin in element buffer
            int-4:
        */

        template<typename Type>
        struct Reducer
        {
            __device__ Type BlockSum(const int id, const Type val, Type* sm)
            {
                sm[id] = val;
                __syncthreads();
                // shared reduction
                for (unsigned int i = ((unsigned int)nodeThreads) >> 1; i > 16; i >>= 1)
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
                for (unsigned int i = ((unsigned int)nodeThreads) >> 1; i > 0; i >>= 1)
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
                for (unsigned int i = ((unsigned int)nodeThreads) >> 1; i > 0; i >>= 1)
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
                    smMask[nodeThreads + 1] = 0;
                }
                __syncthreads();
                int gatherDistance = 1;

                while (gatherDistance < nodeThreads)
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
            __device__ void BlockCompact2(Type val, Type2 val0, bool mask, Type* sm, Type2* sm0, Type* sm2, Type2* sm20,
                TypeMask* smMask, const int id, Type* out1, Type2* out2, char* smChunkTracking, char* smChunkTracking2, char chunkId, char* outChunk)
            {

                sm[id] = val;
                sm0[id] = val0;
                smMask[id + 1] = mask;
                smChunkTracking[id] = chunkId;
                if (id == 0)
                {
                    smMask[0] = 0;
                    smMask[nodeThreads + 1] = 0;
                }
                __syncthreads();
                int gatherDistance = 1;

                while (gatherDistance < nodeThreads)
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




        // counts keys that should fall inside each child node
        // if total is less than limit, it stays leaf, else, it will create child nodes
        template<typename KeyType, typename ValueType>
        __global__ void buildTreeCountKeys(
            int* __restrict__ tree, const KeyType* __restrict__ inputKey, const ValueType* __restrict__ inputValue, KeyType* __restrict__ treeKeys,
            ValueType* __restrict__ treeValues, const int allocSize, const int nodeMaxDepth, int* __restrict__ taskQueueUsed, int* __restrict__ taskQueueGenerated,

            /*
                taskComputeResource means how many blocks a task has for computing power.the deeper the node the lesser the power.root node has highest power.
                blocks:
                task1 task1 task1 task2 task2 task2 .... taskN taskN taskN
                root blocks: number of SM units
                second layer: min(1,num SM / number of child nodes)
                third layer: min(1,num SM / num children ^2)
                ...
            */
            const int taskComputeResource
        )
        {
            const int id = threadIdx.x;
            const int gid = blockIdx.x;
            const int bs = blockDim.x;
            const int gs = gridDim.x;
            const int numStrideThreads = taskComputeResource * bs;
            const int strideThreadId = (id + gid * bs) % numStrideThreads;
            const int taskBegin = (gid / taskComputeResource) * numTaskParameters + 1;




            // node offset, node depth, isinputinput,elementstartindex
            __shared__ int nodeOffsetSm;
            __shared__ int nodeDepthSm;
            __shared__ bool isInputInputSm;
            __shared__ int elementOffsetSm;
            __shared__ int minKeySm;
            __shared__ int maxKeySm;
            __shared__ int inputSizeSm;
            if (id == 0)
            {
                nodeOffsetSm = taskQueueUsed[taskBegin]; // this is first element of task
                nodeDepthSm = taskQueueUsed[taskBegin + 1];
                isInputInputSm = taskQueueUsed[taskBegin + 2];
                elementOffsetSm = taskQueueUsed[taskBegin + 3];
                // todo: separate array with KeyType type (int is not enough)
                minKeySm = taskQueueUsed[taskBegin + 4];
                maxKeySm = taskQueueUsed[taskBegin + 5];
                inputSizeSm = taskQueueUsed[taskBegin + 6];
            }
            __syncthreads();
            const int nodeOffset = nodeOffsetSm;
            const int nodeDepth = nodeDepthSm;
            const bool isInputInput = isInputInputSm;
            const int elementOffset = elementOffsetSm;
            const KeyType minKey = minKeySm;
            const KeyType maxKey = maxKeySm;
            const int inputSize = inputSizeSm;

            // if chunk size is zero or node depth is maximum, it won't require any child node
            const int chunkSize = ((float)maxKey - (float)minKey) / numChildNodesPerParent;
            const bool childNodePossible = (chunkSize > 0) && (nodeDepth < nodeMaxDepth);

            // current node's data is already computed from parent. so no more compute necessary
            if (!childNodePossible)
                return;

            // it can distribute data to child nodes now
            const int thisNodeIndex = (nodeOffset - nodeTreeHeader) / nodeSize;
            const int firstChildNodeIndex = (thisNodeIndex + 1) * numChildNodesPerParent - 1;
            int childNodeOffset[numChildNodesPerParent];
            int childNodeRangeStart[numChildNodesPerParent];
            int childNodeRangeStop[numChildNodesPerParent];
            for (int i = 0; i < numChildNodesPerParent; i++)
            {
                const int curIndex = firstChildNodeIndex + i;
                const int curOffset = curIndex * nodeSize + nodeTreeHeader;
                childNodeOffset[i] = curOffset;
            }
            int curRangeStart = minKey;
            for (int i = 0; i < numChildNodesPerParent; i++)
            {

                int curRangeStop = curRangeStart + ((float)maxKey - (float)minKey) / numChildNodesPerParent;
                if (i == numChildNodesPerParent - 1)
                    curRangeStop = maxKey;
                if (curRangeStart <= curRangeStop)
                {
                    childNodeRangeStart[i] = curRangeStart;
                    childNodeRangeStop[i] = curRangeStop;
                }

                curRangeStart = curRangeStop + 1;
            }

            __shared__ int smInt[nodeThreads + 2];

            Reducer<int> reducer;



            const int strideSteps = 1 + (inputSize - 1) / numStrideThreads;

            const KeyType* ptrKey = (isInputInput ? inputKey : treeKeys);
            const ValueType* ptrVal = (isInputInput ? inputValue : treeValues);

            // register reduction
            int newElementCount[numChildNodesPerParent];
            for (int i = 0; i < numChildNodesPerParent; i++)
                newElementCount[i] = 0;

            for (int i = 0; i < strideSteps; i++)
            {
                const int curIndex = i * numStrideThreads + strideThreadId;

                if (curIndex < inputSize)
                {
                    const KeyType key = ptrKey[curIndex + elementOffset];

                    for (int j = 0; j < numChildNodesPerParent; j++)
                        newElementCount[j] += (key >= childNodeRangeStart[j] && key <= childNodeRangeStop[j]);
                }
            }


            // shared memory reduction
            int allocationRequired[numChildNodesPerParent];
            for (int j = 0; j < numChildNodesPerParent; j++)
                allocationRequired[j] = reducer.BlockSum(id, newElementCount[j], smInt);

            // global memory reduction
            if (id == 0)
            {
                int total = 0;
                for (int i = 0; i < numChildNodesPerParent; i++)
                {
                    // update task with size of chunk calculated
                    atomicAdd(&taskQueueUsed[taskBegin + 7 + i], allocationRequired[i]);
                }

            }
        }

        template<typename KeyType, typename ValueType>
        __global__ void buildTreeAllocateSpace(
            int* __restrict__ tree, const KeyType* __restrict__ inputKey, const ValueType* __restrict__ inputValue, KeyType* __restrict__ treeKeys,
            ValueType* __restrict__ treeValues, const int allocSize, const int nodeMaxDepth, int* __restrict__ taskQueueUsed, int* __restrict__ taskQueueGenerated,
            const int taskComputeResource
        )
        {
            const int id = threadIdx.x;
            const int gid = blockIdx.x;
            const int bs = blockDim.x;
            const int gs = gridDim.x;
            const int numStrideThreads = taskComputeResource * bs;
            const int strideThreadId = (id + gid * bs) % numStrideThreads;
            const int taskBegin = (gid / taskComputeResource) * numTaskParameters + 1;

            // atomically allocate from buffer
            // replace allocation size info with allocation offset for nodes
            // so that next kernel can directly move data to there
            if (id == 0 && gid == 0)
            {
                for (int i = 0; i < numChildNodesPerParent; i++)
                {
                    taskQueueUsed[taskBegin + 7 + i] = atomicAdd(&tree[1], taskQueueUsed[taskBegin + 7 + i]);
                    taskQueueUsed[taskBegin + 7 + i + numChildNodesPerParent] = 0;
                }
            }
        }



        template<typename KeyType, typename ValueType>
        __global__ void buildTreeDistributeElements(
            int* __restrict__ tree, const KeyType* __restrict__ inputKey, const ValueType* __restrict__ inputValue, KeyType* __restrict__ treeKeys,
            ValueType* __restrict__ treeValues, const int allocSize, const int nodeMaxDepth, int* __restrict__ taskQueueUsed, int* __restrict__ taskQueueGenerated,
            const int taskComputeResource, const int pingPongState, const int arraySize
        )
        {

            const int id = threadIdx.x;
            const int gid = blockIdx.x;
            const int bs = blockDim.x;
            const int gs = gridDim.x;
            const int numStrideThreads = taskComputeResource * bs;
            const int taskBegin = (gid / taskComputeResource) * numTaskParameters + 1;
            const int strideThreadId = (id + gid * bs) % numStrideThreads;

            __shared__ int nodeElementOffsetsSm[numChildNodesPerParent];



            // atomically allocate from buffer
            // replace allocation size info with allocation offset for nodes
            // so that next kernel can directly move data to there
            if (id == 0)
            {
                for (int i = 0; i < numChildNodesPerParent; i++)
                    nodeElementOffsetsSm[i] = taskQueueUsed[taskBegin + 7 + i];
            }

            __syncthreads();
            int nodeElementOffsets[numChildNodesPerParent];
            for (int i = 0; i < numChildNodesPerParent; i++)
                nodeElementOffsets[i] = nodeElementOffsetsSm[i];




            // node offset, node depth, isinputinput,elementstartindex
            __shared__ int nodeOffsetSm;
            __shared__ int nodeDepthSm;
            __shared__ bool isInputInputSm;
            __shared__ int elementOffsetSm;
            __shared__ int minKeySm;
            __shared__ int maxKeySm;
            __shared__ int inputSizeSm;
            if (id == 0)
            {
                nodeOffsetSm = taskQueueUsed[taskBegin]; // this is first element of task
                nodeDepthSm = taskQueueUsed[taskBegin + 1];
                isInputInputSm = taskQueueUsed[taskBegin + 2];
                elementOffsetSm = taskQueueUsed[taskBegin + 3];
                // todo: separate array with KeyType type (int is not enough)
                minKeySm = taskQueueUsed[taskBegin + 4];
                maxKeySm = taskQueueUsed[taskBegin + 5];
                inputSizeSm = taskQueueUsed[taskBegin + 6];
            }
            __syncthreads();
            const int nodeOffset = nodeOffsetSm;
            const int nodeDepth = nodeDepthSm;
            const bool isInputInput = isInputInputSm;
            const int elementOffset = elementOffsetSm;
            const KeyType minKey = minKeySm;
            const KeyType maxKey = maxKeySm;
            const int inputSize = inputSizeSm;

            // if chunk size is zero or node depth is maximum, it won't require any child node
            const int chunkSize = ((float)maxKey - (float)minKey) / numChildNodesPerParent;
            const bool childNodePossible = (chunkSize > 0) && (nodeDepth < nodeMaxDepth);

            // current node's data is already computed from parent. so no more compute necessary
            if (!childNodePossible)
            {

                return;
            }
            // it can distribute data to child nodes now
            const int thisNodeIndex = (nodeOffset - nodeTreeHeader) / nodeSize;
            const int firstChildNodeIndex = (thisNodeIndex + 1) * numChildNodesPerParent - 1;
            int childNodeOffset[numChildNodesPerParent];
            int childNodeRangeStart[numChildNodesPerParent];
            int childNodeRangeStop[numChildNodesPerParent];
            for (int i = 0; i < numChildNodesPerParent; i++)
            {
                const int curIndex = firstChildNodeIndex + i;
                const int curOffset = curIndex * nodeSize + nodeTreeHeader;
                childNodeOffset[i] = curOffset;
            }
            int curRangeStart = minKey;
            for (int i = 0; i < numChildNodesPerParent; i++)
            {

                int curRangeStop = curRangeStart + chunkSize;
                if (i == numChildNodesPerParent - 1)
                    curRangeStop = maxKey;
                if (curRangeStart <= curRangeStop)
                {
                    childNodeRangeStart[i] = curRangeStart;
                    childNodeRangeStop[i] = curRangeStop;
                }

                curRangeStart = curRangeStop + 1;
            }


            __shared__ KeyType smKey[nodeThreads + 2];
            __shared__ ValueType smValue[nodeThreads + 2];
            __shared__ KeyType smKey2[nodeThreads + 2];
            __shared__ ValueType smValue2[nodeThreads + 2];
            __shared__ char smChunkTracking[nodeThreads + 2];
            __shared__ char smChunkTracking2[nodeThreads + 2];
            __shared__ int smInt[nodeThreads + 2];

            Reducer<int> reducer;
            StreamCompacter<KeyType, int> keyCompacter;

            const int strideSteps = 1 + (inputSize - 1) / numStrideThreads;
            // reading pingponog order is opposite of writing order (reads one half writes on other half)
            const KeyType* ptrKey = (isInputInput ? inputKey : (pingPongState == 1 ? treeKeys : (treeKeys + arraySize)));
            const ValueType* ptrVal = (isInputInput ? inputValue : (pingPongState == 1 ? treeValues : (treeValues + arraySize)));

            // mask for compaction
            int newElement[numChildNodesPerParent];

            __shared__ int offsetSm;
            __shared__ int sharedAtomicReductionForOffsets[numChildNodesPerParent]; // <-- this will be added atomically on global
            __shared__ int sharedCompactionKeys[nodeThreads]; // <--- this will get coalesced node data rather than interleaved
            __shared__ int sharedCompactionValues[nodeThreads];
            __shared__ int sharedCompactionSegmentIndex[numChildNodesPerParent]; // <--- prefix summed offsets within sharedCompaction
            for (int i = 0; i < numChildNodesPerParent; i++)
                sharedAtomicReductionForOffsets[i] = 0;
            __syncthreads();
            for (int i = 0; i < strideSteps; i++)
            {
                const int curIndex = i * numStrideThreads + strideThreadId;
                KeyType key;
                ValueType val;
                int newElement = 0;// = 0;// [numChildNodesPerParent] ;
                char chunkIndex = -1;
                if (curIndex < inputSize)
                {
                    key = ptrKey[curIndex + elementOffset];
                    val = ptrVal[curIndex + elementOffset];
                    newElement = (key >= minKey && key <= maxKey);
                    for (int j = 0; j < numChildNodesPerParent; j++)
                    {
                        if (key >= childNodeRangeStart[j] && key <= childNodeRangeStop[j])
                        {
                            chunkIndex = j;
                        }
                    }
                }

                const int numCopy = reducer.BlockSum(id, newElement, smInt);

                KeyType compactedKey;
                ValueType compactedValue;
                char compactedChunkId = -1;
                // stream compaction that keeps keys with their own values and chunk indices
                keyCompacter.BlockCompact2(key, val, newElement, smKey, smValue, smKey2, smValue2, smInt, id, &compactedKey, &compactedValue,
                    smChunkTracking, smChunkTracking2, chunkIndex, &compactedChunkId);



                if (id < numCopy)
                {

                    const int ofs = atomicAdd(&taskQueueUsed[taskBegin + 7 + numChildNodesPerParent + compactedChunkId], 1);
                    // writes to left
                    if (pingPongState == 0)
                    {
                        treeKeys[nodeElementOffsets[compactedChunkId] + ofs] = compactedKey;
                        treeValues[nodeElementOffsets[compactedChunkId] + ofs] = compactedValue;
                    }
                    else // write to right
                    {
                        treeKeys[arraySize + nodeElementOffsets[compactedChunkId] + ofs] = compactedKey;
                        treeValues[arraySize + nodeElementOffsets[compactedChunkId] + ofs] = compactedValue;
                    }
                }


            }
        }


        template<typename KeyType, typename ValueType>
        __global__ void buildTreeCreateTasks(
            int* __restrict__ tree, const KeyType* __restrict__ inputKey, const ValueType* __restrict__ inputValue, KeyType* __restrict__ treeKeys,
            ValueType* __restrict__ treeValues, const int allocSize, const int nodeMaxDepth, int* __restrict__ taskQueueUsed, int* __restrict__ taskQueueGenerated,
            const int taskComputeResource, const int pingPongState, const int arraySize
        )
        {

            const int id = threadIdx.x;
            const int gid = blockIdx.x;
            const int bs = blockDim.x;
            const int gs = gridDim.x;
            const int numStrideThreads = taskComputeResource * bs;
            const int taskBegin = (gid / taskComputeResource) * numTaskParameters + 1;
            const int strideThreadId = (id + gid * bs) % numStrideThreads;

            __shared__ int nodeElementOffsetsSm[numChildNodesPerParent];



            // atomically allocate from buffer
            // replace allocation size info with allocation offset for nodes
            // so that next kernel can directly move data to there
            if (id == 0)
            {
                for (int i = 0; i < numChildNodesPerParent; i++)
                    nodeElementOffsetsSm[i] = taskQueueUsed[taskBegin + 7 + i];
            }

            __syncthreads();
            int nodeElementOffsets[numChildNodesPerParent];
            for (int i = 0; i < numChildNodesPerParent; i++)
                nodeElementOffsets[i] = nodeElementOffsetsSm[i];


            // node offset, node depth, isinputinput,elementstartindex
            __shared__ int nodeOffsetSm;
            __shared__ int nodeDepthSm;
            __shared__ bool isInputInputSm;
            __shared__ int elementOffsetSm;
            __shared__ int minKeySm;
            __shared__ int maxKeySm;
            __shared__ int inputSizeSm;
            if (id == 0)
            {
                nodeOffsetSm = taskQueueUsed[taskBegin]; // this is first element of task
                nodeDepthSm = taskQueueUsed[taskBegin + 1];
                isInputInputSm = taskQueueUsed[taskBegin + 2];
                elementOffsetSm = taskQueueUsed[taskBegin + 3];
                // todo: separate array with KeyType type (int is not enough)
                minKeySm = taskQueueUsed[taskBegin + 4];
                maxKeySm = taskQueueUsed[taskBegin + 5];
                inputSizeSm = taskQueueUsed[taskBegin + 6];
            }
            __syncthreads();
            const int nodeOffset = nodeOffsetSm;
            const int nodeDepth = nodeDepthSm;
            const bool isInputInput = isInputInputSm;
            const int elementOffset = elementOffsetSm;
            const KeyType minKey = minKeySm;
            const KeyType maxKey = maxKeySm;
            const int inputSize = inputSizeSm;

            // if chunk size is zero or node depth is maximum, it won't require any child node
            const int chunkSize = ((float)maxKey - (float)minKey) / numChildNodesPerParent;
            const bool childNodePossible = (chunkSize > 0) && (nodeDepth < nodeMaxDepth);

            // current node's data is already computed from parent. so no more compute necessary
            if (!childNodePossible)
            {

                return;
            }
            // it can distribute data to child nodes now
            const int thisNodeIndex = (nodeOffset - nodeTreeHeader) / nodeSize;
            const int firstChildNodeIndex = (thisNodeIndex + 1) * numChildNodesPerParent - 1;
            int childNodeOffset[numChildNodesPerParent];
            int childNodeRangeStart[numChildNodesPerParent];
            int childNodeRangeStop[numChildNodesPerParent];
            for (int i = 0; i < numChildNodesPerParent; i++)
            {
                const int curIndex = firstChildNodeIndex + i;
                const int curOffset = curIndex * nodeSize + nodeTreeHeader;
                childNodeOffset[i] = curOffset;
            }
            int curRangeStart = minKey;
            for (int i = 0; i < numChildNodesPerParent; i++)
            {

                int curRangeStop = curRangeStart + chunkSize;
                if (i == numChildNodesPerParent - 1)
                    curRangeStop = maxKey;
                if (curRangeStart <= curRangeStop)
                {
                    childNodeRangeStart[i] = curRangeStart;
                    childNodeRangeStop[i] = curRangeStop;
                }

                curRangeStart = curRangeStop + 1;
            }


            if (id == 0)
            {
                for (int i = 0; i < numChildNodesPerParent; i++)
                {
                    int curRangeStart = childNodeRangeStart[i];
                    int curRangeStop = childNodeRangeStop[i];

                    if (taskQueueUsed[taskBegin + 7 + i + numChildNodesPerParent] > nodeElements)
                    {

                        tree[childNodeOffset[i]] = 0;
                        tree[childNodeOffset[i] + 1] = curRangeStart;
                        tree[childNodeOffset[i] + 2] = curRangeStop;
                        tree[childNodeOffset[i] + 3] = nodeElementOffsets[i]; // todo: debug this
                        tree[childNodeOffset[i] + 4] = 0; // reserved



                        const int allocatedTaskIndex = atomicAdd(&taskQueueGenerated[0], 1) * numTaskParameters + 1;


                        /*

                                    taskQueueHost[1] = nodeTreeHeader; // offset
                                    taskQueueHost[2] = 0; // node depth
                                    taskQueueHost[3] = 1; // its input is input array, not another node
                                    taskQueueHost[4] = 0; // element starting index in tree's element buffer
                                    taskQueueHost[5] = minMaxHost[0]; // range start
                                    taskQueueHost[6] = minMaxHost[1]; // range stop
                                    taskQueueHost[7] = inputSize; // number of elements to scan
                        */

                        taskQueueGenerated[allocatedTaskIndex] = childNodeOffset[i]; // offset of the task itself
                        taskQueueGenerated[allocatedTaskIndex + 1] = nodeDepth + 1; // child node depth
                        taskQueueGenerated[allocatedTaskIndex + 2] = 0; // child node ==> 0 means it uses tree's buffer instead of input buffer
                        taskQueueGenerated[allocatedTaskIndex + 3] = taskQueueUsed[taskBegin + 7 + i]; // element starting index in tree's element buffer
                        taskQueueGenerated[allocatedTaskIndex + 4] = curRangeStart; // key range start
                        taskQueueGenerated[allocatedTaskIndex + 5] = curRangeStop; // key range stop
                        taskQueueGenerated[allocatedTaskIndex + 6] = taskQueueUsed[taskBegin + 7 + i + numChildNodesPerParent]; // num elements to scan
                    }
                }
            }
        }



        __global__ void resetWork(int* work)
        {
            work[0] = 0;
        }
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
            __shared__ KeyType sm[nodeThreads];
            Reducer<KeyType> reducer;
            const KeyType minData = reducer.BlockMin(tid, keyMin, sm);
            const KeyType maxData = reducer.BlockMax(tid, keyMax, sm);
            if (tid == 0)
            {
                atomicMin(&minMaxData[0], minData);
                atomicMax(&minMaxData[1], maxData);
            }
        }

        __global__ void applyGeneratedTaskQueueAndReset(int* taskQueueUsed, int* taskQueueGenerated)
        {
            const int id = threadIdx.x;
            __shared__ int numSm;
            if (id == 0)
                numSm = taskQueueGenerated[0];
            __syncthreads();
            const int n = numSm * numTaskParameters + 1;
            const int numStep = 1 + n / 1024;

            for (int i = 0; i < numStep; i++)
            {
                const int curId = i * 1024 + id;
                if (curId < n)
                {
                    taskQueueUsed[curId] = taskQueueGenerated[curId];
                    taskQueueGenerated[curId] = 0;
                }
            }


        }



        __global__ void resetAllocation(int* tree)
        {
            tree[1] = 0;
        }

        __global__ void debug1(int* taskQueueGenerated)
        {
            printf(" generated task = %i \n", taskQueueGenerated[0]);
        }
    }

    template<typename KeyType, typename ValueType>
	struct Tree
	{
	private:


        std::shared_ptr<Sloth::Buffer<KeyType>> treeKey;
        std::shared_ptr<Sloth::Buffer<ValueType>> treeValue;
        std::shared_ptr<Sloth::Buffer<int>> work;


        std::shared_ptr<Sloth::Buffer<KeyType>>  minMaxData;
        std::shared_ptr<Sloth::Buffer<int>> tree;

   
        std::shared_ptr < Sloth::Buffer<KeyType>> inputKeyData;
        std::shared_ptr < Sloth::Buffer<ValueType>> inputValueData;



        std::shared_ptr<Sloth::Buffer<int>> taskQueueUsed;
        std::shared_ptr<Sloth::Buffer<int>> taskQueueGenerated;
        

        int lastInputSize;
        int allocSize;
	public:
		Tree()
		{
            allocSize = 1024 * 1024 * 4;
            cudaSetDevice(0);
            // assumption
    
            lastInputSize = 0;

            work = std::make_shared< Sloth::Buffer<int>>("work", 1, 0, false);
            minMaxData = std::make_shared< Sloth::Buffer<KeyType>>("minMaxData", 2, 0, false);
            tree = std::make_shared< Sloth::Buffer<int>>("tree", allocSize, 0, false);

            taskQueueUsed = std::make_shared< Sloth::Buffer<int>>("taskQueueUsed", 1+allocSize* TreeInternalKernels::numTaskParameters, 0, false);
            taskQueueGenerated = std::make_shared< Sloth::Buffer<int>>("taskQueueGenerated", 1 + allocSize * TreeInternalKernels::numTaskParameters, 0, false);
		}

        void Build(std::vector<KeyType>& keys, std::vector<ValueType>& values, bool debugEnabled = false)
        {
            const int inputSize = keys.size();
            if (inputSize != values.size())
            {
                std::cout << "ERROR: keys size and values size are not equal." << std::endl;
                return;
            }
            if (lastInputSize < inputSize)
            {
                // ping-pong buffers during building
                treeKey = std::make_shared<Sloth::Buffer<KeyType>>("treeKey", 2 * inputSize, 0, false);
                treeValue = std::make_shared<Sloth::Buffer<ValueType>>("treeValue", 2 * inputSize, 0, false);
                inputKeyData = std::make_shared<Sloth::Buffer<KeyType>>("inputKeyData", inputSize, 0, false);
                inputValueData = std::make_shared<Sloth::Buffer<ValueType>>("inputValueData", inputSize, 0, false);
                lastInputSize = inputSize;
            }
            std::vector<int> taskQueueHost(TreeInternalKernels::numTaskParameters + 1);


            KeyType minMaxHost[2];
            int treeHost[5] = { 0,0,0,0,0 };

           

            int pingPongState = 0;
            size_t t;
            {
                Sloth::Bench bench(&t);



                cudaMemcpy(inputKeyData->Data(), keys.data(), inputSize * sizeof(KeyType), cudaMemcpyHostToDevice);
                cudaMemcpy(inputValueData->Data(), values.data(), inputSize * sizeof(ValueType), cudaMemcpyHostToDevice);
                TreeInternalKernels::resetWork << <1, 1 >> > (work->Data());
                TreeInternalKernels::minMaxReduction << <1 + (inputSize / TreeInternalKernels::nodeThreads) / 4, TreeInternalKernels::nodeThreads >> > (inputKeyData->Data(), work->Data(), inputSize, minMaxData->Data());
                cudaDeviceSynchronize();
                cudaMemcpy(minMaxHost, minMaxData->Data(), 2 * sizeof(KeyType), cudaMemcpyDeviceToHost);

                for (int j = 0; j < 3; j++)
                {
                    treeHost[j] = 0;
                }
                // range: [10,20]
                treeHost[3] = minMaxHost[0];
                treeHost[4] = minMaxHost[1];

                // node offset, node depth, isinputinput,elementstartindex
                taskQueueHost[0] = 1; // number of tasks given
                // task 1
                taskQueueHost[1] = TreeInternalKernels::nodeTreeHeader; // offset
                taskQueueHost[2] = 0; // node depth
                taskQueueHost[3] = 1; // its input is input array, not another node
                taskQueueHost[4] = 0; // element starting index in tree's element buffer
                taskQueueHost[5] = minMaxHost[0]; // range start
                taskQueueHost[6] = minMaxHost[1]; // range stop
                taskQueueHost[7] = inputSize; // number of elements to scan
                for (int k = 0; k < TreeInternalKernels::numChildNodesPerParent * 2; k++)
                    taskQueueHost[8 + k] = 0; // reset counts of (pseudo)allocation sizes per child nodes and their offsets
                std::cout << "min:" << minMaxHost[0] << " max:" << minMaxHost[1] << std::endl;

                cudaMemcpy(taskQueueGenerated->Data(), taskQueueHost.data(), (1 + TreeInternalKernels::numTaskParameters) * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(tree->Data(), treeHost, 5 * sizeof(int), cudaMemcpyHostToDevice);
                int numTasks = 1;


                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, 0);
                int taskComputeResource = 1;


                // 1 means copying from left to right
                // 0 means copying from right to left (also initial run uses input arrays and writes to left)
                pingPongState = 0;
                int maxDebug = 100;
                while (numTasks > 0)
                {
                    if (maxDebug-- < 0)
                        break;
                    taskComputeResource = prop.multiProcessorCount / numTasks;
                    if (taskComputeResource < 1)
                        taskComputeResource = 1;

                    TreeInternalKernels::applyGeneratedTaskQueueAndReset << <1, 1024 >> > (taskQueueUsed->Data(), taskQueueGenerated->Data());
                    TreeInternalKernels::resetAllocation << <1, 1 >> > (tree->Data());


                    // counts number of elements per child node
                    TreeInternalKernels::buildTreeCountKeys << <taskComputeResource * numTasks, TreeInternalKernels::nodeThreads >> > (tree->Data(), inputKeyData->Data(), inputValueData->Data(),
                        treeKey->Data(), treeValue->Data(), allocSize, TreeInternalKernels::nodeMaxDepth, taskQueueUsed->Data(), taskQueueGenerated->Data(),
                        taskComputeResource);

                    // allocates space from atomic counter so that next kernel knows where to copy data
                    TreeInternalKernels::buildTreeAllocateSpace << <numTasks, 1 >> > (tree->Data(), inputKeyData->Data(), inputValueData->Data(),
                        treeKey->Data(), treeValue->Data(), allocSize, TreeInternalKernels::nodeMaxDepth, taskQueueUsed->Data(), taskQueueGenerated->Data(),
                        1);

                    TreeInternalKernels::buildTreeDistributeElements << <taskComputeResource * 8 * numTasks, TreeInternalKernels::nodeThreads >> > (tree->Data(), inputKeyData->Data(), inputValueData->Data(),
                        treeKey->Data(), treeValue->Data(), allocSize, TreeInternalKernels::nodeMaxDepth, taskQueueUsed->Data(), taskQueueGenerated->Data(),
                        taskComputeResource * 8, pingPongState, inputSize);

                    TreeInternalKernels::buildTreeCreateTasks << <1 * numTasks, TreeInternalKernels::nodeThreads >> > (tree->Data(), inputKeyData->Data(), inputValueData->Data(),
                        treeKey->Data(), treeValue->Data(), allocSize, TreeInternalKernels::nodeMaxDepth, taskQueueUsed->Data(), taskQueueGenerated->Data(),
                        1, pingPongState, inputSize);


                    //debug1 << <1, 1 >> > (taskQueueGenerated.Data());
                    cudaDeviceSynchronize();
                    pingPongState = 1 - pingPongState;

                    cudaMemcpy(taskQueueHost.data(), taskQueueGenerated->Data(), (1 + TreeInternalKernels::numTaskParameters) * sizeof(int), cudaMemcpyDeviceToHost);
                    numTasks = taskQueueHost[0];
                    if (debugEnabled)
                    {
                        std::cout << "number of tasks = " << numTasks<< "    compute resource per task = "<< taskComputeResource << std::endl;
                        
                    }
                }
                cudaDeviceSynchronize();


            }

            std::cout << "gpu: " << t / 1000000000.0 << "s" << std::endl;

            if (debugEnabled)
            {
                cudaMemcpy(treeHost, tree->Data(), 2 * sizeof(int), cudaMemcpyDeviceToHost);
                std::cout << "allocated buffer  = " << treeHost[1] << std::endl;
                int sum = 0;
                for (int k = 0; k < TreeInternalKernels::numChildNodesPerParent; k++)
                    sum += taskQueueHost[8 + TreeInternalKernels::numChildNodesPerParent + k];
                std::cout << "n = " << sum << std::endl;
                std::vector<KeyType> keyTmp(inputSize);
                if (pingPongState == 0)
                    cudaMemcpy(keyTmp.data(), treeKey->Data(), inputSize * sizeof(KeyType), cudaMemcpyDeviceToHost);
                else
                    cudaMemcpy(keyTmp.data(), treeKey->Data() + inputSize, inputSize * sizeof(KeyType), cudaMemcpyDeviceToHost);

                std::unordered_map<int, int> test, test2;
                for (int i = 0; i < inputSize; i++)
                    test[keys[i]]++;
                for (int i = 0; i < inputSize; i++)
                    test2[keyTmp[i]]++;


                int errCtr = 0;
                for (int i = 0; i < inputSize; i++)
                {
                    if (test[keys[i]] != test2[keys[i]])
                    {
                        errCtr++;
                    }
                }
                if (errCtr > 0)
                {
                    std::cout << test.size() << "==" << test2.size() << std::endl;
                    std::cout << test[400000] << "==" << test2[400000] << std::endl;
                    std::cout << test[40000] << "==" << test2[40000] << std::endl;
                    std::cout << test[4000] << "==" << test2[4000] << std::endl;
                    std::cout << test[400] << "==" << test2[400] << std::endl;
                    std::cout << test[40] << "==" << test2[40] << std::endl;
                    std::cout << "errctr = " << errCtr << std::endl;
                    exit(0);
                }
            }

            {
                Sloth::Bench bench(&t);
                std::unordered_map<KeyType, ValueType> map;
                for (int j = 0; j < inputSize; j++)
                {
                    map[keys[j]] = j;
                }
            }
            std::cout << "cpu: " << t / 1000000000.0 << "s" << std::endl;
        }

		~Tree()
		{

		}
	

	};
}
