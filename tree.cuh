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



        // each task is 1 block of threads
        // each chunk can have multiple tasks
        static constexpr int taskThreads = 1024;
        static constexpr int nodeElements = 1024;
        static constexpr int nodeMaxDepth = 10;
        static constexpr int numChildNodesPerParent =2;
        
       

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

        /* separate array into numChildNodesPerParent chunks (maximum 64 chunks)
         each chunk has a different range of values:
         chunk 1      chunk 2         chunk 3
         a      b     b+1   c         c+1   d
        */ 
        template<typename KeyType, typename ValueType>
        __global__ void createChunks(
            int * taskInCounter, int * taskInChunkId,
            int * chunkLength, int * chunkProgress, int * chunkNumTasks,
            KeyType * chunkRangeMin, KeyType * chunkRangeMax,
            KeyType * keyIn, ValueType * valueIn, KeyType * keyOut, KeyType * valueOut,
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
            const int chunkId = loadSingle(taskInChunkId, smLoadInt, tid);
            const int totalWorkSize = loadSingle(chunkLength + chunkId, smLoadInt, tid);
            const int chunkTasks = loadSingle(chunkNumTasks + chunkId, smLoadInt, tid);
            const KeyType chunkMin = loadSingle(chunkRangeMin + chunkId, smLoadInt, tid);
            const KeyType chunkMax = loadSingle(chunkRangeMin + chunkId, smLoadInt, tid);

            // if parent range is [0,640] and if there are 64 children per node, then each child will have 10 sized range, [0,9],[10,19],...
            const int chunkChildRange = ((double)chunkMax - (double)chunkMin) / numChildNodesPerParent;

            // chunk is processed in multiple leaps of this stride 
            const int chunkStride = taskThreads * chunkTasks;
            const int strideThreadId = tid + (bid % chunkTasks)*bs; // this allows variable amount of tasks per chunk within same kernel
            const int numStrides = 1 + (totalWorkSize - 1) / chunkStride;

            // each child-node has its own key range to compare so every deeper node will have a closer number to target when searching a number
            // inclusive values
            KeyType childNodeRangeMin[numChildNodesPerParent];
            KeyType childNodeRangeMax[numChildNodesPerParent];
            // preparing min-max range per child
            KeyType curBegin = chunkMin;

            for (int i = 0; i < numChildNodesPerParent; i++)
            {
                int curEnd = chunkMin + chunkChildRange;
                if (i == numChildNodesPerParent - 1)
                    curEnd = chunkMax;
                childNodeRangeMin[i] = curBegin;
                childNodeRangeMax[i] = curEnd;
                curBegin = curEnd + 1;
            }

            for (int i = 0; i < numStrides; i++)
            {
                const int currentId = strideThreadId + i * chunkStride;
                if (currentId < totalWorkSize)
                {
                   
                    atomicAdd(&debugBuffer[0], 1);
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

        std::shared_ptr<Sloth::Buffer<int>> taskInCounter;
        std::shared_ptr<Sloth::Buffer<int>> taskInChunkId;

        std::shared_ptr<Sloth::Buffer<int>> taskOutCounter;
        std::shared_ptr<Sloth::Buffer<int>> taskOutChunkId;

        std::shared_ptr<Sloth::Buffer<int>> chunkCounter;
        std::shared_ptr<Sloth::Buffer<int>> chunkNumTasks;
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

            inputMinMax = std::make_shared< Sloth::Buffer<int>>("inputMinMax", 1, 0, false);
            minMaxWorkCounter = std::make_shared< Sloth::Buffer<int>>("minMaxCounter", 1, 0, false);

            taskInCounter = std::make_shared< Sloth::Buffer<int>>("taskInCounter", 1, 0, false);
            taskOutCounter = std::make_shared< Sloth::Buffer<int>>("taskOutCounter", 1, 0, false);
            taskInChunkId = std::make_shared< Sloth::Buffer<int>>("taskInChunkId", maxTasks, 0, false);
            taskOutChunkId = std::make_shared< Sloth::Buffer<int>>("taskOutChunkId", maxTasks, 0, false);

            chunkCounter = std::make_shared< Sloth::Buffer<int>>("chunkCounter", 1, 0, false);
            chunkNumTasks = std::make_shared< Sloth::Buffer<int>>("chunkNumTasks", maxChunks, 0, false);
            chunkLength = std::make_shared< Sloth::Buffer<int>>("chunkLength", maxChunks, 0, false);
            chunkProgress = std::make_shared< Sloth::Buffer<int>>("chunkProgress", maxChunks, 0, false);
            chunkRangeMin = std::make_shared< Sloth::Buffer<KeyType>>("chunkRangeMin", maxChunks, 0, false);
            chunkRangeMax = std::make_shared< Sloth::Buffer<KeyType>>("chunkRangeMax", maxChunks, 0, false);


		}

        void Build(std::vector<KeyType>& keys, std::vector<ValueType>& values, bool debugEnabled = false)
        {
            const int inputSize = keys.size();
            if (debugEnabled)
            {
                if (debugBufferSize == 0)
                {
                    debugBufferSize = 1024 * 1024 * 100;
                    debugBuffer = std::make_shared< Sloth::Buffer<int>>("debugBuffer", debugBufferSize, 0, false);
                }
                TreeInternalKernels::resetDebugBuffer << <debugBufferSize / 1024, 1024 >> > (debugBuffer->Data());
            }
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

                // starting n tasks depending on chunk length
                const int nTasks = 1 + (inputSize / ( 16*TreeInternalKernels::taskThreads));
             
                // compute min-max range for first step
                TreeInternalKernels::minMaxReduction << <nTasks, 1024 >> > (keyIn->Data(), minMaxWorkCounter->Data(), inputSize, inputMinMax->Data());


                // first chunk = input array
                // number of chunks = 1 as input
                chunkCounter->Set(0, 1);

                chunkNumTasks->Set(0, nTasks);
                chunkLength->Set(0, inputSize);
                chunkProgress->Set(0, 0);


                // number of tasks launched
                taskInCounter->Set(0, nTasks);
                std::vector<int> tasks(nTasks);
                for (int i = 0; i < nTasks; i++)
                {                    
                    tasks[i] = 0;
                    
                }
                taskInChunkId->CopyFrom(tasks.data(),nTasks);
                
                taskOutCounter->Set(0, 0);
                TreeInternalKernels::createChunks<<<nTasks, TreeInternalKernels::taskThreads>>>(
                    taskInCounter->Data(), taskInChunkId->Data(),
                    chunkLength->Data(), chunkProgress->Data(),chunkNumTasks->Data(),
                    chunkRangeMin->Data(), chunkRangeMax->Data(),
                    keyIn->Data(), valueIn->Data(), keyOut->Data(), valueOut->Data(),
                    taskOutCounter->Data(),
                    debugBuffer->Data()
                );
                gpuErrchk(cudaDeviceSynchronize());


                if (debugEnabled)
                {


                    std::cout<<" total elements parsed: " << debugBuffer->Get(0) << std::endl;
                    for(int i=0;i<TreeInternalKernels::numChildNodesPerParent;i++)
                        std::cout << " child node " << i << ": "<<debugBuffer->Get(1+i) << std::endl;;
                }
            }
            std::cout << "gpu: " << t / 1000000000.0 << "s" << std::endl;

          

        }

		~Tree()
		{

		}
	

	};
}
