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



        // each task is computed by multiple blocks of threads
        static constexpr int nodeThreads = 128;
        static constexpr int nodeElements = 128;
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

        template<typename Type>
        __device__ Type loadSingle(Type * ptr,Type * ptrSm, const int id)
        {
            if (id == 0)
            {
                *ptrSm = *ptr;
            }
            __syncthreads();
            const Type result = *ptrSm;
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
            int * chunkLength, int * chunkProgress,
            KeyType * keyIn, ValueType * valueIn, KeyType * keyOut, KeyType * valueOut
        )
        {
            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int bs = blockDim.x;
            const int gs = gridDim.x;
            __shared__ int smLoadInt[1];
            const int chunkId = loadSingle(taskInChunkId, smLoadInt, tid);
            const int totalWorkSize = loadSingle(chunkLength + chunkId, smLoadInt, tid);

            if(tid==0)
                printf(" (chunk id = %i   work size = %i) ", chunkId,totalWorkSize);

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


        std::shared_ptr<Sloth::Buffer<int>> chunkLength;
        std::shared_ptr<Sloth::Buffer<int>> chunkProgress; // tasks atomically use this to get more work



        std::shared_ptr<Sloth::Buffer<KeyType>> keyIn;
        std::shared_ptr<Sloth::Buffer<ValueType>> valueIn;

        std::shared_ptr<Sloth::Buffer<KeyType>> keyOut;
        std::shared_ptr<Sloth::Buffer<ValueType>> valueOut;

        int maxTasks;
        int maxChunks;
        int lastInputSize;
        int allocSize;
	public:
		Tree()
		{
            maxTasks = 1024 * 1024;
            maxChunks = 1 + std::pow(TreeInternalKernels::numChildNodesPerParent, TreeInternalKernels::nodeMaxDepth);
            cudaSetDevice(0);

            lastInputSize = 0;
            taskInCounter = std::make_shared< Sloth::Buffer<int>>("taskInCounter", 1, 0, false);
            taskOutCounter = std::make_shared< Sloth::Buffer<int>>("taskOutCounter", 1, 0, false);
            taskInChunkId = std::make_shared< Sloth::Buffer<int>>("taskInChunkId", maxTasks, 0, false);
            taskOutChunkId = std::make_shared< Sloth::Buffer<int>>("taskOutChunkId", maxTasks, 0, false);

            chunkLength = std::make_shared< Sloth::Buffer<int>>("chunkLength", maxChunks, 0, false);
            chunkProgress = std::make_shared< Sloth::Buffer<int>>("chunkProgress", maxChunks, 0, false);
		}

        void Build(std::vector<KeyType>& keys, std::vector<ValueType>& values, bool debugEnabled = false)
        {
            const int inputSize = keys.size();

            size_t t;
            {
                Sloth::Bench bench(&t);
                if (lastInputSize < inputSize)
                {
                    keyIn = std::make_shared< Sloth::Buffer<int>>("keyIn", inputSize, 0, false);
                    keyOut = std::make_shared< Sloth::Buffer<int>>("keyOut", inputSize, 0, false);
                    valueIn = std::make_shared< Sloth::Buffer<int>>("valueIn", inputSize, 0, false);
                    valueOut = std::make_shared< Sloth::Buffer<int>>("valueOut", inputSize, 0, false);
                    lastInputSize = inputSize;
                }


                // starting n tasks depending on chunk length
                const int nTasks = 1 + (inputSize / ( 16*TreeInternalKernels::nodeThreads));
                taskInCounter->Set(0, nTasks);                
                chunkLength->Set(0, inputSize);
                chunkProgress->Set(0, 0);
                for (int i = 0; i < nTasks; i++)
                {                    
                    taskInChunkId->Set(i, 0);                    
                }
                
                TreeInternalKernels::createChunks<<<nTasks, TreeInternalKernels::nodeThreads>>>(
                    taskInCounter->Data(), taskInChunkId->Data(),
                    chunkLength->Data(), chunkProgress->Data(),
                    keyIn->Data(), valueIn->Data(), keyOut->Data(), valueOut->Data()
                );
                
            }
            std::cout << "gpu: " << t / 1000000000.0 << "s" << std::endl;

          

        }

		~Tree()
		{

		}
	

	};
}
