#include"globals.cuh"
#include"helper.cuh"
#include<string>
namespace Sloth
{

	template<typename Type>
	struct Buffer
	{
	private:

		bool compressed;
		bool allocated;
		Type* data;
		int n;
		std::string name;
	public:
		Buffer()
		{

			compressed = false;
			data = nullptr;
			n = 0;
			name = "unnamed";
			allocated = false;
		}
		Buffer(std::string nameString, int numElements, CUdevice currentDevice, bool optInCompression)
		{

			//cudaSetDevice(currentDevice);
			name = nameString;
			data = nullptr;
			n = numElements;

			allocated = false;




			if (optInCompression)
			{

				if (CUDA_SUCCESS != SlothHelper::allocateCompressible((void**)&data, n * sizeof(Type), true))
				{
					compressed = false;
					std::cout << name << " buffer has CUDA ERROR: compressible memory failed. Trying normal allocation" << std::endl;
				}
				else
				{
					compressed = true;
					allocated = true;

				}
			}
			else
			{
				compressed = false;
			}



			if (!compressed)
			{

				auto errCu = cudaMalloc(&data, n * sizeof(Type));
				Sloth::gpuErrchk(errCu);
				allocated = true;
			}

		}

		void CopyTo(Type * ptr,const int numElements)
		{
			gpuErrchk(cudaMemcpy(ptr,data,numElements*sizeof(Type),cudaMemcpyDeviceToHost));
		}

		void CopyFrom(Type* ptr, const int numElements)
		{
			gpuErrchk(cudaMemcpy(data,ptr, numElements * sizeof(Type), cudaMemcpyHostToDevice));
		}

		void Set(const int id, const Type val) const
		{
			gpuErrchk(cudaMemcpy(data+id, &val, sizeof(Type), cudaMemcpyHostToDevice));
		}

		Type Get(const int id) const
		{
			Type result;
			gpuErrchk(cudaMemcpy(&result, data, sizeof(Type), cudaMemcpyDeviceToHost));
			return result;
		}

		Type* Data()
		{
			return data;
		}

		bool CompressionEnabled()
		{
			return compressed;
		}

		~Buffer()
		{

			if (allocated && data != nullptr && n > 0 && name != "unnamed")
			{

				if (compressed)
				{
					if (CUDA_SUCCESS != SlothHelper::freeCompressible((void*)data, n * sizeof(Type), true))
					{
						std::cout << name << " buffer has CUDA ERROR: compressible memory-free failed. Trying normal deallocation" << std::endl;
						Sloth::gpuErrchk(cudaFree(data));
					}
				}
				else
					Sloth::gpuErrchk(cudaFree(data));
			}
		}
	};
}