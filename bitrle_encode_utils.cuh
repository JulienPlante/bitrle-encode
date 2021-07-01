#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>


// Little header file library to reduce main file's clutter
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
cudaError_t check(cudaError_t result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error at " << file << ':' << line << " code=" << static_cast<unsigned int>(result) << '(' << cudaGetErrorName(result) << ") \"" << func << "\"\n" << std::endl;
		exit(result);
	}
	return result;
}

struct bitrle_elem // Struct taken from original implementation, representing each element of the RLE
{
	uint32_t position;
	uint32_t duration;
};

__device__ int n_pulses; // Just to simplify the example, this is the number of pulses encoded (originally stored in a class, I simplified here)

template <typename InputIteratorT>
__global__ void fill_kernel(InputIteratorT d_bitset, const int n_words) // Fill bitset with dummy data (only 1s for even words, 0s for odd words)
{
	using InputT = typename std::iterator_traits<InputIteratorT>::value_type;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_words; i += blockDim.x * gridDim.x)
		d_bitset[i] = (i % 2) ? InputT(-1) : InputT(0); // This can be changed to anything else
}


__global__ void check_results(bitrle_elem* elems) // print results from device for simplicity (no need for perfs here)
{
	printf("%d pulses encoded", n_pulses);
	for (int i = 0; i < min(n_pulses, 32); i++)
		printf("elems[%d]: position = %u, duration = %u\n", i, elems[i].position, elems[i].duration);
}


// Ready flag manipulation
__device__ __forceinline__ uint2 set_ready_flag(const uint2& v)
{
	return uint2{ v.x, v.y | 0x80000000 };
}

__device__ __forceinline__ bool get_ready_flag(const uint2& v)
{
	return bool(v.y & 0x80000000);
}

__device__ __forceinline__ uint2 reset_ready_flag(const uint2& v)
{
	return uint2{ v.x, v.y & 0x7fffffff };
}


// C++ operators for uint2 (used by CUB)
__device__ __forceinline__ uint2 operator+(const uint2& lhs, const uint2& rhs)
{
	return uint2{ lhs.x + rhs.x, lhs.y + rhs.y };
}

__device__ __forceinline__ uint2 operator+=(uint2& lhs, const uint2& rhs)
{
	return uint2{ lhs.x += rhs.x, lhs.y += rhs.y };
}



