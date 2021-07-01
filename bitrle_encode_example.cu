#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>


#include <iostream>
#include "bitrle_encode_utils.cuh"



template <typename T>
__device__ __forceinline__ T funnelshift_left(const T& lo, const T& hi) // Emulates __funnelshift_l(hi, lo, 1) for any type (not just uint32)
{
	constexpr size_t word_size = sizeof(T) * 8;
	constexpr T MSB_mask = T(1) << (word_size - 1);

	return (hi << 1) | bool(lo & MSB_mask);
}

template <>
__device__ __forceinline__ uint32_t funnelshift_left(const uint32_t& lo, const uint32_t& hi) // If working with uint32, use the special integer intrinsic
{
	return __funnelshift_l(lo, hi, 1);
}


/* This is the bit-RLE encode kernel
 * Template parameters:
 * - InputIteratorT: the iterator type used to manipulate the bitset (usually uint64_t* or uint32_t*).
 * - nThreads: number of threads on which the kernel is launched (used for CUB's block-wide primitives). Can we infer this from kernel launch parameters ?
 *
 * Parameters:
 * - bitset: the bitset to encode.
 * - n_words: the number of words to encode from bitset.
 * - elems: output vector of RLE elements (position and duration of each pulse)
 * - maxElems: max number of pulses to encode
 * - block_status: temporary array used to synchronize blocks
 * 
 * Steps of this kernel:
 * 1. Rising and falling edges generation
 *     1.1. Each thread of each block reads one word of the bitset (corresponding to its thread rank in the device), and the previous word
 *          TODO: shared memory could be used here to halve the number of global memory reads
 *     1.2. Using these two words, we can shift the portion of the bitset by one bit (<==> funnelshift)
 *     1.3. Compute the rising and falling edges of the input signal at word level
 *
 * 2. Device wide prefix sum (used to do stream compaction and store RLE elements at the right place)
 *     2.1. "Expand" the rising and falling edges words to uint32 arrays
 *     2.2. Compute the per-block prefix sum and catch the block-wide aggregate 
 *          TODO: compute the per-block aggregate only for slightly better performances, and use a BlockPrefixCallbackOp
 *          later on to set the global prefix.
 *     2.3. Set the block-wide aggregate in the `block_status` array, with a flag to indicate that this block's aggregate is ready
 *          NOTE: I used the MSB of the falling edges prefix sum as flag, but this might not be a good idea...
 *     2.4. Poll `block_status` using as many threads as possible to gather previous blocks' aggregates.
 *          This is on my opinion the most critical part, polling with each thread might be too intense.
 *          From what I saw in CUB's source code, they use a __threadfence_block at this step, I'll try.
 *          That is also where I tried the grid_group::sync, without any success.
 *          TODO: use CUB's ScanTileState class for an already working solution
 *     2.5. Sum all the aggregates gathered
 *     2.6. Add the global prefix to the previously computed block-wide prefix sum (or actually do the prefix sum now if following the alternative of 2.2)
 *
 * 3. Write RLE elements
 *     3.1. if we are on a rising edge:
 *              - write the index of the rising edge as position. The `bitrle_elem` to write is determined by the rising edge prefix sum.
 *              - atomically substract the index to the duration of the same element.
 *     3.2. if we are on a falling edge:
 *              - atomically add the index to the duration of the appropriate element (found using the falling edge prefix sum)
 *     3.3. the last thread of the last block sets the final number of pulses, and rounds up the last element if we finish without a falling edge
 *          EXAMPLE: 0000111000001111: no falling edge at the end -> bad logic, needs a correction
 */
template <typename InputIteratorT, int32_t nThreads>
__global__ void bitrle_encode(InputIteratorT bitset, const size_t n_words, bitrle_elem* elems, const uint32_t maxElems, uint2* block_status)
{
	using InputT = typename std::iterator_traits<InputIteratorT>::value_type;
	constexpr size_t word_size = sizeof(InputT) * 8;

	typedef cub::BlockReduce<uint2, nThreads> BlockReduce;
	typedef cub::BlockScan<uint2, nThreads>   BlockScan;

	__shared__ union {
		typename BlockReduce::TempStorage reduce;
		typename BlockScan::TempStorage   scan;
	} temp_storage;

	__shared__ uint2 block_prefix;


	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
// Note: no `if (tid < n_words)` or so, because it was causing __syncthreads() and cooperative groups issues

// 1.
// 1.1.
	const InputT cur  = (tid < n_words) ? reinterpret_cast<const InputT*>(bitset)[tid] : 0; // Care is taken to respect boundaries
	const InputT prev = (0 < tid) ? (tid <= n_words) ? reinterpret_cast<const InputT*>(bitset)[tid - 1] : 0 : 0;
// 1.2.
	const InputT tm1  = funnelshift_left(prev, cur);

// 1.3.
	const InputT rising_edges  = ~tm1 & cur;
	const InputT falling_edges = tm1 & ~cur;


// 2.
	uint2 expanded[word_size];
	uint2 indices[word_size];
// 2.1.
#pragma unroll
	for (int j = 0; j < word_size; j++)
	{
		expanded[j].x = cub::BFE(rising_edges, j, 1);
		expanded[j].y = cub::BFE(falling_edges, j, 1);
	}

// 2.2.
	uint2 block_aggregate;
	BlockScan(temp_storage.scan).ExclusiveSum(expanded, indices, block_aggregate);
	
// 2.3.
	if (threadIdx.x == 0)
	{
		block_status[blockIdx.x] = set_ready_flag(block_aggregate);
		block_prefix = uint2{0, 0};
	}

	__syncthreads();

// 2.4.
// This for loop dispatches threads to poll blocks (especially useful if blockDim.x < blockIdx.x), while avoiding thread divergence (bugs with __syncthreads() / cooperative groups)
	for (int j = threadIdx.x; j < ((blockIdx.x + blockDim.x - 1) / blockDim.x) * blockDim.x; j += blockDim.x)
	{
		uint2 previous_aggregate{0, 0}; // Default value in case the thread does not have to poll

		if (j < blockIdx.x) // These threads do have to poll
		{
			do
			{
				__threadfence_block();
				previous_aggregate = block_status[j];
			} while (!get_ready_flag(previous_aggregate)); // Loop while the aggregate is not set by block j
			previous_aggregate = reset_ready_flag(previous_aggregate);
		}

// 2.5.
		previous_aggregate = BlockReduce(temp_storage.reduce).Sum(previous_aggregate);

		if (threadIdx.x == 0)
			block_prefix += previous_aggregate;
		__syncthreads();
	}

#pragma unroll
	for (int j = 0; j < word_size; j++)
	{
// 2.6.
		indices[j] += block_prefix;

// 3.
		const uint32_t pos = tid * word_size + j;
// 3.1
		if (expanded[j].x && (indices[j].x < maxElems)) // Rising edge
		{
			elems[indices[j].x].position = pos;
			elems[indices[j].x].duration -= pos;
		}
// 3.2
		else if (expanded[j].y && indices[j].y < maxElems) // Falling edge
		{
			elems[indices[j].y].duration += pos;
		}
	}

// 3.3
	if (tid == n_words - 1)
	{
		if ((indices[word_size - 1].x < maxElems) && (indices[word_size - 1].x > indices[word_size - 1].y))
			elems[indices[word_size - 1].x].duration += n_words * word_size;
		n_pulses = min(indices[word_size - 1].x, maxElems);
	}
}





int main(int argc, char* argv[])
{
	using BitsetWordT = uint64_t; // Underlying type of the bitset
	constexpr size_t word_size = sizeof(BitsetWordT) * 8; // Number of bits in one word

	const size_t N = 1 << 20; // Number of bits in the bitset
	const size_t n_words = (N + word_size - 1) / word_size; // Enough words to hold all the bits
	const size_t maxElems = 10000; // Max number of pulses for the RLE

	BitsetWordT* d_bitset;
	checkCudaErrors(cudaMalloc(&d_bitset, n_words * sizeof(BitsetWordT)));

	bitrle_elem* d_bitrle;
	checkCudaErrors(cudaMalloc(&d_bitrle, maxElems * sizeof(bitrle_elem)));
	checkCudaErrors(cudaMemset(d_bitrle, 0, maxElems * sizeof(bitrle_elem)));


	fill_kernel<<<64, 512>>>(d_bitset, n_words);
	checkCudaErrors(cudaDeviceSynchronize());


	const size_t nThreads = 256;
	const size_t nBlocks = (((N + word_size - 1) / word_size) + nThreads - 1) / nThreads;

	uint2* block_status;
	checkCudaErrors(cudaMalloc(&block_status, nBlocks * sizeof(uint2)));
	checkCudaErrors(cudaMemset(block_status, 0, nBlocks * sizeof(uint2)));


	bitrle_encode<BitsetWordT*, nThreads><<<nBlocks, nThreads>>>(d_bitset, n_words, d_bitrle, maxElems, block_status);
	checkCudaErrors(cudaDeviceSynchronize());

	check_results<<<1, 1>>>(d_bitrle);
	checkCudaErrors(cudaDeviceSynchronize());

	return 0;
}

