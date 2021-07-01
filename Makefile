NVCC=/usr/local/cuda/bin/nvcc
NCU=/usr/local/cuda/bin/ncu
# CUDA_FLAGS=-g -G
CUDA_FLAGS=-O3


bitrle_encode_example: bitrle_encode_example.cu bitrle_encode_utils.cuh
	$(NVCC) bitrle_encode_example.cu -o bitrle_encode_example $(CUDA_FLAGS)


profile: bitrle_encode_example
	$(NCU) --set full -k bitrle_encode --print-summary per-kernel -f -o bitrle_encode ./bitrle_encode_example


clean:
	rm bitrle_encode_example

