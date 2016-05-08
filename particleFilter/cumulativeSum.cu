
#include "cumulativeSum.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

static float *d_cumsumBlockSum;
static int cumsumBlockSumSize;

__global__ void cumulativeSumStep1(float *idata, float *odata, float *blockSums, int n)
{
	// Algorithm from GPU Gems 3 chapter 39
	extern __shared__ float s [];
	float *temp = s;
	int thid = threadIdx.x;
	int bid = blockIdx.x;
	int id = blockIdx.x * blockDim.x + thid;
	int numElements = blockDim.x * 2;
	int offset = 1;

	// Load input into shared memory
	if (2 * id >= n) {
		temp[2 * thid] = 0;
	}
	else {
		temp[2 * thid] = idata[2 * id];
	}
	if (2 * id + 1 >= n) {
		temp[2 * thid + 1] = 0;
	}
	else {
		temp[2 * thid + 1] = idata[2 * id + 1];
	}

	// build sum in place up the tree
	for (int d = numElements >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;

	}
	if (thid == 0) {
		temp[numElements] = temp[numElements - 1];
		temp[numElements - 1] = 0;
	}

	for (int d = 1; d < numElements; d *= 2){
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	if (2 * id < n){
		odata[2 * id] = temp[2 * thid];
	}
	if (2 * id + 1 < n) {
		odata[2 * id + 1] = temp[2 * thid + 1];
	}
	if (blockSums != nullptr && thid == 0) {
		blockSums[bid] = temp[numElements];
	}
}

__global__ void cumulativeSumStep2(float *odata, float *blockSums, int n) {
	int thid = threadIdx.x;
	int bid = blockIdx.x;
	int id = blockIdx.x * blockDim.x + thid;
	float val = blockSums[bid];
	if (id * 2 >= n) return;
	odata[id * 2] += val;
	if (id * 2 + 1 >= n) return;
	odata[id * 2 + 1] += val;
	if (id == n / 2 - 1) {
		odata[n] = blockSums[bid+1];
	}
}

void cumulativeSumInit()
{
	d_cumsumBlockSum = nullptr;
	cumsumBlockSumSize = 0;
}

void cumulativeSumFree()
{
	if (d_cumsumBlockSum != nullptr) {
		cudaFree(d_cumsumBlockSum);
		d_cumsumBlockSum = nullptr;
		cumsumBlockSumSize = 0;
	}
}

bool isPowerOfTwo(int x) {
	return ((x & (x - 1)) == 0);
}

void cumulativeSum(float *in, float *out, int n)
{
	// Function taken from NVIDIA CUDA SDK samples
	unsigned int numThreads = THREADS_CUMSUM;
	unsigned int numBlocks = (int) ceil((float) n / (2.0f * numThreads));
	if (numBlocks < 1) numBlocks = 1;
	
	if (cumsumBlockSumSize < numBlocks + 1) {
		if (d_cumsumBlockSum != nullptr) {
			cudaFree(d_cumsumBlockSum);
			d_cumsumBlockSum = nullptr;
			cumsumBlockSumSize = 0;
		}
		cudaMalloc(&d_cumsumBlockSum, sizeof(float) * (numBlocks + 1));
		cumsumBlockSumSize = numBlocks + 1;
	}

	

	cumulativeSumStep1 << <numBlocks, numThreads, sizeof(float) * (numThreads * 2 + 1) >> >(in, out, d_cumsumBlockSum, n);
	int numThreadsBlock = numBlocks / 2;
	numThreadsBlock = (int)powf(2, ceil(logf(numThreadsBlock) / logf(2)));
	cumulativeSumStep1 << <1, numThreadsBlock, sizeof(float) * (numBlocks + 1) >> >(d_cumsumBlockSum, d_cumsumBlockSum, &d_cumsumBlockSum[numBlocks], numBlocks);
	cumulativeSumStep2 << <numBlocks, numThreads >> >(out, d_cumsumBlockSum, n);

}