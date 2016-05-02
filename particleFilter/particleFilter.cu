
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>

#include <random>
#include <stdio.h>
#include <cmath>
#include <numeric>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <Windows.h>

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#define NOISE
#define M_PI 3.14159265359

//#define COPY_DATA_BACK 1

#define T 200
#define N 10000
#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 32

#define THREADS_SUM 256

#define XSTR(s) STR(s)
#define STR(s) #s
#ifdef NOISE
#define SETTINGSSTRING "NOISE, T=" XSTR(T) ", N=" XSTR(N) ", TPB=" XSTR(THREADS_PER_BLOCK)
#else
#define SETTINGSSTRING "T=" XSTR(T) ", N=" XSTR(N) ", TPB=" XSTR(THREADS_PER_BLOCK)
#endif

float *d_x_p_update;
float *d_z_update;
float *d_x_p;
float *d_p_w;
float *d_cumsum;
float *d_randomProcessData;
float *d_randomUniformData;
float *d_sum;

float x_p_update[N];
float z_update[N];
float x_p[N];
float p_w[N];
float cumsum[N];
float randomProcessData[N];
float randomUniformData[N];

float real[T];
float est[T];
nvtxEventAttributes_t markInit;
nvtxEventAttributes_t markGen;
nvtxEventAttributes_t markWeigh;
nvtxEventAttributes_t markNormalize;
nvtxEventAttributes_t markResample;

void initMarkers() {
	markInit.version = NVTX_VERSION;
	markInit.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	markInit.category = 0;
	markInit.color = 0xFF880000;
	markInit.colorType = NVTX_COLOR_ARGB;
	markInit.message.ascii = "Initializing";
	markInit.messageType = NVTX_MESSAGE_TYPE_ASCII;
	markInit.payload.llValue = 0;
	markInit.payloadType = NVTX_PAYLOAD_TYPE_INT64;
	markInit.reserved0 = 0;
	markGen = markInit;
	markGen.message.ascii = "Generating Particles";
	markWeigh = markInit;
	markWeigh.message.ascii = "Generating weights";
	markWeigh.color = 0xFF00FF00;
	markWeigh.payload.llValue = 1;
	markNormalize = markInit;
	markNormalize.message.ascii = "Normalizing";
	markNormalize.color = 0xFF0000FF;
	markNormalize.payload.llValue = 2;
	markResample = markInit;
	markResample.message.ascii = "Resampling";
	markResample.color = 0xFF00FFFF;
	markResample.payload.llValue = 3;

}

__global__ void genWeights(int t, float z, float x_R, float *randomProcess, float *x_p, float *z_update, float *x_p_update, float *p_w) {
	int id =  threadIdx.y * blockDim.x + threadIdx.x;
	extern __shared__ float s[];
	float sum = 0.0f;
	if (id * ELEMENTS_PER_THREAD < N) {
		s[id] = 0.0f;
		for (int i = id * ELEMENTS_PER_THREAD; i < N && i < (id + 1) * ELEMENTS_PER_THREAD; i++) {
#ifdef NOISE
			float update = 0.5f*x_p[i] + 25 * x_p[i] / (1 + x_p[i] * x_p[i]) + 8 * cosf(1.2f*t) + randomProcess[i];
#else
			float update = 0.5f*x_p[i] + 25 * x_p[i] / (1 + x_p[i] * x_p[i]) + 8 * cosf(1.2f*t);
#endif
			z_update[i] = update * update / 20.0f;
			x_p_update[i] = update;
			float weight = (1 / sqrtf(2 * M_PI * x_R)) * exp(-1 * powf(z - z_update[i], 2) / (2 * x_R));
			p_w[i] = weight;
			s[id] += weight;
		}
	}
	__syncthreads();
	// Calculate sum
	if (id == 0) {
		for (int i = 0; i * ELEMENTS_PER_THREAD < N ; i++) {
			sum += s[i];
		}
		s[0] = sum;
	}
	// Normalize
	__syncthreads();
	if (id * ELEMENTS_PER_THREAD < N) {
		for (int i = id * ELEMENTS_PER_THREAD; i < N && i < (id + 1) * ELEMENTS_PER_THREAD; i++) {
			p_w[i] = p_w[i] / s[0];
		}
	}
}

__global__ void cumulativeSum(float *data, int number, float *result) {
	int id = threadIdx.y * blockDim.x + threadIdx.x;
	if (id * ELEMENTS_PER_THREAD < number) {
		for (int i = id * ELEMENTS_PER_THREAD; i < number && i < (id + 1) * ELEMENTS_PER_THREAD; i++) {
			float sum = 0;
			for (int j = 0; j <= i; j++){
				sum += data[j];
			}
			result[i] = sum;
		}
	}
}

__device__ void SearchFirst(float *data, float value, int *start, int *end, int id, int numThreads) {
	int dataLen = *end - *start + 1;
	int searchSize = (dataLen + numThreads - 1) / numThreads;
	int threadStart = *start + searchSize * id;
	int threadEnd = threadStart + searchSize;

	if (threadStart >= *end)	return;
	if (threadEnd > *end) threadEnd = *end;

	if (data[threadStart] <= value && data[threadEnd] > value) {
		*start = threadStart;
		*end = threadEnd;
	}
}

__global__ void resample(float *random, float *cumsum, float *x_p, float *x_p_update) {
	int particleId = blockIdx.x;
	int id = threadIdx.x;
	__shared__ int start, end;
	if (particleId < N) {
		float searchValue = random[particleId];
		int correctIdx = -1;
		/*int correctIdxCheck = N - 1;
		for (int j = 0; j < N; j++) {
			if (searchValue <= cumsum[j]) {
				correctIdxCheck = j;
				break;
			}
		}*/
		if (id == 0) {
			start = 0;
			end = N - 1;
			// Rare case when there is no particle that fullfills the condition
			if (searchValue >= cumsum[end]) {
				start = end - 1;
			}
			else if (searchValue < cumsum[start]) {
				end = start + 1;
			}
		}
		__syncthreads();
		while (start != end-1) {
			SearchFirst(cumsum, searchValue, &start, &end, id, blockDim.x);
			__syncthreads();
		}
		if (id == 0) {
			if (cumsum[start] >= searchValue) {
				correctIdx = start;
			}
			else {
				correctIdx = end;
			}
			//assert(correctIdx == correctIdxCheck);
			x_p[particleId] = x_p_update[correctIdx];
		}
	}
}

__global__ void sum(const float *data, int size, float *result) {
	extern __shared__ float partSum[];
	int id = threadIdx.x;
	int numThreads = blockDim.x;
	partSum[id] = 0.0f;
	int pos = id;
	while (pos < size) {
		partSum[id] += data[pos];
		pos += numThreads;
	}
	__syncthreads();
	do {
		numThreads /= 2;
		if (id >= numThreads) return;
		partSum[id] = partSum[id] + partSum[id + numThreads];
		__syncthreads();
	} while (numThreads > 1);
	*result = partSum[0];
}

int main()
{
	initMarkers();
	nvtxNameOsThread(GetCurrentThreadId(), "MAIN" );
	nvtxRangePush(__FUNCTION__ SETTINGSSTRING);
	nvtxRangePushEx(&markInit);
	cudaProfilerStart();
	
	// Init CUDA
	cudaSetDevice(0);
	cudaMalloc<float>(&d_x_p_update, sizeof(float) * N);
	cudaMalloc<float>(&d_z_update, sizeof(float) * N);
	cudaMalloc<float>(&d_x_p, sizeof(float) * N);
	cudaMalloc<float>(&d_p_w, sizeof(float) * N);
	cudaMalloc<float>(&d_cumsum, sizeof(float) * N);
	cudaMalloc<float>(&d_randomProcessData, sizeof(float) * N);
	cudaMalloc<float>(&d_randomUniformData, sizeof(float) * N);
	cudaMalloc<float>(&d_sum, sizeof(float));

	float x = 0.1f;
	float z;
	float x_N = 1;
	float x_R = 1;

	float V = 10;

	dim3 blocks = dim3(1, 1, 1);
	dim3 threads = dim3(THREADS_PER_BLOCK,
						(N + THREADS_PER_BLOCK * ELEMENTS_PER_THREAD - 1) / (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD), 1);
	int sharedMemSize = sizeof(float) * (N + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;

	// Random seed
	//std::random_device rd;
	//int seed = rd();
	// Fixed seed
	int seed = 1234;
	std::mt19937 generator(seed);
	std::normal_distribution<float> normal_start(0, V);
	std::normal_distribution<float> normal_process(0, x_N);
	std::normal_distribution<float> normal_meas(0, x_R);
	std::uniform_real_distribution<float> uniform_resamp(0, 1);
	
	float z_out = x*x / 20 + normal_meas(generator);
	float x_est = x;
	nvtxRangePushEx(&markGen);
	for (int i = 0; i < N; i++) {
		x_p[i] = x + normal_start(generator);
	}
	cudaMemcpy(d_x_p, x_p, sizeof(float) * N, cudaMemcpyHostToDevice);
	
	nvtxRangePop();
	nvtxRangePop();
	nvtxRangePush("Executing");
	
	for (int t = 0; t < T; t++) {

		real[t] = x;
#ifdef NOISE
		x = 0.5f*x + 25 * x / (1 + x*x) + 8 * cosf(1.2f*t) + normal_process(generator);
		z = x*x / 20 + normal_meas(generator);
#else
		x = 0.5f*x + 25 * x / (1 + x*x) + 8 * cosf(1.2f*t);
		z = x*x / 20;
#endif
		//nvtxRangePush("Update");
		// Create random data for genWeights
		for (int i = 0; i < N; i++) {
			randomProcessData[i] = normal_process(generator);
		}
		cudaMemcpy(d_randomProcessData, randomProcessData, sizeof(float) * N, cudaMemcpyHostToDevice);

		nvtxRangePushEx(&markWeigh);
		genWeights << <blocks, threads, sharedMemSize >> >(t, z, x_R, d_randomProcessData, 
														   d_x_p, d_z_update, d_x_p_update, d_p_w);
		nvtxRangePop();
#ifdef COPY_DATA_BACK
		cudaMemcpy(x_p, d_x_p, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(x_p_update, d_x_p_update, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(z_update, d_z_update, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_w, d_p_w, sizeof(float) * N, cudaMemcpyDeviceToHost);
#endif
		//nvtxRangePop();
		nvtxRangePushEx(&markResample);

		cumulativeSum << <blocks, threads >> >(d_p_w, N, d_cumsum);
#ifdef COPY_DATA_BACK
		cudaMemcpy(cumsum, d_cumsum, sizeof(float) * N, cudaMemcpyDeviceToHost);
#endif
		// Create random data for resampling
		for (int i = 0; i < N; i++) {
			randomUniformData[i] = uniform_resamp(generator);
		}
		cudaMemcpy(d_randomUniformData, randomUniformData, sizeof(float) * N, cudaMemcpyHostToDevice);
		resample << <N, 32 >> >(d_randomUniformData, d_cumsum, d_x_p, d_x_p_update);
		int sharedMemSumSize = THREADS_SUM * sizeof(float);
		sum << <1, THREADS_SUM, sharedMemSumSize >> >(d_x_p, N, d_sum);
		float sum;
		cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
#ifdef COPY_DATA_BACK
		cudaMemcpy(x_p, d_x_p, sizeof(float) * N, cudaMemcpyDeviceToHost);
#endif
		nvtxRangePop();
		est[t] = x_est;
		x_est = sum / N;
		printf("estimation: %4.2f real: %4.2f \n", x_est, x);
	}
	nvtxRangePop();
	nvtxRangePush("WriteFile");
	std::ofstream out("real.txt");
	std::ofstream out2("est.txt");

	// Make it a fixed value extending 5 digits past the decimal point
	for (int i = 0; i < T; i++) {
		out << std::fixed << std::setprecision(5) << real[i] << std::endl;
		out2 << std::fixed << std::setprecision(5) << est[i] << std::endl;
	}
	out.close();
	out2.close();
	nvtxRangePop();
	nvtxRangePop();
	
	cudaFree(d_cumsum);
	cudaFree(d_p_w);
	cudaFree(d_randomProcessData);
	cudaFree(d_randomUniformData);
	cudaFree(d_x_p);
	cudaFree(d_x_p_update);
	cudaFree(d_z_update);

	cudaProfilerStop();



	return 0;
}
