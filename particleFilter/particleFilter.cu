
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
#include <Windows.h>

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#define NOISE
#define M_PI 3.14159265359

#define T 200
#define N 10000
#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 32

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
float randomProcessData[(N + 1) * T];
float randomUniformData[N];

float real[T];
float est[T];

__global__ void genWeights(int t, float z, float x_R, float *randomProcess, float *x_p, float *z_update, float *x_p_update, float *p_w) {
	int id =  blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ float s[];
	float sum = 0.0f;
	if (id * ELEMENTS_PER_THREAD < N) {
		s[id] = 0.0f;
		for (int i = id * ELEMENTS_PER_THREAD; i < N && i < id + ELEMENTS_PER_THREAD; i++) {
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
		for (int i = 0; i < N / ELEMENTS_PER_THREAD; i++) {
			sum += s[id];
		}
	}
	// Normalize
	__syncthreads();
	if (id * ELEMENTS_PER_THREAD < N) {
		for (int i = id * ELEMENTS_PER_THREAD; i < N && i < id + ELEMENTS_PER_THREAD; i++) {
			p_w[i] = p_w[i] / sum;
		}
	}
}

__global__ void partialSum(float *data, int number, float *result) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id * ELEMENTS_PER_THREAD < number) {
		for (int i = id * ELEMENTS_PER_THREAD; i < number && i < id + ELEMENTS_PER_THREAD; i++) {
			float sum = 0;
			for (int j = 0; j <= i; j++){
				sum += data[j];
			}
			result[i] = sum;
		}
	}
}

__global__ void resample(float *random, float *sum, float *cumsum, float *x_p, float *x_p_update) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ float s [];
	*sum = 0.0f;
	if (id * ELEMENTS_PER_THREAD < N) {
		s[id] = 0.0f;
		for (int i = id * ELEMENTS_PER_THREAD; i < N && i < id + ELEMENTS_PER_THREAD; i++) {
			for (int j = 0; j < N; j++) {
				if (random[i] <= cumsum[j]) {
					x_p[i] = x_p_update[j];
					s[id] += x_p[i];
					break;
				}
			}
		}
	}
	__syncthreads();
	// Calculate sum
	if (id == 0) {
		for (int i = 0; i < N / ELEMENTS_PER_THREAD; i++) {
			*sum += s[id];
		}
	}
}

int main()
{
	nvtxNameOsThread(GetCurrentThreadId(), "MAIN" );
	nvtxRangePush(__FUNCTION__ SETTINGSSTRING);
	nvtxRangePush("Initializing");
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

	int blocks = (N + THREADS_PER_BLOCK -1) / (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD);
	int threads = THREADS_PER_BLOCK;

	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<float> normal_start(0, V);
	std::normal_distribution<float> normal_process(0, x_N);
	std::normal_distribution<float> normal_meas(0, x_R);
	std::uniform_real_distribution<float> uniform_resamp(0, 1);
	
	float z_out = x*x / 20 + normal_meas(generator);
	float x_est = x;
	nvtxRangePush("GenParticles");
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
		nvtxRangePush("Update");
		// Create random data for genWeights
		for (int i = 0; i < N; i++) {
			randomProcessData[i] = normal_process(generator);
		}
		cudaMemcpy(d_randomProcessData, randomProcessData, sizeof(float) * N, cudaMemcpyHostToDevice);

		int sharedMemSize = sizeof(float) * (N + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
		genWeights << <blocks, threads, sharedMemSize >> >(t, z, x_R, d_randomProcessData, 
														   d_x_p, d_z_update, d_x_p_update, d_p_w);
		nvtxRangePop();
		nvtxRangePush("Resampling");
		partialSum <<<blocks, threads >>>(d_p_w, N, d_cumsum);
		// Create random data for resampling
		for (int i = 0; i < N; i++) {
			randomUniformData[i] = uniform_resamp(generator);
		}
		cudaMemcpy(d_randomUniformData, randomUniformData, sizeof(float) * N, cudaMemcpyHostToDevice);
		resample << <blocks, threads, sharedMemSize >> >(d_randomUniformData, d_sum, d_cumsum, d_x_p, d_x_p_update);
		float sum;
		cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
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
