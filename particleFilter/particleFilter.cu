
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>
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

#define NOISE
#define M_PI 3.14159265359

#define T 200
#define N 10000

#define XSTR(s) STR(s)
#define STR(s) #s
#ifdef NOISE
#define SETTINGSSTRING "NOISE, T=" XSTR(T) ", N=" XSTR(N)
#else
#define SETTINGSSTRING "T=" XSTR(T) ", N=" XSTR(N)
#endif

float x_p_update[N];
float z_update[N];
float x_p[N];
float p_w[N];
float cumsum[N];

float real[T];
float est[T];

int main()
{
	nvtxNameOsThread(GetCurrentThreadId(), "MAIN" );
	nvtxRangePush(__FUNCTION__ SETTINGSSTRING);
	nvtxRangePush("Initializing");
	//cudaProfilerStart();
	// Init CUDA


	float x = 0.1f;
	float z;
	float x_N = 1;
	float x_R = 1;

	float V = 10;
	float sum;

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
		sum = 0.0f;
		nvtxRangePush("Update");
		nvtxRangePush("GenWeights");
		for (int i = 0; i < N; i++) {
#ifdef NOISE
			x_p_update[i] = 0.5f*x_p[i] + 25 * x_p[i] / (1 + x_p[i] * x_p[i]) + 8 * cosf(1.2f*t) + normal_process(generator);
#else
			x_p_update[i] = 0.5f*x_p[i] + 25 * x_p[i] / (1 + x_p[i] * x_p[i]) + 8 * cosf(1.2f*t);
#endif
			z_update[i] = x_p_update[i] * x_p_update[i] / 20;
			p_w[i] = (1 / sqrtf(2 * M_PI * x_R)) * exp(-1 * powf(z - z_update[i], 2) / (2 * x_R));
			sum = sum + p_w[i];
		}
		nvtxRangePop();
		nvtxRangePush("Normalize");
		for (int i = 0; i < N; i++) 
		{
			p_w[i] = p_w[i] / sum;
		}
		nvtxRangePop();
		nvtxRangePop();
		nvtxRangePush("Resampling");
		std::partial_sum(p_w, p_w + N, cumsum);
		sum = 0.0f;
		for (int i = 0; i < N; i++) {
			float uniform = uniform_resamp(generator);
			nvtxRangePush("FindRand");
			for (int j = 0; j < N; j++) {
				if (uniform <= cumsum[j]) {
					x_p[i] = x_p_update[j];
					sum = sum + x_p[i];
					break;
				}
			}
			nvtxRangePop();
		}
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
	cudaProfilerStop();
	return 0;
}
