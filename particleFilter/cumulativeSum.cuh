#pragma once

#define THREADS_CUMSUM 256

void cumulativeSumInit();
void cumulativeSumFree();
void cumulativeSum(float *in, float *out, int n);