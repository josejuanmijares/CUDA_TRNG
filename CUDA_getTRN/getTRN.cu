/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */


#include <thrust/host_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <cuda.h>

using namespace std;

#define SIZE 16
#define MAX_ITERATIONS 1024


/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void TRNGkernel(int M, int S)
{
	extern __shared__ int shared[];
	int *s = &shared[0];
	int *out = &shared[S+1];
	int k=0;

	int id =threadIdx.x;

	do{
		s[id] = id;
		out[id]=s[id+1];
		k++;
	} while(k<M);
}

void generate_raw_numbers(thrust::host_vector<float> *RN , int S, int M, int N, float CL){

	int i;
	cudaEvent_t start, stop;
	float kernelTime;

	for(i=0;i<N;i++)
	{
		// activate the timer in the graphic card
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord( start, 0 );

		// kernel with the race conditions
		TRNGkernel<<<1,S,(2*S+1)*sizeof(int)>>>(M, S);

		cudaThreadSynchronize();	// Wait for the GPU launched work to complete

		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &kernelTime, start, stop );
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		// stores the raw numbers
		(*RN)[i] = fmod(kernelTime*CL,float(1.0));

	};

	cudaDeviceReset();
}

int main(int argc, char *argv[]) {

	int N_REPETITIONS = atoi(argv[1]);

	thrust::host_vector<float> RawNumbers(N_REPETITIONS);

	float CompressionLevel = 100.00;
	generate_raw_numbers(&RawNumbers,SIZE,MAX_ITERATIONS,N_REPETITIONS, CompressionLevel);

	for (int k=0;k<N_REPETITIONS;k++)
		cout<< RawNumbers[k] << "  ";
	cout << endl;

	cudaThreadExit();
	return 0;
}
