#ifndef ANG_H
#define ANG_H


// this file will call populate.cu and human.cu


#pragma once

#include <math.h>

#include "util/error_utils.cuh"
#include "data.cuh"

#include "populate.cu"
#include "human.cu"

using namespace std;

void AngelaKernels( Square* h_unsolved, Square* d_unsolved, Square* d_solved, int n) {

	// TODO: memcpy
	// TODO: set grid/TPB
	// TODO: call populate kernel (located in populate.cu)
	// TODO: after populate works, test human.cu
	// TODO: after every human call, need to populate again


	int memsize = sizeof(Square) * n * n;

	ERROR_CHECK( cudaMemcpy(d_unsolved, h_unsolved, memsize,
         	cudaMemcpyHostToDevice) );

	int threadsPerBlock = n*n;
	int blocksPerGrid = (n + threadsPerBlock -1) / threadsPerBlock;
//	int blocksPerGrid = 1;

	populate<<<blocksPerGrid, threadsPerBlock>>>(d_unsolved);

	ERROR_CHECK( cudaPeekAtLastError() );
	ERROR_CHECK( cudaDeviceSynchronize() );

	ERROR_CHECK( cudaMemcpy(h_unsolved, d_unsolved, memsize,
		cudaMemcpyDeviceToHost) );

	debug_values(h_unsolved);

	  int* d_points;
	  ERROR_CHECK( cudaMalloc((void**) &d_points, sizeof(int)) );

	human<<<blocksPerGrid, threadsPerBlock>>>(d_unsolved, n, d_points);

	ERROR_CHECK( cudaPeekAtLastError() );
	ERROR_CHECK( cudaDeviceSynchronize() );
	
//	ERROR_CHECK( cudaMemcpy(h_unsolved, d_unsolved, memsize,
//		cudaMemcpyDeviceToHost) );


}
#endif
