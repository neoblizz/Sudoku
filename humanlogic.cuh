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

void HumanLogic (Square * h_unsolved, Square * d_unsolved,
								 Square * d_solved, int n) {

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

	 	 int* d_points;
		  ERROR_CHECK( cudaMalloc((void**) &d_points, sizeof(int)) );

		int* h_points = (int*) malloc(sizeof(int));
//		ERROR_CHECK( cudaMemcpy(h_points, d_points, sizeof(int),
//			cudaMemcpyDeviceToHost));

//for(int jj=0; jj<1; jj++) {
	populate<<<blocksPerGrid, threadsPerBlock>>>(d_unsolved);

	ERROR_CHECK( cudaPeekAtLastError() );
	ERROR_CHECK( cudaDeviceSynchronize() );

	ERROR_CHECK( cudaMemcpy(h_unsolved, d_unsolved, memsize,
		cudaMemcpyDeviceToHost) );

	debug_values(h_unsolved);

//	  int* d_points;
//	  ERROR_CHECK( cudaMalloc((void**) &d_points, sizeof(int)) );

	human<<<blocksPerGrid, threadsPerBlock>>>(d_unsolved, n, d_points);

	ERROR_CHECK( cudaPeekAtLastError() );
	ERROR_CHECK( cudaDeviceSynchronize() );

//		int* h_points = (int*) malloc(sizeof(int));
		ERROR_CHECK( cudaMemcpy(h_points, d_points, sizeof(int),
			cudaMemcpyDeviceToHost));

	printf("Amount of work done this round is %d.\n", *h_points);
//}

	ERROR_CHECK( cudaMemcpy(h_unsolved, d_unsolved, memsize,
		cudaMemcpyDeviceToHost) );


     const char * finished = "/********** Angela's (C) **********/";
    output(finished, "-alg", n, false, h_unsolved);

//round 2
	populate<<<blocksPerGrid, threadsPerBlock>>>(d_unsolved);

	ERROR_CHECK( cudaPeekAtLastError() );
	ERROR_CHECK( cudaDeviceSynchronize() );

	ERROR_CHECK( cudaMemcpy(h_unsolved, d_unsolved, memsize,
		cudaMemcpyDeviceToHost) );

	debug_values(h_unsolved);

//	  int* d_points;
//	  ERROR_CHECK( cudaMalloc((void**) &d_points, sizeof(int)) );

	human<<<blocksPerGrid, threadsPerBlock>>>(d_unsolved, n, d_points);

	ERROR_CHECK( cudaPeekAtLastError() );
	ERROR_CHECK( cudaDeviceSynchronize() );

//		int* h_points = (int*) malloc(sizeof(int));
		ERROR_CHECK( cudaMemcpy(h_points, d_points, sizeof(int),
			cudaMemcpyDeviceToHost));

	printf("Amount of work done this round is %d.\n", *h_points);

	ERROR_CHECK( cudaMemcpy(h_unsolved, d_unsolved, memsize,
		cudaMemcpyDeviceToHost) );

//    const char * finished = "/********** Angela's (C) **********/";
    output(finished, "-bee", n, false, h_unsolved);


}
#endif
