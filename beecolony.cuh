#ifndef BEE_H
#define BEE_H

// ----------------------------------------------------------------
// Sudoku -- Puzzle Solver on GPU using CUDA
// ----------------------------------------------------------------

#pragma once

#include <string.h>

#include <iostream>
#include <fstream>

#include <curand.h>
#include <curand_kernel.h>


#include "util/error_utils.cuh"
#include "data.cuh"

#define CHAINS 15
#define CYCLES (1<<10)

using namespace std;

/**
 * @file
 * beecolony.cuh
 *
 * @brief ArtificialBeeColony algorithm implementation.
 */

/**
 * @brief Random number generator across cores for cuda.
 */
__global__ void cuRandomNumberGenerator (curandState *state) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  /* we have to initialize the state */
  curand_init(1337,   /* the seed controls the sequence of random values that are produced */
              index,  /* the sequence number is only important with multiple cores */
              0,      /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state[index]);

}


/**
 * @brief Determine number of unique elements in a row or col.
 */
__device__ int foodQuality () {

}

/**
 * @brief Used to determines the quality of the solution by calculating
 * uniqueness of an elements.
 */
__device__ int hiveQuality () {

}

__global__ void BeeColony (Square * d_unsolved, Square * d_solved, int n, curandState *state) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  // TODO: Copy d_unsolved to shared memory;
  __shared__ Square sudoku[9][9];

  for(int i = 0; i < CYCLES; i++) {
    // USAGE: number = (int) curand_uniform(&state[index]);
    // TODO: Scout to begin the process.
    // TODO: Randomly select a block.
    // TODO: Employed Bees
    // TODO: Onlooker Bees

  }

}

void ArtificialBeeColony (Square * d_unsolved, Square * d_solved, int n) {

  /*
    int cycles;
    int num_employees;
    int num_onlookers;
    int num_scouts = 0.1 * num_employees;
  */

  // TODO: Employed Bee; neighborhood search population evaluation.
  // TODO: Onlooker Bee; Calculate probability values, search again & evaluate.
  // TODO: Scout Bee; determined abandoned solutions, replace them with new &
  // better solutions.

  int threadsPerBlock = 9;

  dim3 grid   (1, CHAINS, 1);
  dim3 block  (threadsPerBlock, threadsPerBlock, 1);

  curandState *d_state;
	ERROR_CHECK( cudaMalloc(&d_state, dimBlock.x * dimBlock.y * dimGrid.x * dimGrid.y));

	cuRandomNumberGenerator<<<grid.x * grid.y, block.x * block.y>>>(d_state);
  ERROR_CHECK( cudaPeekAtLastError() );
  ERROR_CHECK( cudaDeviceSynchronize() );

  BeeColony<<< grid, block >>>(d_unsolved, d_solved, n, d_state);
  ERROR_CHECK( cudaPeekAtLastError() );
  ERROR_CHECK( cudaDeviceSynchronize() );

  /*
    Gets copy of the Sudoku puzzle, which is the food source.
    Bee will randomly place digits from 1-9 on each open square.
    Evaluate the population and cycle until criteria is met:
      - Maximum number of cycles have reached; not able to find a viable
        solution in given number of cycles.
      - Fitness value of 1; Optimal solution was found and is ready to be
        displayed.
  */

  /*
    Neighborhood Search:
      - Given X(i) and its neighborhood X(k) generate random # j;
      - V(ij) = X(ij) + rand[0,1] * abs(X(ij) - X(kj));
      - V(ij) is not a locked square;
      - if V(ij) > 9 then mod value + 1 is used;
  */

  /*
    Fittness:
      - fit(i) = 1 / (1 + f(i));
      - Fit is the feasible solution of bee i
      - f(i) is penality value of each solution
      - Penality value is total number of missing values in each row and column

    Probability:
      - P(i) = fit(i) / [sum(fit(i)) 1 -> n];
  */

  /*
    Abandoning Solution:
      - Compare fitness of worst employed bee solution with scout bee solution.
  */

}

#endif
