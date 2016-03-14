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

#define CHAINS 4
#define CYCLES (1<<10)
#define INIT_EFFORT 0.4
#define EFFORT_MIN 0.001
#define INIT_TOLERANCE 1
#define DELTA_TOLERANCE 0.2

__constant__ int MASK[81];

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
__device__ int foodQuality (int food, Square sudoku[][9], bool rc) {

  int index, quality;
  int nums[9]={1,2,3,4,5,6,7,8,9};

      quality = 0;

      for(int i = 0; i < 9; i++) {
          if(rc == 1) {
            index = sudoku[i][food].value - 1;
          }
          else {
            index = sudoku[food][i].value - 1;
          }

          if(index == -1) { return -1; }

          if(nums[index] != 0) {
              quality += 1;
              nums[index] = 0;
          }
      }

      return quality;

}

/**
 * @brief Used to determines the quality of the solution by calculating
 * uniqueness of an elements.
 */
__device__ int hiveQuality (Square sudoku[][9]) {

  	int q_hive = 0;

  	for(int i = 0; i < 9; i++) {
      q_hive += foodQuality(i, sudoku, 0) + foodQuality(i, sudoku, 1);
    }

  	return 162 - q_hive;
}

__global__ void BeeColony (Square * d_unsolved, Square * d_solved,
                           int n, curandState * state, int c_hive,
                           int * qualityHives, float effort,
                           Square * d_solved1, Square * d_solved2,
                           Square * d_solved3, Square * d_solved4) {

  int index = threadIdx.y + blockIdx.x * blockDim.x;
  int block = blockIdx.y + blockIdx.x * blockDim.x;
  int index_x = threadIdx.x;
  int index_y = threadIdx.y;

  // TODO: Copy d_unsolved to shared memory;
  __shared__ Square sudoku[9][9];

  sudoku[index_x][index_y].value = d_unsolved[index_x + 9*index_y].value;
  sudoku[index_x][index_y].isLocked = d_unsolved[index_x + 9*index_y].isLocked;

  if (index != 0) {
    return;
  }

  int sub_blockx, sub_blocky;
  int unmask1_x, unmask1_y, unmask2_x, unmask2_y;
  int q_hive;

  for(int i = 0; i < CYCLES; i++) {
    // USAGE: number = (int) curand_uniform(&state[index]);
    // TODO: Scout to begin the process.
    // TODO: Randomly select a block.
    // TODO: Employed Bees
    // TODO: Onlooker Bees

		sub_blockx = 3 * (int) (3.0 * curand_uniform(&state[block]));
		sub_blocky = 3 * (int) (3.0 * curand_uniform(&state[block]));

    do {
			unmask1_x = (int) 3.0 * curand_uniform(&state[block]);
			unmask1_y = (int) 3.0 * curand_uniform(&state[block]);

		} while(MASK[(sub_blockx + unmask1_x) + 9 * (sub_blocky + unmask1_y)] == 1);

    do {
			unmask2_x = (int) 3.0 * curand_uniform(&state[block]);
			unmask2_y = (int) 3.0 * curand_uniform(&state[block]);

		} while(MASK[(sub_blockx + unmask2_x) + 9 * (sub_blocky + unmask2_y)] == 1);

    int swap;
    swap = sudoku[sub_blockx+unmask1_x][sub_blocky+unmask1_y].value;
    sudoku[sub_blockx+unmask1_x][sub_blocky+unmask1_y].value = sudoku[sub_blockx+unmask2_x][sub_blocky+unmask2_y].value;
    sudoku[sub_blockx+unmask2_x][sub_blocky+unmask2_y].value = swap;

    q_hive = hiveQuality(sudoku);

    if(q_hive < c_hive) {
      c_hive = q_hive;
    } else {

      if(exp((float) (c_hive - q_hive)/effort) > curand_uniform(&state[block])) {
        c_hive = q_hive;
      } else {
        swap = sudoku[sub_blockx + unmask1_x][sub_blocky + unmask1_y].value;
        sudoku[sub_blockx + unmask1_x][sub_blocky + unmask1_y].value = sudoku[sub_blockx + unmask2_x][sub_blocky + unmask2_y].value;
        sudoku[sub_blockx + unmask2_x][sub_blocky + unmask2_y].value = swap;
      }
    }

    if (q_hive == 0) { break; }
  }

  for(int m = 0; m < 9; m++) {
		for(int n = 0; n < 9; n++) {
      switch (block) {
        case 0:
          d_solved1[m + 9 * n] = sudoku[m][n];
        case 1:
          d_solved2[m + 9 * n] = sudoku[m][n];
        case 2:
          d_solved3[m + 9 * n] = sudoku[m][n];
        case 3:
          d_solved4[m + 9 * n] = sudoku[m][n];
        default:
          break;
      }
    }
  }

  qualityHives[block] = c_hive;

}

void ArtificialBeeColony (Square * d_unsolved, Square * d_solved, int n) {

  // TODO: Employed Bee; neighborhood search population evaluation.
  // TODO: Onlooker Bee; Calculate probability values, search again & evaluate.
  // TODO: Scout Bee; determined abandoned solutions, replace them with new &
  // better solutions.

  int threadsPerBlock = n;

  dim3 grid   (1, CHAINS, 1);
  dim3 block  (threadsPerBlock, threadsPerBlock, 1);

  /* Memory Allocations */
  float effort = INIT_EFFORT;
  int memsize = sizeof(Square) * n * n;
  Square * d_solved1;
  Square * d_solved2;
  Square * d_solved3;
  Square * d_solved4;
  int * qualityHives;
  int * h_qualityHives;

  ERROR_CHECK( cudaHostAlloc((void**) &h_qualityHives, sizeof(int)*CYCLES,
                                      cudaHostAllocDefault));

  ERROR_CHECK( cudaMalloc((void**) &d_solved1, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved2, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved3, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved4, memsize));
  ERROR_CHECK( cudaMalloc((void**) &MASK, memsize));

  ERROR_CHECK( cudaMalloc((void**) &qualityHives, sizeof(int)*CYCLES));

  curandState *d_state;
	ERROR_CHECK( cudaMalloc((void**) &d_state, block.x * block.y *
                                   grid.x * grid.y));

	cuRandomNumberGenerator<<<grid.x * grid.y, block.x * block.y>>>(d_state);
  ERROR_CHECK( cudaPeekAtLastError() );
  ERROR_CHECK( cudaDeviceSynchronize() );


  int tolerance = INIT_TOLERANCE;
  int minimum, min_index;
  int c_hive, quality;

  int p_hive = c_hive;

  do {

    minimum = 200;
    min_index = 200;

    BeeColony<<< grid, block >>>(d_unsolved, d_solved, n, d_state, c_hive,
                               qualityHives, effort, d_solved1, d_solved2, d_solved3,
                               d_solved4);

    ERROR_CHECK( cudaPeekAtLastError() );
    ERROR_CHECK( cudaDeviceSynchronize() );

    ERROR_CHECK( cudaMemcpy(h_qualityHives, qualityHives, sizeof(int) * CHAINS,
                            cudaMemcpyDeviceToHost));

    for(quality = 0; quality < CHAINS; quality++) {
      if (h_qualityHives[quality] < minimum) {
        minimum = h_qualityHives[quality];
        min_index = quality;
      }
    }

    switch (min_index) {
      case 0:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved1, memsize,
                                cudaMemcpyDeviceToDevice));
        c_hive = minimum;
      case 1:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved2, memsize,
                                cudaMemcpyDeviceToDevice));
        c_hive = minimum;
      case 2:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved3, memsize,
                                cudaMemcpyDeviceToDevice));
        c_hive = minimum;
      case 3:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved4, memsize,
                                cudaMemcpyDeviceToDevice));
        c_hive = minimum;
      default:
        break;
    }

    if (c_hive == 0) { break; }
    if (c_hive == p_hive) { tolerance--; }
    else { tolerance = INIT_TOLERANCE; }

    if (tolerance < 0) { }

    p_hive = c_hive;
    if (c_hive == 0) { break; }
    effort = effort * 0.8;

  } while (effort > EFFORT_MIN);

  ERROR_CHECK( cudaMemcpy(d_solved, d_unsolved, memsize, cudaMemcpyDeviceToHost));

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
