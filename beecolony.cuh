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
#define CYCLES (1<<20) // 1000000
#define INIT_EFFORT 0.4
#define EFFORT_MIN 0.001
#define INIT_TOLERANCE 1
#define DELTA_TOLERANCE 0.2

using namespace std;

/**
 * @file
 * beecolony.cuh
 *
 * @brief ArtificialBeeColony algorithm implementation.
 */

/**
 * @brief Converts a 1D array to 2D array.
 *
 */
void ONEDtoTWOD (Square h_sudoku[][9], Square * h_unsolved, int n) {
 for (int x = 0; x < n; x++) {
   for (int y = 0; y < n; y++) {
       h_sudoku[x][y].value = h_unsolved[x + n * y].value;
       h_sudoku[x][y].isLocked = h_unsolved[x + n * y].isLocked;
   }
 }
}

/**
 * @brief Converts row major order to column major.
 *
 */
__host__ __device__ void CSRtoCSC (Square * sudoku, int n) {

    Square Swap[81];

    int limit = 0;
    for (int i = 0; i < 9; i++) {
     for (int j = 0; j < 9; j++) {
       Swap[i + n * j].value = sudoku[limit].value;
       Swap[i + n * j].isLocked = sudoku[limit].isLocked;
       limit++;
     }
    }

    for (int k = 0; k < 81; k++) {
       sudoku[k].value = Swap[k].value;
       sudoku[k].isLocked = Swap[k].isLocked;
    }

}

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
__host__ __device__ int foodQuality (int food, Square sudoku[][9], bool rc) {

  int index, quality;
  int nums[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

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
__host__ __device__ int hiveQuality (Square sudoku[][9]) {

  	int q_hive = 0;

  	for(int i = 0; i < 9; i++) {
      q_hive += foodQuality(i, sudoku, 0) + foodQuality(i, sudoku, 1);
    }

  	return 162 - q_hive;
}


__global__ void toCSC (Square * sudoku, int n) {
    CSRtoCSC (sudoku, n);
}

__global__ void BeeColony (Square * d_unsolved, Square * d_solved,
                           int n, curandState * state, int c_hive,
                           int * qualityHives, float effort, // Square * d_sub) {
                           Square * d_solved1, Square * d_solved2,
                           Square * d_solved3, Square * d_solved4,
                           Square * d_solved5, Square * d_solved6,
                           Square * d_solved7, Square * d_solved8,
                           Square * d_solved9, Square * d_solved10,
                           Square * d_solved11, Square * d_solved12,
                           Square * d_solved13, Square * d_solved14,
                           Square * d_solved15) {

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

    /* SCOUT BEE */
		sub_blockx = 3 * (int) (3.0 * curand_uniform(&state[block]));
		sub_blocky = 3 * (int) (3.0 * curand_uniform(&state[block]));

    /* WORKER BEE */
    do {
			unmask1_x = (int) 3.0 * curand_uniform(&state[block]);
			unmask1_y = (int) 3.0 * curand_uniform(&state[block]);

		} while(sudoku[sub_blockx + unmask1_x][sub_blocky + unmask1_y].isLocked == -1);

    do {
			unmask2_x = (int) 3.0 * curand_uniform(&state[block]);
			unmask2_y = (int) 3.0 * curand_uniform(&state[block]);

		} while(sudoku[sub_blockx + unmask2_x][sub_blocky + unmask2_y].isLocked == -1);

    int swap;
    swap = sudoku[sub_blockx+unmask1_x][sub_blocky+unmask1_y].value;
    sudoku[sub_blockx+unmask1_x][sub_blocky+unmask1_y].value = sudoku[sub_blockx+unmask2_x][sub_blocky+unmask2_y].value;
    sudoku[sub_blockx+unmask2_x][sub_blocky+unmask2_y].value = swap;

    /* ONLOOKER BEE */
    q_hive = hiveQuality(sudoku);

    if(q_hive < c_hive) {
      c_hive = q_hive;
    } else {

      if(exp((float) (c_hive - q_hive)/effort) > curand_uniform(&state[block])) {
        c_hive = q_hive;
      } else {
        swap = sudoku[sub_blockx + unmask1_x][sub_blocky + unmask1_y].value;
        sudoku[sub_blockx + unmask1_x][sub_blocky + unmask1_y].value = \
        sudoku[sub_blockx + unmask2_x][sub_blocky + unmask2_y].value;
        sudoku[sub_blockx + unmask2_x][sub_blocky + unmask2_y].value = swap;
      }
    }

    if (q_hive == 0) { break; }
  }

  for(int m = 0; m < 9; m++) {
		for(int y = 0; y < 9; y++) {
      // d_sub[(m + y) + (block * (n * n))].value = sudoku[m][y].value;
      // d_sub[(m + y) + (block * (n * n))].isLocked = sudoku[m][y].isLocked;

      switch (block) {
        case 0:
          d_solved1[m + 9 * y] = sudoku[m][y];
        case 1:
          d_solved2[m + 9 * y] = sudoku[m][y];
        case 2:
          d_solved3[m + 9 * y] = sudoku[m][y];
        case 3:
          d_solved4[m + 9 * y] = sudoku[m][y];
        case 4:
          d_solved5[m + 9 * y] = sudoku[m][y];
        case 5:
          d_solved6[m + 9 * y] = sudoku[m][y];
        case 6:
          d_solved7[m + 9 * y] = sudoku[m][y];
        case 7:
          d_solved8[m + 9 * y] = sudoku[m][y];
        case 8:
          d_solved9[m + 9 * y] = sudoku[m][y];
        case 9:
          d_solved10[m + 9 * y] = sudoku[m][y];
        case 10:
          d_solved11[m + 9 * y] = sudoku[m][y];
        case 11:
          d_solved12[m + 9 * y] = sudoku[m][y];
        case 12:
          d_solved13[m + 9 * y] = sudoku[m][y];
        case 13:
          d_solved14[m + 9 * y] = sudoku[m][y];
        case 14:
          d_solved15[m + 9 * y] = sudoku[m][y];
        default:
          break;
      }
    }
  }

  qualityHives[block] = c_hive;

}

/**
 * @brief Used to restart the randomness if stuck in local minimum.
 *
 */
void restart (Square * h_unsolved, Square * d_unsolved, int memsize) {

  ERROR_CHECK( cudaMemcpy(h_unsolved, d_unsolved, memsize,
                          cudaMemcpyDeviceToHost));

  int ar[3]={0,3,6};
  int swap;
  int rand1 = random()%3;
  int rand2 = random()%3;

  int r1_x,r1_y,r2_x,r2_y;
  int block_x,block_y;

  for (int suf = 0; suf < random()%10; suf++) {

    block_x = ar[rand1];
    block_y = ar[rand2];
    do {
      r1_x=random()%3;
      r1_y=random()%3;
    } while (h_unsolved[(block_x + r1_x) + 9 * (block_y + r1_y)].isLocked == -1);

    do {
      r2_x=random()%3;
      r2_y=random()%3;
    } while (h_unsolved[(block_x + r2_x) + 9 * (block_y + r2_y)].isLocked == -1);

    swap = h_unsolved[(block_x + r1_x) + 9 * (block_y + r1_y)].value;
    h_unsolved[(block_x + r1_x) + 9 * (block_y + r1_y)].value = \
    h_unsolved[(block_x + r2_x) + 9 * (block_y + r2_y)].value;
    h_unsolved[(block_x + r2_x) + 9 * (block_y + r2_y)].value = swap;
  }

  ERROR_CHECK( cudaMemcpy(d_unsolved, h_unsolved, memsize,
                          cudaMemcpyHostToDevice));

}

/**
 * @brief Initializes the ABC algorithm, and also satisfies 3x3 clause.
 *
 */
void init_ArtificalBeeColony (Square * h_unsolved, int n) {

  int nums_filed[9];
  int nums_sol[9];

  int x, y, p, q, index;

  for(int block_i = 0; block_i < 3; block_i++) {
    for(int block_j = 0; block_j < 3; block_j++) {
        for(int k = 0; k < 9; k++) { nums_filed[k] = k + 1; }
        for(int i = 0; i < 3; i++) {
          for(int j = 0; j < 3; j++) {
            x = block_i * 3 + i;
            y = block_j * 3 + j;

            if(h_unsolved[x + 9 * y].isLocked != 0){
              p = h_unsolved[x + 9 * y].value;
              nums_filed[p - 1] = 0;
            }
          }
        }
        q = -1;
        for(int k = 0; k < 9; k++) {
          if(nums_filed[k] != 0) {
            q += 1;
            nums_sol[q] = nums_filed[k];
          }
        }
        index = 0;
        for(int i = 0; i < 3; i++) {
          for(int j = 0; j < 3; j++) {
            x = block_i * 3 + i;
            y = block_j * 3 + j;

            if(h_unsolved[x + 9 * y].isLocked == 0) {
              h_unsolved[x + 9 * y].value = nums_sol[index];
              index += 1;
            }
          }
        }

      }
    }
}


/**
 * @brief Main ABC algorithm controller.
 *
 */
void ArtificialBeeColony (Square * h_unsolved, Square * d_unsolved,
                          Square * d_solved, int n) {

  // TODO: Employed Bee; neighborhood search population evaluation.
  // TODO: Onlooker Bee; Calculate probability values, search again & evaluate.
  // TODO: Scout Bee; determined abandoned solutions, replace them with new &
  // better solutions.

  /* Memory Allocations */
  float effort = INIT_EFFORT;
  int memsize = sizeof(Square) * n * n;
  int * qualityHives;
  int * h_qualityHives;

  ERROR_CHECK( cudaHostAlloc((void**) &h_qualityHives, sizeof(int)*CYCLES,
                                      cudaHostAllocDefault));

  Square * d_solved1;
  Square * d_solved2;
  Square * d_solved3;
  Square * d_solved4;
  Square * d_solved5;
  Square * d_solved6;
  Square * d_solved7;
  Square * d_solved8;
  Square * d_solved9;
  Square * d_solved10;
  Square * d_solved11;
  Square * d_solved12;
  Square * d_solved13;
  Square * d_solved14;
  Square * d_solved15;

  ERROR_CHECK( cudaMalloc((void**) &d_solved1, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved2, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved3, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved4, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved5, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved6, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved7, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved8, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved9, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved10, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved11, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved12, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved13, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved14, memsize));
  ERROR_CHECK( cudaMalloc((void**) &d_solved15, memsize));

  // Square * d_sub;
  // ERROR_CHECK( cudaMalloc((void**) &d_sub, memsize * CHAINS));

  ERROR_CHECK( cudaMalloc((void**) &qualityHives, sizeof(int)*CYCLES));

  int threadsPerBlock = n;
  dim3 grid   (1, CHAINS, 1);
  dim3 block  (threadsPerBlock, threadsPerBlock, 1);

  curandState *d_state;
	ERROR_CHECK( cudaMalloc((void**) &d_state, block.x * block.y *
                                   grid.x * grid.y));

  cuRandomNumberGenerator<<<grid.x * grid.y, block.x * block.y>>>(d_state);
  ERROR_CHECK( cudaPeekAtLastError() );
  ERROR_CHECK( cudaDeviceSynchronize() );


  int tolerance = INIT_TOLERANCE;
  int minimum, min_index;

  Square h_sudoku[9][9];
  CSRtoCSC(h_unsolved, n);
  init_ArtificalBeeColony(h_unsolved, n);

  // const char * inprogress = "/********** Bee Colony (CSC) **********/";
  // output(inprogress, "-bee", n, false, h_unsolved);

  ONEDtoTWOD(h_sudoku, h_unsolved, n);

  int c_hive = hiveQuality(h_sudoku);
  int p_hive = c_hive;

  ERROR_CHECK( cudaMemcpy(d_unsolved, h_unsolved, memsize,
                          cudaMemcpyHostToDevice) );

  do {

    minimum = 200;
    min_index = 200;

    BeeColony<<< grid, block >>>(d_unsolved, d_solved, n, d_state, c_hive,
                               qualityHives, effort, // d_sub);
                               d_solved1, d_solved2, d_solved3,
                               d_solved4, d_solved5, d_solved6,
                               d_solved7, d_solved8, d_solved9,
                               d_solved10, d_solved11, d_solved12,
                               d_solved13, d_solved14, d_solved15);

    ERROR_CHECK( cudaPeekAtLastError() );
    ERROR_CHECK( cudaDeviceSynchronize() );

    ERROR_CHECK( cudaMemcpy(h_qualityHives, qualityHives, sizeof(int) * CHAINS,
                            cudaMemcpyDeviceToHost));

    for(int quality = 0; quality < CHAINS; quality++) {
      if (h_qualityHives[quality] < minimum) {
        minimum = h_qualityHives[quality];
        min_index = quality;
      }
    }

    c_hive = minimum;

    /* // TODO: Condense all the solution numbers to one giant array with ptrs.
    ERROR_CHECK( cudaMemcpy(d_unsolved, d_sub+(min_index*n*n), memsize,
                            cudaMemcpyDeviceToDevice));

    toCSC<<<1,1>>>(d_unsolved, n);
    ERROR_CHECK( cudaPeekAtLastError() );
    ERROR_CHECK( cudaDeviceSynchronize() );
    */


    switch (min_index) {
      case 0:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved1, memsize,
                                cudaMemcpyDeviceToDevice));
      case 1:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved2, memsize,
                                cudaMemcpyDeviceToDevice));
      case 2:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved3, memsize,
                                cudaMemcpyDeviceToDevice));
      case 3:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved4, memsize,
                                cudaMemcpyDeviceToDevice));
      case 4:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved5, memsize,
                                cudaMemcpyDeviceToDevice));
      case 5:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved6, memsize,
                                cudaMemcpyDeviceToDevice));
      case 6:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved7, memsize,
                                cudaMemcpyDeviceToDevice));
      case 7:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved8, memsize,
                                cudaMemcpyDeviceToDevice));
      case 8:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved9, memsize,
                                cudaMemcpyDeviceToDevice));
      case 9:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved10, memsize,
                                cudaMemcpyDeviceToDevice));
      case 10:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved11, memsize,
                                cudaMemcpyDeviceToDevice));
      case 11:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved12, memsize,
                                cudaMemcpyDeviceToDevice));
      case 12:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved13, memsize,
                                cudaMemcpyDeviceToDevice));
      case 13:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved14, memsize,
                                cudaMemcpyDeviceToDevice));
      case 14:
        ERROR_CHECK( cudaMemcpy(d_unsolved, d_solved15, memsize,
                                cudaMemcpyDeviceToDevice));
      default:
        break;
    }


    if (c_hive == 0) { break; }
    if (c_hive == p_hive) { tolerance--; }
    else { tolerance = INIT_TOLERANCE; }

    if (tolerance < 0) {
      // printf("WARNING: Reached tolerance level. \n");
      // printf("WARNING: Restarting randomness of ABC. \n");

      restart(h_unsolved, d_unsolved, memsize);
      ONEDtoTWOD(h_sudoku, h_unsolved, n);
      c_hive = hiveQuality(h_sudoku);

      tolerance = INIT_TOLERANCE;
      effort = effort + DELTA_TOLERANCE;

    }

    p_hive = c_hive;
    if (c_hive == 0) { break; }
    effort = effort * 0.8;

  } while (effort > EFFORT_MIN);

  ERROR_CHECK( cudaMemcpy(d_solved, d_unsolved, memsize, cudaMemcpyDeviceToDevice));

  return;

}

#endif
