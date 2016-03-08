// ----------------------------------------------------------------
// Sudoku -- Puzzle Solver on GPU using CUDA
// ----------------------------------------------------------------

/**
 * @file
 * sudoku.cu
 *
 * @brief main sudoku file to init and execute
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cuda.h>
#include <cuda_runtime.h>

// includes, kernels
#include "kernels.cuh"

// includes, utilities
#include "util/error_utils.cuh"
#include "util/io_utils.cuh"
#include "data.cuh"


int main(int argc, char** argv) {

    /* Gets arguments from command line and puzzle from a file */
    CommandLineArgs * build = new CommandLineArgs;
    input(argc, argv, build);
    KernelManager((*build).size, &(*build).unsolved, (*build).graphpics);

}

void KernelManager(int n, Square * h_unsolved, bool o_graphics) {

  /* CUDA event setup */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Memory Allocations */
  int memsize = sizeof(Square) * n * n;

  Square * d_unsolved;
  ERROR_CHECK( cudaMalloc((void**) &d_unsolved, memsize) );
  ERROR_CHECK( cudaMemcpy(d_unsolved, h_unsolved, memsize,
                          cudaMemcpyHostToDevice) );

  Square * d_solved;
  ERROR_CHECK( cudaMalloc((void**) &d_solved, memsize) );

  float elapsedTime;
  cudaEventRecord(start, 0);

  // TODO: Kernel execution
  // TODO: All of them can go one by one,
  // TODO: we'll just need to reset event record,
  // TODO: for multiple timing/performance measurements.

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  Square * h_solved = (Square *) malloc(memsize);
  ERROR_CHECK( cudaMemcpy(h_solved, d_solved, memsize,
                          cudaMemcpyDeviceToHost) );

  /* Destroy CUDA event */
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // TODO: Terminal Output will go here.

  /* Free Memory Allocations */
  free(h_unsolved);
  ERROR_CHECK( cudaFree(d_unsolved) );
  ERROR_CHECK( cudaFree(d_solved) );
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
