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

// includes, utilities
#include "util/error_utils.cuh"
#include "util/io_utils.cuh"
#include "data.cuh"

// includes, kernels
// #include "kernels.cuh"
#include "beecolony.cuh"

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

  ArtificialBeeColony (d_unsolved, d_solved, n);

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
  const char * finished = "/********** Output Puzzle **********/";
  output(finished, n, false, h_solved);

  /* Free Memory Allocations */
  free(h_unsolved);
  ERROR_CHECK( cudaFree(d_unsolved) );
  ERROR_CHECK( cudaFree(d_solved) );
}

int main(int argc, char** argv) {

    /* Gets arguments from command line and puzzle from a file */
    CommandLineArgs * build = new CommandLineArgs;
    input(argc, argv, build);
    KernelManager((*build).size, (*build).Puzzle, (*build).graphics);

}
