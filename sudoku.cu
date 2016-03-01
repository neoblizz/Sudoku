// ----------------------------------------------------------------
// Sudoku -- Puzzle Solver on GPU using CUDA
// ----------------------------------------------------------------

/**
 * @file
 * sudoku.cu
 *
 * @brief main sudoku file to init and execute
 */

#pragma once

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

void KernelManager(Size n, Puzzle * h_unsolved, bool o_graphics) {

  /* CUDA event setup */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Memory Allocations */
  memsize = sizeof(Puzzle) * (*build).size * (*build).size;

  Puzzle * d_unsolved;
  ERROR_CHECK( cudaMalloc((void**) &d_unsolved, memsize) );
  ERROR_CHECK( cudaMemcpy(d_unsolved, h_unsolved, memsize,
                          cudaMemcpyHostToDevice) );

  float elapsedTime;
  cudaEventRecord(start, 0);

  // TODO: Kernel execution
  // TODO: All of them can go one by one,
  // TODO: we'll just need to reset event record,
  // TODO: for multiple timing/performance measurements.

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  Puzzle * d_solved;
  ERROR_CHECK( cudaMalloc((void**) &d_solved, memsize) );

  /* Destroy CUDA event */
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // TODO: Terminal Output will go here.

  /* Free Memory Allocations */
  free(h_unsolved);
  ERROR_CHECK( cudaFree(d_unsolved) );
  ERROR_CHECK( cudaFree(d_solved) );
}
