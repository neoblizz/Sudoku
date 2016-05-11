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
#include "beecolony.cuh"
#include "humanlogic.cuh"
#include "breadthfirstsearch.cuh"

void print (int n, Square * h_solved, const char * alg, float elapsedTime) {

  if (!strcmp(alg, "-bee")) {
    CSRtoCSC(h_solved, n);
    const char * finished = "/********** Bee Colony (C) **********/";
    output(finished, alg, n, false, h_solved);
  } else if (!strcmp(alg, "-log")) {
    const char * finished = "/********** Human-Logic(C) **********/";
    output(finished, alg, n, false, h_solved);
  } else if (!strcmp(alg, "-bfs")) {
    const char * finished = "/************ BFS-DFS (C) ***********/";
    output(finished, alg, n, false, h_solved);
  }

  const char* statistics = "/******* Statistics (Begin) ********/";
  printf("%s\n", statistics);
  printf("Elapsed Time: %f (ms)\n", elapsedTime);
  const char* statistics_end = "/******** Statistics (End) *********/";
  printf("%s\n", statistics_end);
}

void release (Square * h_unsolved, Square * d_unsolved, Square * d_solved) {
  /* Free Memory Allocations */
  free(h_unsolved);
  ERROR_CHECK( cudaFree(d_unsolved) );
  ERROR_CHECK( cudaFree(d_solved) );
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
  /* IMPORTANT: PLEASE ADD THIS IN YOUR KERNEL MANAGER FUNCTION */
  /*ERROR_CHECK( cudaMemcpy(d_unsolved, h_unsolved, memsize,
                          cudaMemcpyHostToDevice) );*/
 /* IMPORTANT: END! */


  Square * d_solved;
  ERROR_CHECK( cudaMalloc((void**) &d_solved, memsize) );

  /* ARTIFICIAL BEE COLONY ALGORITHM */
  float elapsedTime;
  cudaEventRecord(start, 0);
  ArtificialBeeColony (h_unsolved, d_unsolved, d_solved, n);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  Square * h_solved = (Square *) malloc(memsize);
  ERROR_CHECK( cudaMemcpy(h_solved, d_solved, memsize,
                          cudaMemcpyDeviceToHost) );

  const char * bee = "-bee";
  print(n, h_solved, bee, elapsedTime);

  /* HUMAN LOGIC BASED ALGORITHM */
  cudaEventRecord(start, 0);
  HumanLogic (h_unsolved, d_unsolved, d_solved, n);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  ERROR_CHECK( cudaMemcpy(h_solved, d_solved, memsize,
                          cudaMemcpyDeviceToHost) );

  const char * logic = "-log";
  print(n, h_solved, logic, elapsedTime);

  /* TREE BASED ALGORITHM */
  cudaEventRecord(start, 0);
  BreadthFirstSearch (h_unsolved, d_unsolved, d_solved, n);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  ERROR_CHECK( cudaMemcpy(h_solved, d_solved, memsize,
                          cudaMemcpyDeviceToHost) );

  const char * tree = "-bfs";
  print(n, h_solved, tree, elapsedTime);

  /* Destroy CUDA event */
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  /* Free Memory */
  release(h_unsolved, d_unsolved, d_solved);

}

int main(int argc, char** argv) {

    /* Gets arguments from command line and puzzle from a file */
    CommandLineArgs * build = new CommandLineArgs;
    input(argc, argv, build);
    KernelManager((*build).size, (*build).Puzzle, (*build).graphics);

}
