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


int main(int argc, char** argv) {

    input(argc, argv);

}
