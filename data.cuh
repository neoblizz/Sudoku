// ----------------------------------------------------------------
// Sudoku -- Puzzle Solver on GPU using CUDA
// ----------------------------------------------------------------

/**
 * @file
 * data.cuh
 *
 * @brief Stores data types to be used.
 */

struct CMD {
  Size size;
  bool gfx_output;
  Puzzle * unsolved;
}; /* Stores command line arguments */
