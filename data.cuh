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

struct Square {
  int value; 		//not needed if we always check len(possValues)
					//else set to 0 for all Squares and set when lock
					//so we could really just skip the lock if just check =0

	bool isLocked,		//may want to replace with int, so we can
					//guess and keep track of what is locked/guessed

	int possValues[9]	//for keeping track of what values this Square could be
					//based on other locked Squares in local row/col/block
};
