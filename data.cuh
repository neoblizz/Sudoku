// ----------------------------------------------------------------
// Sudoku -- Puzzle Solver on GPU using CUDA
// ----------------------------------------------------------------

/**
 * @file
 * data.cuh
 *
 * @brief Stores data types to be used.
 */
#define PUZZLE_SIZE 9

/*
 * Locked = -1;
 * Open = 0;
 * Guessed = 1;
 */
typedef int Lock;
typedef int vAnswer;
typedef int vPossible;

struct Square {
  vAnswer value;
	Lock isLocked;
	vPossible possValues[PUZZLE_SIZE];
} Puzzle; /* Stores per square value for a puzzle */

struct CommandLineArgs {
  Size size;
  bool graphics;
  Puzzle unsolved[PUZZLE_SIZE*PUZZLE_SIZE];
}; /* Stores command line arguments */
