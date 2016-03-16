#ifndef DATA_H
#define DATA_H

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
 * Guessed = 1-9;
 */
typedef int Lock;
typedef int vAnswer;
typedef int vPossible;

struct Square {
  int value;
	int isLocked;
	int possValues[PUZZLE_SIZE];
}; /* Stores per square value for a puzzle */

struct CommandLineArgs {
  int size;
  bool graphics;
  Square * Puzzle = new Square[PUZZLE_SIZE*PUZZLE_SIZE];
}; /* Stores command line arguments */

#endif
