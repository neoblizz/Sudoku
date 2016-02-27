// ----------------------------------------------------------------
// Sudoku -- Puzzle Solver on GPU using CUDA
// ----------------------------------------------------------------

/**
 * @file
 * io_utils.cu
 *
 * @brief Input & output handling routines
 */

/**
 * @brief Parses the commandline input and stores in CMD struct
 * @param[in] argc          Total number of arguments.
 * @param[in] argv          List of arguments.
 * @param[out] CMD          Struct that stores the commands
 */
void input(int argc, char** argv) {
  if (argc != 4) {
      usage(argv);
  }

  // Set size n (supports 9x9)
  int size = atoi(argv[1]);

  // Check output method.
  bool graphics = false;

  if (!strcmp(argv[2], "-gfx")) {
      graphics = true;
  } else if (!strcmp(argv[2], "-txt")) {
      graphics = false;
  } else {
    usage(argv);
  }

  // Set filename

}
