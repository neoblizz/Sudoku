// ----------------------------------------------------------------
// Sudoku -- Puzzle Solver on GPU using CUDA
// ----------------------------------------------------------------

/**
 * @file
 * io_utils.cu
 *
 * @brief Input & output handling routines
 */

#pragma once

#include <string.h>

#include <iostream>
#include <fstream>

namespace std;

/**
 * @brief Parses the commandline input and stores in CMD struct
 * @param[in] argc          Total number of arguments.
 * @param[in] argv          List of arguments.
 * @param[out] CMD          Struct that stores the commands
 */
void input(int argc, char** argv, CommandLineArgs * build) {
  if (argc != 4) {
      usage(argv);
  }

  // Set size n (supports 9x9)
  int n = atoi(argv[1]);
  (*build).size = n;

  // Check output method.
  bool graphics = false;

  if (!strcmp(argv[2], "-gfx")) {
      (*build).graphics = true;
  } else if (!strcmp(argv[2], "-txt")) {
      (*build).graphics = false;
  } else {
    usage(argv);
  }

  // Set filename
  char filename[1024];

  if (sizeof filename >= length) { strcpy (filename, argv[3]); }
  else { overFilename(argv); }

  const char * token = "x";
  ifstream is(filename);

  char problem;
  int i = 0;
  while(is.get(problem) && (i < n*n)) {

    if (strncmp(problem, token, sizeof(graph_token))) {
        (*build).unsolved[i].value = atoi(problem);
        (*build).unsolved[i].isLocked = -1;
    } else {
        (*build).unsolved[i].isLocked = 0;
    }

    i++;
  }

  is.close();
  return;

}
