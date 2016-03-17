#ifndef IO_UTILS_H
#define IO_UTILS_H

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

#include "error_utils.cuh"
#include "../data.cuh"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_RESET   "\x1b[0m"

using namespace std;

/**
 * @brief Displays the puzzle at any state.
 * @param[in] description       Description of what you're displaying.
 * @param[in] n                 Size of the puzzle.
 * @param[in] graphics          GFX or TXT output?
 * @param[in] display           Actual puzzle to display, format:
 *          -------------------------------------
 *          | v | v | v | v | v | v | v | v | v |
 *          -------------------------------------
 *          | v | v | v | v | v | v | v | v | v |
 *          -------------------------------------
 *          | v | v | v | v | v | v | v | v | v |
 *          -------------------------------------
 *          | v | v | v | v | v | v | v | v | v |
 *          -------------------------------------
 *          | v | v | v | v | v | v | v | v | v |
 *          -------------------------------------
 *          | v | v | v | v | v | v | v | v | v |
 *          -------------------------------------
 *          | v | v | v | v | v | v | v | v | v |
 *          -------------------------------------
 *          | v | v | v | v | v | v | v | v | v |
 *          -------------------------------------
 *          | v | v | v | v | v | v | v | v | v |
 *          -------------------------------------
 */
void output(const char * description, const char * algorithm,
            int n, bool graphics, Square * display) {

  cout << description << endl;
  cout << "-------------------------------------" << endl;
  cout << "| ";

  int k = 0;
  while (k < n*n) {

    if (display[k].isLocked == -1) {
        printf(ANSI_COLOR_RED "%d" ANSI_COLOR_RESET, display[k].value);
    } else if (display[k].isLocked == 0 && strcmp(algorithm, "-bee")) {
        cout << "x";
    } else if (display[k].isLocked == 0 && !strcmp(algorithm, "-bee")) {
        cout << (int) display[k].value;
    }
    cout << " | ";

    if (k == 8 || k == 17 || k == 26 || k == 35 ||
        k == 44 || k == 53 || k == 62 || k == 71 || k == 80) {
          cout << "" << endl;
          cout << "-------------------------------------" << endl;
          if (k != 80) { cout << "| "; }
    }
    k++;

  }

}

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
  // cout << (*build).size << endl;

  // Check output method.
  if (!strcmp(argv[2], "-gfx")) {
      (*build).graphics = true;
  } else if (!strcmp(argv[2], "-txt")) {
      (*build).graphics = false;
  } else {
    usage(argv);
  }

  // cout << (*build).graphics << endl;

  // Set filename
  char filename[1024];

  if (sizeof(filename) >= 1024) { strcpy (filename, argv[3]); }
  else { overFilename(argv); }

  //cout << filename << endl;

  char token = 'x';
  ifstream is(filename);

  char problem;
  int i = 0;
  while(is.get(problem) && (i < n*n)) {

    if (strncmp(&problem, &token, sizeof(token))) {
        (*build).Puzzle[i].value = problem - 48; // atoi(&problem);
        // cout << (*build).Puzzle[i].value;
        (*build).Puzzle[i].isLocked = -1;
    } else {
        for (int j = 0; j < 9; j++) {
            (*build).Puzzle[i].possValues[j] = j+1;
        }
        (*build).Puzzle[i].value = 0;
        (*build).Puzzle[i].isLocked = 0;
    }

    i++;
  }

  // cout << i << endl;

  is.close();
  const char * unsolved = "/********** Input Puzzle **********/";
  output(unsolved, "-in", (*build).size, (*build).graphics, (*build).Puzzle);
  return;

}

#endif
