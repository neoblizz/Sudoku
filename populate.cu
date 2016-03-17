#ifndef POP_H
#define POP_H

#pragma once

#include "device_function.cu"

// Populates the possValues array
// Removes all values that are not possible, based on starting board



// returns an array containing all the Squares in the current row
__device__
void getRow(int tid, Square* board, Square* localRow) {

	int rowNum = floor(tid/9);
	int startOfRow = rowNum*9;
	//Square output[9];

	for (int i = 0; i<9; i++) {
		localRow[i].value = board[startOfRow + i].value;
		localRow[i].isLocked = board[startOfRow + i].isLocked;
	
		for (int j=0; j<9; j++) {
			localRow[i].possValues[j] = board[startOfRow + i].possValues[j];	
		}
	}

	//return output;

}

__device__
void getCol(int tid, Square* board, Square* localCol) {

	int colNum = tid%9; // also first element of the column
	//Square output[9];
	
	for (int i = 0; i<9; i++) {
		localCol[i].value = board[colNum + (9*i)].value;
		localCol[i].isLocked = board[colNum + (9*i)].isLocked;
	
		for (int j=0; j<9; j++) {
			localCol[i].possValues[j] = board[colNum + (9*i)].possValues[j];	
		}

	}

	//return output;

}

__device__
void getBlock(int tid, Square* board, Square* localBlock) {

	int blockRow; // tells us if it's in the top/mid/bot
	
	if (tid<27)
		blockRow = 0; //top

	else if (tid<54)
		blockRow = 1; //middle
	
	else
		blockRow = 2; //bottom


	int blockCol; // tells us if it's on the left/mid/right
	int col = tid%9;

	if (col<3)
		blockCol = 0; //left side

	else if (col<6)
		blockCol = 1; //middle

	else
		blockCol = 2; //right side


	//now we know exactly which block we are dealing with, sooooo

	int starter = blockRow*27 + blockCol*3;

	localBlock[9] =
		(board[starter], board[starter+1], board[starter+2],
		board[starter+9], board[starter+10], board[starter+11],
		board[starter+18], board[starter+19], board[starter+20]);	

	//return output;

}


__global__ void populate(Square* board) {

	__shared__ Square s_board[81];

	if (threadIdx.x == 0) {
		for(int i = 0; i<81; i++) {
			s_board[i] = board[i];
		}
	}
		

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid<81) { //one thread per square

		// start by filling the possValues array with numbers 1-9
		// board[tid].possValues = (1, 2, 3, 4, 5, 6, 7, 8, 9);
		// this is now done in io_utils.cuh

		// initialize arrays for the current square's row/col/block

		Square localRow[9];
		getRow(tid, s_board, localRow);
		Square localCol[9];
		getCol(tid, s_board, localCol);
		Square localBlock[9];
		getBlock(tid, s_board, localBlock);

		int localRowValues[9];
		int localColValues[9];
		int localBlockValues[9];

		for (int i=0; i<9; i++) {
			localRowValues[i] = localRow[i].value;
			localColValues[i] = localCol[i].value;
			localBlockValues[i] = localBlock[i].value;
		}


		// use popoff to remove invalid values from the possValues array

		int cur;
		for (int i=0; i<9; i++) {
			cur = s_board[tid].possValues[i];
			
			if (cur==NULL)
				break;

			// check if another Square in the row/col/block makes cur
				// invalid for the current Square
			if (!validNum(cur, localBlockValues, localRowValues,
					localColValues)) {

				//if there is a conflict, pop it off
				//isPossibleNum checks if it's even in the
				//array of possValues
				popoff(i, s_board[i].possValues);

			}

		}
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		for (int i = 0; i<81; i++)
			board[i] = s_board[i];
	}

}
#endif
