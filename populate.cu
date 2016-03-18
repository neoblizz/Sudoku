#ifndef POP_H
#define POP_H

#pragma once

#include "device_function.cu"

// Populates the possValues array
// Removes all values that are not possible, based on starting board



// returns an array containing all the Squares in the current row
__device__
void getRow(int tid, Square* board, Square* localRow) {

	int rowNum = (tid/9); // had floor here before
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
	int offset;

	for (int i=0; i<9; i++) {

		if (i<3)
			offset = i;
		else if (i<6)
			offset = i -3 +9;
		else
			offset = i -6 +18;

		localBlock[i].value = board[starter + offset].value;
		localBlock[i].isLocked = board[starter + offset].isLocked;

		for(int j=0; j<9; j++)
				localBlock[i].possValues[j] = board[starter+offset].possValues[j];
	}


/*	localBlock =
		{board[starter], board[starter+1], board[starter+2],
		board[starter+9], board[starter+10], board[starter+11],
		board[starter+18], board[starter+19], board[starter+20]};
*/	//return output;

}


__global__ void populate(Square* board) {

	__shared__ Square s_board[81];

	if (threadIdx.x == 0) {
		for(int i = 0; i<81; i++) {
			s_board[i].value = board[i].value;
			s_board[i].isLocked = board[i].isLocked;

			for (int j=0; j<9; j++) {
					s_board[i].possValues[j] = board[i].possValues[j];
			}
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

		//int cur;
		int rowVal, colVal, blockVal;
		for (int i=0; i<9; i++) {
			//cur = s_board[tid].possValues[i];
		
			rowVal = localRowValues[i];
			colVal = localColValues[i];
			blockVal = localBlockValues[i];	

			if (rowVal>=1 && rowVal<=9)
				s_board[tid].possValues[rowVal-1] = 0;
	
//			if (colVal != 0)
			if (colVal>=1 && colVal<=9)
				s_board[tid].possValues[colVal-1] = 0;

			if (blockVal>=1 && blockVal<=9)
				s_board[tid].possValues[blockVal-1] = 0;


/*			if (cur==NULL)
				break;

			// check if another Square in the row/col/block makes cur
				// invalid for the current Square
			if (!validNum(cur, localBlockValues, localRowValues,
					localColValues)) {

				//if there is a conflict, pop it off
				//isPossibleNum checks if it's even in the
				//array of possValues
				popoff(i, s_board[i].possValues);
*/
		

		}
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		for (int i=0; i<81; i++) {
			board[i].value = s_board[i].value;
			board[i].isLocked = s_board[i].isLocked;

			for (int j=0; j<9; j++)
				board[i].possValues[j] = s_board[i].possValues[j];

		}
	}

}
#endif
