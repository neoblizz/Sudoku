// ----------------------------------------------------------------
// Sudoku -- Puzzle Solver on GPU using CUDA
// ----------------------------------------------------------------

/**
 * @file
 * device_function.cu
 *
 * @brief File for all our device functions.
 */

#pragma once

#include "data.cuh"

__device__
bool validNum(vPossible newNum, int* box, int* row, int* col){

	for(int i =0; i < 9; i++){
		if(newNum == squre[i] || newNum == row[i] || newNum == col[i]){
			return false;
		}
	}
	return true;
}



//possibleNum array either has 10 numbers or terminate with null char
//return the index of target Num if exist
//return -1 if it doens't exist in the possible num array
__device__
int isPossibleNum(vPossible newNum, vPossible * possibleNum){
	for(int i = 0; i < 9; i++){
		if(possibleNum[i] != '\0'){
			if(possibleNum == newNum){
				return i;
			}
		}
	}
	return -1;
}

//pop off the given number in the possibleNum array
//fill the rest of the array to null('\0')
__device__
void popoff(vPossible num, vPossible *possibleNum){
	int pop = isPossibleNum(num, possibleNum);
	if(pop != -1){ //num is in the possible num array, remove it
		for(int i = pop; i < 9; i++){

			possibleNum[i] = possibleNum[i+1]; //shift possible num array left

			if(possibleNum[i] == '\0'){ //the end of possible num array
				break;
			}
		}
	}

}
