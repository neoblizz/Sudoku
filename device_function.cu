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


//checks newNum against the other values in the respective row/col/block
//row/col/block must be loaded before calling this function
	//into 3 separate arrays, and the int*s here are pointers to those arrays
//returns true if newNum is valid
//returns false if there is a conflict
__device__
bool validNum(vPossible newNum, int* block, int* row, int* col){

	for(int i =0; i < 9; i++){
		if(newNum == block[i] || newNum == row[i] || newNum == col[i]){
			return false;
		}
	}
	return true;
}



//checks to see if a number is in the array of possible values
//possibleNum array either has 9 numbers or terminate with null char
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


			//first, check if no more possible values (even though not 9 elements)
			if(possibleNum[i] == '\0'){ //the end of possible num array
				break;
		
			}

			//if array was previously 9 elements, the last element should now be \0
			if(i=8) {
				possibleNum[i] = '\0';
				break;
			}

			//shift left
			possibleNum[i] = possibleNum[i+1]; //shift possible num array left

		}
	}

}
