
//5120 MBytes (5368512512 bytes)
//11520 MBytes (12079136768 bytes)

#ifndef BFS_H
#define BFS_H

#pragma once


#include <string.h>

#include <iostream>
#include <fstream>


#include "util/error_utils.cuh"
#include "data.cuh"


using namespace std;

__device__ int solution_num;

__global__
void reset(){
	solution_num = 0;
}

__device__ bool valid(vPossible newNum, vPossible* sudoku, int pos){


	int row = pos / 9;

	int column = pos % 9;

	//bool ans = true;
	//check row
	for(int i = row; i < row+9; i ++){
		if(newNum == sudoku[i]){
			return false;
		}
	}


		//printf("newNum %d ?= %d\n", newNum, sudoku[i]);

	for(int j = column; j <= column+8*9; j += 9 ){
		if(newNum == sudoku[j]){
			return false;
		}
	}

	//check sub-sudoku

	int sub_row = row / 3;
	int sub_col = column / 3;

	int first_index = sub_row*3*9 + sub_col*3;

	for(int i = first_index; i <= first_index+2*9; i += 9){
		for(int j = i; j < first_index+3; j ++){
			if(newNum == sudoku[j]){
				return false;
			}
		}
	}


	return true;

}

__device__ void printPoss(int* poss){
	printf("print_possible\n");
	for(int i = 0; i < 9; i++){
		for(int j = 0; j < 9; j++){
			printf("%d", poss[9*i+j]);
		}

	}

}

//vPossible
//empty square --> -1

__global__ void bfs(int explore, vAnswer * newCombo, vAnswer * oldCombo,
							vPossible * d_possValues, int num_poss, int square){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid <= explore){

		int solution_index = 0;

		vAnswer *sudoku = &oldCombo[tid];

		for(int i = 0; i < num_poss; i++){

			//d_possValues
			vPossible vp = d_possValues[i];

			if(valid(vp, sudoku, square)){//pass in sudoku is the parent sudoku
				//printf("valid");
				solution_index = atomicAdd(&solution_num,1);
				//printf("soluiton_index: %d", solution_index);
				//copy parent sudoku onto old Combo array
				for(int j = 0; j < 81; j++){
					if(j != square){
						newCombo[solution_index*81 + j] = sudoku[j];
					}else{
						newCombo[solution_index*81 + j] = vp;
					}
				}
			}
		}

	}

}
#endif
