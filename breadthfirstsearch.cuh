#ifndef BFSK_H
#define BFSK_H


#include <math.h>

#include "util/error_utils.cuh"
#include "data.cuh"

#include "populate.cu"
#include "bfs.cu"
//#include "human.cu"

using namespace std;

void BreadthFirstSearch (Square * h_unsolved, Square * d_unsolved,
												 Square * d_solved, int n) {

	// TODO: memcpy
	// TODO: set grid/TPB
	// TODO: call populate kernel (located in populate.cu)
	// TODO: after populate works, test human.cu
	// TODO: after every human call, need to populate again


	int memsize = sizeof(Square) * n * n;

	//ERROR_CHECK( cudaMemcpy(d_unsolved, h_unsolved, memsize, cudaMemcpyHostToDevice) );

	int threadsPerBlock = n;
	int blocksPerGrid = (n + threadsPerBlock -1) / threadsPerBlock;

	vPossible * emptySquare = (int *) malloc(n*n*sizeof(vPossible));
	vAnswer * h_initial_sudoku = (int *) malloc(n*n*sizeof(vAnswer));


	int emptyCount = 0;

	//load inital sudoku and count of possible values for empty square
	for(int i =0 ; i < n*n; i++){
		if (h_unsolved[i].isLocked == 0){ //square is open
			int count = 0;
			for(int j = 0; j < n; j++){
				if(h_unsolved[i].possValues[j] != '\0'){
					count ++;
				}
			}
			emptySquare[i] = count;
			h_initial_sudoku[i] = 0;
			//emptyCount ++;
		}else{
			emptySquare[i] = 0;
			h_initial_sudoku[i] = h_unsolved[i].value;
		}
	}


	unsigned long int combosize = sizeof(int) * 81 * 81 *81 *81*9;

	//ERROR_CHECK(cudaMalloc((void**) &d_solved, memsize) );
	//ERROR_CHECK( cudaMemcpy(d_unsolved, h_unsolved, memsize, cudaMemcpyHostToDevice) );

	//vAnswer *h_newCombo = (int *)malloc(combosize);
	//vAnswer *h_oldCombo;


	vAnswer *d_newCombo;
	vAnswer *d_oldCombo;
	vPossible *d_possValues;

	ERROR_CHECK(cudaMalloc((void**)&d_newCombo, combosize));
	ERROR_CHECK(cudaMalloc((void**)&d_oldCombo, combosize));
	ERROR_CHECK(cudaMalloc((void**)&d_possValues, sizeof(int)*n));

	//initally load the intial sudoku into the old Combo
	ERROR_CHECK(cudaMemcpy(d_oldCombo, h_initial_sudoku, sizeof(vAnswer)*n*n, cudaMemcpyHostToDevice));
	//ERROR_CHECK(cudaMemcpy(d_oldCombo, puzzle, sizeof(vAnswer)*81, cudaMemcpyHostToDevice));


	int explore = 1;
	int *e = &explore;

  //int emptyCount = 0;
	//int thread;
	//int block;

	for(int i = 0; i < 3; i++){

		if(h_unsolved[i].isLocked == 0){ //open square
		//if(puzzle[i] == 0){
			//emptyCount ++;
			int num_possible = emptySquare[i];
			ERROR_CHECK(cudaMemcpy(d_possValues, h_unsolved[i].possValues, sizeof(vPossible)*num_possible, cudaMemcpyHostToDevice));
			//ERROR_CHECK(cudaMemcpy(d_possValues, poss, sizeof(vPossible)*9, cudaMemcpyHostToDevice));

			//set thread and block size later
			//explore should be the thread nums
			if(emptyCount % 2 ==1){

				//thread
				//block

				bfs<<<256,1024>>>(explore, d_newCombo, d_oldCombo, d_possValues, num_possible, i);
				//bfsKerkel<<<256,1024>>>(explore, d_newCombo, d_oldCombo, d_possValues, 9, i);

				ERROR_CHECK(cudaMemcpyFromSymbol(e, solution_num, sizeof(int)));
				printf("solution_num at iteration %d: %d\n", i, explore);

				reset<<<1,1>>>();

			}else{

				//thread
				//block

				bfs<<<256,1024>>>(explore, d_oldCombo, d_newCombo, d_possValues, num_possible, i);
				//bfsKerkel<<<256,1024>>>(explore, d_oldCombo, d_newCombo, d_possValues, 9, i);
				ERROR_CHECK(cudaMemcpyFromSymbol(e, solution_num, sizeof(int)));
				printf("solution_num at iteration %d: %d\n", i, explore);

				reset<<<1,1>>>();
			}

		}
	}




/*
	bfs<<<blocksPerGrid, threadsPerBlock>>>(d_unsolved);

	ERROR_CHECK( cudaPeekAtLastError() );
  ERROR_CHECK( cudaDeviceSynchronize() );
*/


}
#endif
