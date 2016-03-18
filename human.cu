// human approach to solving Sudoku
// requires fully populated arrays of possValues

#ifndef HUMAN_H
#define HUMAN_H


__device__
int findLocalBlockIdx(int tid) {
	int blockRow = tid/27; // used to be floor
	int col = tid%9;
	int blockCol;
	
	if (col<3)
		blockCol = 0;	
	else if (col<6) 
		blockCol = 1;
	else 
		blockCol = 2;

	int starter = (blockRow*27) + (blockCol*3);

	int difference = tid - starter;

	if (difference==0)
		return 0;
	else if (difference==1)
		return 1;
	else if (difference==2)
		return 2;
	else if (difference==9)
		return 3;
	else if (difference==10)
		return 4;
	else if (difference==11)
		return 5;
	else if (difference==18)
		return 6;
	else if (difference==19)
		return 7;
	else
		return 8;

}



__global__ void human(Square* d_board, int n) {

	__shared__ Square s_board[81];

	if (threadIdx.x == 0) {
		// initialize shared memory
		for (int i = 0; i<(n*n); i++) {
			s_board[i].value = d_board[i].value;
			s_board[i].isLocked = d_board[i].isLocked;

			for (int j=0; j<n; j++) 
				s_board[i].possValues[j] = d_board[i].possValues[j];

		}
	}

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int points = 0; // for keeping track of work done
	int numPossValues;

	if ( (tid<(n*n)) && s_board[tid].isLocked!=-1) {
		// enter the if statement if the thread is a valid Square
		// and if the Square we're looking at is NOT locked (-1)

		// first, check if only one option in possValues array
			//numPossValues = sizeof(s_board[tid].possValues) / sizeof(int);
			for (int k=0; k<n; k++) {
				if (s_board[tid].possValues[k] != 0)
					numPossValues++;
			}

		if (numPossValues==1) {
			// only 1 number in possValues array
			s_board[tid].value = s_board[tid].possValues[0];
			s_board[tid].isLocked = -1;
			points++;
		}
		
		Square localRow[9];
		Square localCol[9];
		Square localBlock[9];
	
		getRow(tid, s_board, localRow);
		getCol(tid, s_board, localCol);
		getBlock(tid, s_board, localBlock);


		int num, nocheck, onlyOne;
		// check if each number can only be in this Square for row/col/block
		for (int i=0; i<9; i++) {
			// cycle through all values in possValues array
			// if any of row/col/block has no other Squares with curVal in possValues
				// that value must be the Square's locked value
			// ex: the first value in tid.possValues is a 4
				// there are two 4's cutting off the other two columns for this block
				// a 4 cuts off one of the rows in this block
				// and the Square just above tid is already locked
					// i.e., s_board[tid-9].isLocked = -1;	
		
			
			num = s_board[tid].possValues[i];
	
			// first, make sure we're looking at a valid int
				if (num==NULL)
					break;

			// now check for num in the possValues arrays for all Squares in the row
			// if we see the number, break and start checking col,
			// otherwise set the value to num and lock it
			// just make sure you don't check against the current square, otherwise never win!
				nocheck = tid%9; // for row, we don't want to check the column tid is in
				for (int j=0; j<n; j++) {
					if (j!=nocheck && localRow[j].isLocked!=-1) {
						// skip checking current tid Square
						// skip Squares that are locked as a precaution
							// since we don't clear possValues after locking

						// look for num in localRow[j].possValues, using device function
						onlyOne = isPossibleNum(num, localRow[j].possValues);	
						if (onlyOne!=-1)
							break;
					}
					// if you get here, means num was not a possValue in any other Square in row
					// so set value and lock it
					s_board[tid].value = num;
					s_board[tid].isLocked = -1;
					points++;
				}

			// now do the same for the column
				nocheck = floor(tid/9); // for col, we don't check the row we're in

				for (int j=0; j<n; j++) {
					if (j!=nocheck && localCol[j].isLocked!=-1) {

						// look for num in localRow[j].possValues, using device function
						onlyOne = isPossibleNum(num, localCol[j].possValues);	
						if (onlyOne!=-1)
							break;
					}
					// if you get here, means num was not a possValue in any other Square in col
					// so set value and lock it
					s_board[tid].value = num;
					s_board[tid].isLocked = -1;
					points++;
				}

			// now do again for block
				nocheck = findLocalBlockIdx(tid);

				for (int j=0; j<n; j++) {
					if (j!=nocheck && localBlock[j].isLocked!=-1) {

						// look for num in localRow[j].possValues, using device function
						onlyOne = isPossibleNum(num, localBlock[j].possValues);	
						if (onlyOne!=-1)
							break;
					}
					// if you get here, means num was not a possValue in any other Square in col
					// so set value and lock it
					s_board[tid].value = num;
					s_board[tid].isLocked = -1;
					points++;
				}

		}

__syncthreads();

	// copy back from shared mem to global mem
	if (threadIdx.x == 0) {
		for (int i=0; i<(n*n); i++) {
			d_board[i].value = s_board[i].value;
			d_board[i].isLocked = s_board[i].isLocked;
	
			if (s_board[i].isLocked!=-1) {
		
				for (int j=0; j<n; j++) {
					d_board[i].possValues[j] = s_board[i].possValues[j];
				}
			}
		}
	}

}

#endif 
