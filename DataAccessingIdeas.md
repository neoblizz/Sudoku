truct Square ( int value, 		//not needed if we always
check len(possValues)
					//else set to 0 for all
					//Squares and set when lock
					//so we could really just skip
					//the lock if just check =0

		bool isLocked,		//may want to replace with
int, so we can
					//guess and keep track of what
					//is locked/guessed

		int possValues[9]	//for keeping track of what
values this Square could be
					//based on locked Squares in
					//local row/col/block
)


Square board[81];

// row is board[i + 9*rowOffset] through board[i+8 + 9*rowOffset]

// col is board[i*9 + colOffset] for (i = 0; i<9; i++)

// block is:  (we are looking at board[n])
	-if (n%9 < 3)
		blockCol = 0; //left side
	-else if (n%9 < 6)
		blockCol = 1; //middle
	-else
		blockCol = 2; //right

	
	-if (n<27)
		blockRow =0; //top
	-else if (n<54)
		blockRow = 1; //middle
	-else
		blockRow = 2; //bottom
	
	
	//can use this to determine which block we’re in, so know
	//exactly which Square to start with

	starter = blockRow*27 + blockCol*3;
	 
	Square block[9] = (board[starter], board[starter+1],
board[starter+2],
				board[starter+9], board[starter+10],
board[starter+11],
				board[starter+18], board[starter+19],
board[starter+20]);

